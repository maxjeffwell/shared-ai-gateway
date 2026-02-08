import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { createHash } from 'crypto';
import Redis from 'ioredis';
import promClient from 'prom-client';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { initProducer, sendAIEvent, closeKafka, isKafkaConnected } from './kafka/producer.js';
import { createAIEvent, createAIErrorEvent } from './kafka/AIEvent.js';

dotenv.config();

// =============================================================================
// PROMETHEUS METRICS
// =============================================================================
promClient.collectDefaultMetrics({ prefix: 'gateway_' });

const requestCounter = new promClient.Counter({
  name: 'gateway_requests_total',
  help: 'Total requests by backend and endpoint',
  labelNames: ['backend', 'endpoint', 'status']
});

const requestDuration = new promClient.Histogram({
  name: 'gateway_request_duration_seconds',
  help: 'Request duration by backend',
  labelNames: ['backend', 'endpoint'],
  buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60]
});

const fallbackCounter = new promClient.Counter({
  name: 'gateway_fallback_total',
  help: 'Fallback events from one tier to another',
  labelNames: ['from_tier', 'to_tier', 'reason']
});

const cacheCounter = new promClient.Counter({
  name: 'gateway_cache_total',
  help: 'Cache hits and misses',
  labelNames: ['result']
});

const backendGauge = new promClient.Gauge({
  name: 'gateway_backend_healthy',
  help: 'Backend health status (1=healthy, 0=unhealthy)',
  labelNames: ['backend']
});

// =============================================================================
// REDIS CACHE CONFIGURATION
// =============================================================================
const REDIS_URL = process.env.REDIS_URL || 'redis://redis:6379';
const CACHE_ENABLED = process.env.CACHE_ENABLED !== 'false';
const CACHE_TTL = parseInt(process.env.CACHE_TTL || '3600'); // 1 hour default

let redis = null;
if (CACHE_ENABLED) {
  try {
    redis = new Redis(REDIS_URL, {
      maxRetriesPerRequest: 3,
      retryDelayOnFailover: 100,
      lazyConnect: true
    });
    redis.on('error', (err) => console.warn('[Redis] Connection error:', err.message));
    redis.on('connect', () => console.log('[Redis] Connected to', REDIS_URL));
  } catch (err) {
    console.warn('[Redis] Failed to initialize:', err.message);
  }
}

// =============================================================================
// KAFKA EVENT PRODUCER
// =============================================================================
if (process.env.KAFKA_ENABLED !== 'false') {
  initProducer().catch(() => {
    console.warn('[Kafka] Producer unavailable, events will not be published');
  });
}

/** Emit an AI event to Kafka (fire-and-forget). */
function emitAIEvent(endpoint, app, result, startTime) {
  sendAIEvent(createAIEvent({ endpoint, app, result, latencyMs: Date.now() - startTime }));
}

/** Emit an error event to Kafka (fire-and-forget). */
function emitAIError(endpoint, app, error, startTime) {
  sendAIEvent(createAIErrorEvent({ endpoint, app, error, latencyMs: Date.now() - startTime }));
}

/**
 * Generate cache key from prompt and options
 */
function getCacheKey(prompt, options = {}) {
  const data = JSON.stringify({ prompt, ...options });
  return 'ai:' + createHash('sha256').update(data).digest('hex').substring(0, 32);
}

/**
 * Get cached response
 */
async function getFromCache(key) {
  if (!redis || !CACHE_ENABLED) return null;
  try {
    const cached = await redis.get(key);
    if (cached) {
      console.log(`[Cache] HIT: ${key.substring(0, 20)}...`);
      cacheCounter.inc({ result: 'hit' });
      return JSON.parse(cached);
    }
    cacheCounter.inc({ result: 'miss' });
  } catch (err) {
    console.warn('[Cache] Get error:', err.message);
  }
  return null;
}

/**
 * Store response in cache
 * Only caches non-empty responses to avoid caching failures
 */
async function setInCache(key, value, ttl = CACHE_TTL) {
  if (!redis || !CACHE_ENABLED) return;

  // Don't cache empty or invalid responses
  if (!value || !value.response || value.response.trim() === '' || value.response.length < 5) {
    console.log(`[Cache] SKIP: Not caching empty/short response`);
    return;
  }

  try {
    await redis.setex(key, ttl, JSON.stringify(value));
    console.log(`[Cache] SET: ${key.substring(0, 20)}... (TTL: ${ttl}s)`);
  } catch (err) {
    console.warn('[Cache] Set error:', err.message);
  }
}

const app = express();
const PORT = process.env.PORT || 8002;

// =============================================================================
// BACKEND CONFIGURATION - 3-Tier Fallback System
// =============================================================================
// Priority 1: HuggingFace Inference API (fast, reliable, cheap)
// Priority 2: VPS CPU (llama-3b - always available fallback)
// Priority 3: RunPod GPU (cloud RTX 4090 - serverless, pay-per-use)
// =============================================================================

// Tier 1: HuggingFace Inference API
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HUGGINGFACE_MODEL = process.env.HF_MODEL || 'mistralai/Mistral-7B-Instruct-v0.3';

// Lens Loop Observability - proxy for LLM tracing (optional)
// When enabled, routes requests through Lens Loop for observability
const LENS_LOOP_PROXY = process.env.LENS_LOOP_PROXY; // e.g., http://host.docker.internal:31300
const LENS_LOOP_PROJECT = process.env.LENS_LOOP_PROJECT || 'lens-loop-project';

// Tier 2: VPS CPU - Llama 3.2 3B via llama.cpp server (always available)
const LOCAL_URL = process.env.LOCAL_URL || 'http://llama-3b-service:8080';
const LOCAL_MODEL = 'llama-3.2-3b-instruct';

// Tier 3: RunPod GPU - Llama 3.1 8B on RTX 4090 (cloud, serverless)
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : null;
const RUNPOD_MODEL = process.env.RUNPOD_MODEL || 'meta-llama/Llama-3.1-8B-Instruct';

// Backend preference: 'auto' (smart fallback), 'huggingface', 'local', 'runpod'
const BACKEND_PREFERENCE = process.env.BACKEND_PREFERENCE || 'auto';

// =============================================================================
// EMBEDDING BACKEND CONFIGURATION - 2-Tier Fallback
// =============================================================================
// Tier 1: VPS CPU Triton (always available)
// Tier 2: Local GPU Triton (GTX 1080 with bge_embeddings via Cloudflare tunnel)
// =============================================================================
const EMBEDDING_PRIMARY_URL = process.env.EMBEDDING_PRIMARY_URL; // e.g., https://embeddings.el-jefe.me
const EMBEDDING_FALLBACK_URL = process.env.EMBEDDING_FALLBACK_URL || 'http://triton-embeddings:8000';
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'bge_embeddings';

// =============================================================================
// ANTHROPIC (CLAUDE) CONFIGURATION - Premium tier for complex reasoning
// =============================================================================
// Use Claude for tasks requiring deep reasoning, K8s analysis, complex debugging
// When Lens Loop is enabled, routes through LiteLLM proxy for observability
// =============================================================================
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-20250514';

// LiteLLM proxy URL - provides OpenAI-compatible interface for Claude
// Used with Lens Loop for observability (Lens Loop → LiteLLM → Anthropic)
// External URL via Cloudflare tunnel so Lens Loop (external) can reach it
const LITELLM_URL = process.env.LITELLM_URL || 'https://litellm.el-jefe.me';

// Initialize Anthropic client (native SDK - fallback when Lens Loop unavailable)
let anthropicClient = null;
if (ANTHROPIC_API_KEY) {
  anthropicClient = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
}

// OpenAI client for Claude via Lens Loop → LiteLLM (initialized after function defs)
let openaiClientForClaude = null;

// =============================================================================
// GROQ CONFIGURATION - Free tier for education/code apps
// =============================================================================
// Fast inference on Llama models, used for code-talk, educationelly apps
// OpenAI-compatible API
// =============================================================================
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
const GROQ_API_URL = 'https://api.groq.com/openai/v1';

// Apps that should use Groq instead of other backends
const GROQ_APPS = ['code-talk', 'educationelly', 'educationelly-graphql', 'bookmarks'];

// Initialize Groq client (OpenAI-compatible)
let groqClient = null;
if (GROQ_API_KEY) {
  groqClient = new OpenAI({
    apiKey: GROQ_API_KEY,
    baseURL: GROQ_API_URL
  });
  console.log(`[Groq] Initialized with model: ${GROQ_MODEL}`);
}

// Health check cache (avoid hammering backends)
const healthCache = {
  huggingface: { status: 'unknown', lastCheck: 0 },
  runpod: { status: 'unknown', lastCheck: 0 },
  local: { status: 'unknown', lastCheck: 0 },
  anthropic: { status: 'unknown', lastCheck: 0 },
  groq: { status: 'unknown', lastCheck: 0 },
  embeddingPrimary: { status: 'unknown', lastCheck: 0 },
  embeddingFallback: { status: 'unknown', lastCheck: 0 }
};
const HEALTH_CACHE_TTL = 30000; // 30 seconds

// Legacy support
const INFERENCE_URL = process.env.INFERENCE_URL || LOCAL_URL;

/**
 * Check if HuggingFace Inference API is configured
 */
function isHuggingFaceConfigured() {
  return !!HUGGINGFACE_API_KEY;
}

/**
 * Check if Lens Loop observability proxy is configured
 */
function isLensLoopConfigured() {
  return !!LENS_LOOP_PROXY;
}

// Initialize OpenAI client for Claude via Lens Loop → LiteLLM
// This enables observability by routing through Lens Loop proxy
if (isLensLoopConfigured() && ANTHROPIC_API_KEY) {
  // Route through Lens Loop proxy to LiteLLM
  // URL format: {lens-loop}/openai/{protocol}/{litellm-host}/v1
  const litellmHost = LITELLM_URL.replace(/^https?:\/\//, '');
  const protocol = LITELLM_URL.startsWith('https') ? 'https' : 'http';
  const lensLoopBaseUrl = `${LENS_LOOP_PROXY}/openai/${protocol}/${litellmHost}/v1`;

  openaiClientForClaude = new OpenAI({
    apiKey: 'not-needed', // LiteLLM uses its own API key
    baseURL: lensLoopBaseUrl,
    defaultHeaders: {
      'X-Loop-Project': LENS_LOOP_PROJECT
    }
  });
  console.log(`[Anthropic] Lens Loop enabled via LiteLLM: ${lensLoopBaseUrl}`);
}

// Initialize Claude client with direct LiteLLM for Langfuse observability
// Used when Lens Loop is unavailable - routes directly to LiteLLM → Claude
let claudeClientWithLiteLLM = null;
if (LITELLM_URL && ANTHROPIC_API_KEY) {
  claudeClientWithLiteLLM = new OpenAI({
    apiKey: 'not-needed', // LiteLLM uses its own API key
    baseURL: `${LITELLM_URL}/v1`
  });
  console.log(`[Anthropic] LiteLLM direct enabled for Langfuse: ${LITELLM_URL}/v1`);
}

// Initialize Groq client with Lens Loop for observability
// Routes through Lens Loop → LiteLLM → Groq (same pattern as Claude)
let groqClientWithLensLoop = null;
if (isLensLoopConfigured() && GROQ_API_KEY) {
  // Route through Lens Loop proxy to LiteLLM (which handles Groq)
  const litellmHost = LITELLM_URL.replace(/^https?:\/\//, '');
  const protocol = LITELLM_URL.startsWith('https') ? 'https' : 'http';
  const lensLoopGroqUrl = `${LENS_LOOP_PROXY}/openai/${protocol}/${litellmHost}/v1`;

  groqClientWithLensLoop = new OpenAI({
    apiKey: 'not-needed', // LiteLLM uses its own API key
    baseURL: lensLoopGroqUrl,
    defaultHeaders: {
      'X-Loop-Project': LENS_LOOP_PROJECT
    }
  });
  console.log(`[Groq] Lens Loop enabled via LiteLLM: ${lensLoopGroqUrl}`);
}

// Initialize Groq client with direct LiteLLM for Langfuse observability
// Used when Lens Loop is unavailable - routes directly to LiteLLM → Groq
let groqClientWithLiteLLM = null;
if (LITELLM_URL && GROQ_API_KEY) {
  groqClientWithLiteLLM = new OpenAI({
    apiKey: 'not-needed', // LiteLLM uses its own API key
    baseURL: `${LITELLM_URL}/v1`
  });
  console.log(`[Groq] LiteLLM direct enabled for Langfuse: ${LITELLM_URL}/v1`);
}

/**
 * Check if RunPod is configured
 */
function isRunPodConfigured() {
  return !!(RUNPOD_API_KEY && RUNPOD_ENDPOINT_ID);
}

/**
 * Check if Anthropic (Claude) is configured
 */
function isAnthropicConfigured() {
  return !!ANTHROPIC_API_KEY && !!anthropicClient;
}

/**
 * Check if Groq is configured
 */
function isGroqConfigured() {
  return !!GROQ_API_KEY && !!groqClient;
}

/**
 * Check if an app should use Groq backend
 */
function shouldUseGroq(appName) {
  return isGroqConfigured() && GROQ_APPS.includes(appName);
}

/**
 * Quick health check with caching (avoids hammering backends)
 */
async function checkBackendHealth(backend) {
  const now = Date.now();
  const cache = healthCache[backend];

  // Return cached status if fresh
  if (cache && (now - cache.lastCheck) < HEALTH_CACHE_TTL) {
    return cache.status === 'healthy';
  }

  try {
    let url, timeout = 3000;

    switch (backend) {
      case 'huggingface':
        if (!isHuggingFaceConfigured()) return false;
        // HuggingFace doesn't have a dedicated health endpoint, check if configured
        healthCache[backend] = { status: 'healthy', lastCheck: now };
        return true;
      case 'runpod':
        if (!isRunPodConfigured()) return false;
        url = `${RUNPOD_BASE_URL}/health`;
        break;
      case 'local':
        url = `${LOCAL_URL}/health`;
        break;
      case 'anthropic':
        // Anthropic doesn't have a health endpoint, just check if configured
        if (!isAnthropicConfigured()) return false;
        healthCache[backend] = { status: 'healthy', lastCheck: now };
        return true;
      default:
        return false;
    }

    const options = {
      method: 'GET',
      signal: AbortSignal.timeout(timeout)
    };

    if (backend === 'runpod') {
      options.headers = { 'Authorization': `Bearer ${RUNPOD_API_KEY}` };
    }

    const response = await fetch(url, options);
    const healthy = response.ok;

    healthCache[backend] = { status: healthy ? 'healthy' : 'unhealthy', lastCheck: now };
    return healthy;
  } catch (error) {
    healthCache[backend] = { status: 'unavailable', lastCheck: now };
    return false;
  }
}

/**
 * Call HuggingFace Inference API
 * Fast, reliable inference for open-source models
 */
async function callHuggingFace(messages, options = {}) {
  if (!isHuggingFaceConfigured()) {
    throw new Error('HuggingFace not configured');
  }

  const { maxTokens = 1024, temperature = 0.7 } = options;

  // Convert messages to a single prompt for text generation
  let prompt = '';
  for (const msg of messages) {
    if (msg.role === 'system') {
      prompt += `<s>[INST] <<SYS>>\n${msg.content}\n<</SYS>>\n\n`;
    } else if (msg.role === 'user') {
      prompt += `${msg.content} [/INST]`;
    } else if (msg.role === 'assistant') {
      prompt += `${msg.content}</s><s>[INST] `;
    }
  }

  console.log(`[HuggingFace] Calling ${HUGGINGFACE_MODEL}`);

  const response = await fetch(`https://router.huggingface.co/hf-inference/models/${HUGGINGFACE_MODEL}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${HUGGINGFACE_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: {
        max_new_tokens: maxTokens,
        temperature: temperature,
        return_full_text: false
      }
    }),
    signal: AbortSignal.timeout(60000) // 60s timeout
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HuggingFace error ${response.status}: ${error}`);
  }

  const data = await response.json();

  // HuggingFace returns array of generated texts
  const content = Array.isArray(data)
    ? data[0]?.generated_text || ''
    : data.generated_text || '';

  console.log(`[HuggingFace] ✓ Response received (${content.length} chars)`);

  return {
    response: content.trim(),
    model: HUGGINGFACE_MODEL,
    backend: 'huggingface',
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0
    }
  };
}

/**
 * Call RunPod vLLM serverless endpoint
 * Uses simple prompt format (not OpenAI chat format)
 */
async function callRunPod(messages, options = {}) {
  if (!isRunPodConfigured()) {
    throw new Error('RunPod not configured');
  }

  const {
    maxTokens = 1024,
    temperature = 0.7
  } = options;

  // Convert messages to Llama 3.1 chat template format
  let prompt = '<|begin_of_text|>';
  for (const msg of messages) {
    if (msg.role === 'system') {
      prompt += `<|start_header_id|>system<|end_header_id|>\n\n${msg.content}<|eot_id|>`;
    } else if (msg.role === 'user') {
      prompt += `<|start_header_id|>user<|end_header_id|>\n\n${msg.content}<|eot_id|>`;
    }
  }
  prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n';

  const payload = {
    input: {
      prompt,
      max_tokens: maxTokens,
      temperature,
      stop: ['<|eot_id|>', '<|end_of_text|>']  // Llama 3.1 stop tokens
    }
  };

  console.log(`[RunPod] Calling ${RUNPOD_MODEL} with prompt (${prompt.length} chars)`);

  const response = await fetch(`${RUNPOD_BASE_URL}/runsync`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(120000) // 2 min timeout
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`RunPod API error ${response.status}: ${error}`);
  }

  const result = await response.json();

  if (result.status === 'FAILED') {
    throw new Error(`RunPod job failed: ${result.error || 'Unknown error'}`);
  }

  // Extract response from vLLM format: output[0].choices[0].tokens[0]
  const content = result.output?.[0]?.choices?.[0]?.tokens?.[0]
    || result.output?.text
    || '';

  // Calculate usage from result
  const usage = result.output?.[0]?.usage || null;

  console.log(`[RunPod] ✓ Response received (${content.length} chars)`);

  return {
    response: content.trim(),
    model: RUNPOD_MODEL,
    backend: 'runpod',
    usage
  };
}

/**
 * Call Anthropic Claude API
 * Best for complex reasoning, K8s analysis, debugging
 * Uses OpenAI SDK via Lens Loop → LiteLLM when enabled for observability
 * Falls back to native Anthropic SDK when Lens Loop unavailable
 */
async function callAnthropic(messages, options = {}) {
  if (!isAnthropicConfigured()) {
    throw new Error('Anthropic not configured');
  }

  const startTime = Date.now();
  const { maxTokens = 1024, temperature = 0.7 } = options;

  // Extract system message if present (for native SDK)
  let systemPrompt = '';
  const chatMessages = [];

  for (const msg of messages) {
    if (msg.role === 'system') {
      systemPrompt = msg.content;
    } else {
      chatMessages.push({
        role: msg.role,
        content: msg.content
      });
    }
  }

  console.log(`[Anthropic] Calling ${ANTHROPIC_MODEL} with ${chatMessages.length} messages`);

  // Try Lens Loop → LiteLLM for observability if configured
  if (openaiClientForClaude) {
    try {
      console.log(`[Anthropic] Using Lens Loop → LiteLLM for observability`);

      // OpenAI format includes system message in messages array
      const openaiMessages = systemPrompt
        ? [{ role: 'system', content: systemPrompt }, ...chatMessages]
        : chatMessages;

      const response = await openaiClientForClaude.chat.completions.create({
        model: 'claude-sonnet', // LiteLLM model alias
        messages: openaiMessages,
        max_tokens: maxTokens,
        temperature
      });

      const content = response.choices[0]?.message?.content || '';
      const usage = {
        prompt_tokens: response.usage?.prompt_tokens,
        completion_tokens: response.usage?.completion_tokens
      };

      const durationMs = Date.now() - startTime;
      console.log(`[Anthropic] ✓ Response via Lens Loop (${content.length} chars) in ${durationMs}ms - traced in project: ${LENS_LOOP_PROJECT}`);

      return {
        response: content.trim(),
        model: ANTHROPIC_MODEL,
        backend: 'anthropic (traced)',
        usage
      };
    } catch (lensLoopError) {
      console.warn(`[Anthropic] Lens Loop error: ${lensLoopError.message}, trying LiteLLM direct...`);
    }
  }

  // Try direct LiteLLM for Langfuse observability (when Lens Loop unavailable)
  if (claudeClientWithLiteLLM) {
    try {
      console.log(`[Anthropic] Using LiteLLM direct for Langfuse observability`);

      const openaiMessages = systemPrompt
        ? [{ role: 'system', content: systemPrompt }, ...chatMessages]
        : chatMessages;

      const response = await claudeClientWithLiteLLM.chat.completions.create({
        model: 'claude-sonnet', // LiteLLM model alias
        messages: openaiMessages,
        max_tokens: maxTokens,
        temperature
      });

      const content = response.choices[0]?.message?.content || '';
      const usage = {
        prompt_tokens: response.usage?.prompt_tokens,
        completion_tokens: response.usage?.completion_tokens
      };

      const durationMs = Date.now() - startTime;
      console.log(`[Anthropic] ✓ Response via LiteLLM/Langfuse (${content.length} chars) in ${durationMs}ms`);

      return {
        response: content.trim(),
        model: ANTHROPIC_MODEL,
        backend: 'anthropic (langfuse)',
        usage
      };
    } catch (litellmError) {
      console.warn(`[Anthropic] LiteLLM error: ${litellmError.message}, falling back to native SDK`);
    }
  }

  // Fallback to native Anthropic SDK (no tracing)
  console.log(`[Anthropic] Using native Anthropic SDK (no tracing)`);

  const response = await anthropicClient.messages.create({
    model: ANTHROPIC_MODEL,
    max_tokens: maxTokens,
    ...(systemPrompt && { system: systemPrompt }),
    messages: chatMessages
  });

  const content = response.content[0]?.text || '';
  const usage = {
    prompt_tokens: response.usage?.input_tokens,
    completion_tokens: response.usage?.output_tokens
  };

  const durationMs = Date.now() - startTime;

  console.log(`[Anthropic] ✓ Response received (${content.length} chars) in ${durationMs}ms`);

  return {
    response: content.trim(),
    model: ANTHROPIC_MODEL,
    backend: 'anthropic',
    usage
  };
}

/**
 * Call Groq API backend (OpenAI-compatible, free tier)
 * Used for code-talk, educationelly, educationelly-graphql apps
 * Routes through Lens Loop for observability when configured
 */
async function callGroq(messages, options = {}) {
  const { maxTokens = 2048, temperature = 0.7 } = options;

  if (!groqClient) {
    throw new Error('Groq client not initialized - missing GROQ_API_KEY');
  }

  console.log(`[Groq] Calling ${GROQ_MODEL} with ${messages.length} messages`);

  const startTime = Date.now();

  // Try Lens Loop → LiteLLM for observability if configured
  if (groqClientWithLensLoop) {
    try {
      console.log(`[Groq] Using Lens Loop → LiteLLM for observability`);

      const response = await groqClientWithLensLoop.chat.completions.create({
        model: 'groq-llama', // LiteLLM model alias for Groq
        messages,
        max_tokens: maxTokens,
        temperature
      });

      const content = response.choices?.[0]?.message?.content || '';
      const usage = response.usage || {};
      const durationMs = Date.now() - startTime;

      console.log(`[Groq] ✓ Response via Lens Loop (${content.length} chars) in ${durationMs}ms - traced in project: ${LENS_LOOP_PROJECT}`);

      return {
        response: content.trim(),
        model: GROQ_MODEL,
        backend: 'groq (traced)',
        usage
      };
    } catch (lensLoopError) {
      console.warn(`[Groq] Lens Loop error: ${lensLoopError.message}, trying LiteLLM direct...`);
    }
  }

  // Try direct LiteLLM for Langfuse observability (when Lens Loop unavailable)
  if (groqClientWithLiteLLM) {
    try {
      console.log(`[Groq] Using LiteLLM direct for Langfuse observability`);

      const response = await groqClientWithLiteLLM.chat.completions.create({
        model: 'groq-llama', // LiteLLM model alias for Groq
        messages,
        max_tokens: maxTokens,
        temperature
      });

      const content = response.choices?.[0]?.message?.content || '';
      const usage = response.usage || {};
      const durationMs = Date.now() - startTime;

      console.log(`[Groq] ✓ Response via LiteLLM/Langfuse (${content.length} chars) in ${durationMs}ms`);

      return {
        response: content.trim(),
        model: GROQ_MODEL,
        backend: 'groq (langfuse)',
        usage
      };
    } catch (litellmError) {
      console.warn(`[Groq] LiteLLM error: ${litellmError.message}, falling back to direct API`);
    }
  }

  // Fallback to direct Groq API
  console.log(`[Groq] Using direct Groq API`);

  const response = await groqClient.chat.completions.create({
    model: GROQ_MODEL,
    messages,
    max_tokens: maxTokens,
    temperature
  });

  const content = response.choices?.[0]?.message?.content || '';
  const usage = response.usage || {};
  const durationMs = Date.now() - startTime;

  console.log(`[Groq] ✓ Response received (${content.length} chars) in ${durationMs}ms`);

  return {
    response: content.trim(),
    model: GROQ_MODEL,
    backend: 'groq',
    usage
  };
}

/**
 * Call local Llama 3.2 3B backend (llama.cpp server - OpenAI compatible)
 */
async function callLocal(messages, options = {}) {
  const { maxTokens = 512, temperature = 0.7 } = options;

  console.log(`[Local] Calling ${LOCAL_MODEL} with ${messages.length} messages`);

  const response = await fetch(`${LOCAL_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: LOCAL_MODEL,
      messages,
      max_tokens: maxTokens,
      temperature
    }),
    signal: AbortSignal.timeout(120000) // 2 min timeout for CPU inference
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Local inference failed: ${response.status} - ${error}`);
  }

  const data = await response.json();

  const content = data.choices?.[0]?.message?.content || '';

  console.log(`[Local] ✓ Response received (${content.length} chars)`);

  return {
    response: content,
    model: LOCAL_MODEL,
    backend: 'local',
    usage: data.usage
  };
}

/**
 * Instrumented backend call wrapper - tracks metrics
 */
async function instrumentedCall(backendName, callFn, endpoint = 'inference') {
  const end = requestDuration.startTimer({ backend: backendName, endpoint });
  try {
    const result = await callFn();
    requestCounter.inc({ backend: backendName, endpoint, status: 'success' });
    end();
    return result;
  } catch (error) {
    requestCounter.inc({ backend: backendName, endpoint, status: 'error' });
    end();
    throw error;
  }
}

/**
 * Unified inference function - 3-tier fallback system with Redis caching
 * Priority: Local GPU (free) → RunPod GPU (paid) → VPS CPU (slow but reliable)
 */
async function inference(prompt, options = {}) {
  const { systemPrompt, maxTokens = 512, temperature = 0.7, forceBackend, skipCache = false } = options;

  const backend = forceBackend || BACKEND_PREFERENCE;

  // Check Redis cache first (skip for temperature > 0.5 since responses should vary)
  if (!skipCache && temperature <= 0.5) {
    const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
    const cached = await getFromCache(cacheKey);
    if (cached) {
      console.log(`[Cache] Using cached response for: ${prompt.substring(0, 40)}...`);
      return { ...cached, fromCache: true };
    }
  }

  // Build messages array (all backends use chat format)
  const messages = [];
  if (systemPrompt) {
    messages.push({ role: 'system', content: systemPrompt });
  }
  messages.push({ role: 'user', content: prompt });

  // Force specific backend
  if (backend === 'huggingface') {
    if (!isHuggingFaceConfigured()) {
      throw new Error('HuggingFace requested but not configured');
    }
    return instrumentedCall('huggingface', () => callHuggingFace(messages, { maxTokens, temperature }));
  }

  if (backend === 'runpod') {
    if (!isRunPodConfigured()) {
      throw new Error('RunPod requested but not configured');
    }
    return instrumentedCall('runpod', () => callRunPod(messages, { maxTokens, temperature }));
  }

  if (backend === 'anthropic' || backend === 'claude') {
    if (!isAnthropicConfigured()) {
      // Anthropic not configured, try Groq as fallback
      if (isGroqConfigured()) {
        console.log('[inference] Anthropic not configured, using Groq fallback');
        return instrumentedCall('groq', () => callGroq(messages, { maxTokens, temperature }));
      }
      throw new Error('Anthropic requested but not configured (Groq fallback also unavailable)');
    }
    // Try Anthropic, fallback to Groq on failure
    try {
      return await instrumentedCall('anthropic', () => callAnthropic(messages, { maxTokens, temperature }));
    } catch (anthropicError) {
      if (isGroqConfigured()) {
        console.warn(`[inference] Anthropic failed: ${anthropicError.message}, falling back to Groq`);
        fallbackCounter.inc({ from_tier: 'anthropic', to_tier: 'groq', reason: 'error' });
        return await instrumentedCall('groq', () => callGroq(messages, { maxTokens, temperature }));
      }
      throw anthropicError;
    }
  }

  if (backend === 'local') {
    return instrumentedCall('vps-cpu', () => callLocal(messages, { maxTokens, temperature }));
  }

  // Auto mode: smart 3-tier fallback with health checks
  // Tier 1: HuggingFace (fast, reliable, cheap)
  if (isHuggingFaceConfigured()) {
    const hfHealthy = await checkBackendHealth('huggingface');
    backendGauge.set({ backend: 'huggingface' }, hfHealthy ? 1 : 0);
    if (hfHealthy) {
      try {
        console.log('[auto] Trying HuggingFace (Tier 1)...');
        const hfResult = await instrumentedCall('huggingface', () => callHuggingFace(messages, { maxTokens, temperature }));
        // Cache the result for low-temperature requests
        if (!skipCache && temperature <= 0.5) {
          const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
          await setInCache(cacheKey, hfResult);
        }
        return hfResult;
      } catch (error) {
        console.warn(`[auto] HuggingFace failed: ${error.message}`);
        fallbackCounter.inc({ from_tier: 'huggingface', to_tier: 'vps-cpu', reason: 'error' });
        // Mark as unhealthy to skip on next request
        healthCache.huggingface = { status: 'unavailable', lastCheck: Date.now() };
      }
    } else {
      console.log('[auto] HuggingFace unavailable, skipping Tier 1');
      fallbackCounter.inc({ from_tier: 'huggingface', to_tier: 'vps-cpu', reason: 'unhealthy' });
    }
  }

  // Tier 2: VPS CPU (reliable fallback)
  const localHealthy = await checkBackendHealth('local');
  backendGauge.set({ backend: 'vps-cpu' }, localHealthy ? 1 : 0);
  if (localHealthy) {
    try {
      console.log('[auto] Trying VPS CPU (Tier 2)...');
      const localResult = await instrumentedCall('vps-cpu', () => callLocal(messages, { maxTokens, temperature }));
      // Cache the result for low-temperature requests
      if (!skipCache && temperature <= 0.5) {
        const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
        await setInCache(cacheKey, localResult);
      }
      return localResult;
    } catch (error) {
      console.warn(`[auto] VPS CPU failed: ${error.message}`);
      fallbackCounter.inc({ from_tier: 'vps-cpu', to_tier: 'runpod', reason: 'error' });
    }
  } else {
    console.log('[auto] VPS CPU unavailable, skipping Tier 2');
    fallbackCounter.inc({ from_tier: 'vps-cpu', to_tier: 'runpod', reason: 'unhealthy' });
  }

  // Tier 3: RunPod GPU (paid cloud fallback)
  if (isRunPodConfigured()) {
    try {
      console.log('[auto] Trying RunPod GPU (Tier 3)...');
      const runpodResult = await instrumentedCall('runpod', () => callRunPod(messages, { maxTokens, temperature }));
      // Cache the result for low-temperature requests
      if (!skipCache && temperature <= 0.5) {
        const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
        await setInCache(cacheKey, runpodResult);
      }
      return runpodResult;
    } catch (error) {
      console.warn(`[auto] RunPod failed: ${error.message}`);
      fallbackCounter.inc({ from_tier: 'runpod', to_tier: 'none', reason: 'error' });
    }
  }

  // All backends failed
  throw new Error('All inference backends failed');
}

app.use(cors());
app.use(express.json());

// System prompts for different applications
const SYSTEM_PROMPTS = {
  bookmarks: `You are an AI that generates relevant tags for web bookmarks.
Given a bookmark's title, URL, and description, generate 3-5 concise, lowercase tags.
Output only a JSON array of tags, nothing else. Example: ["javascript", "tutorial", "web-development"]`,

  education: `You are an educational AI assistant that helps create learning content.
Generate clear, concise, and educational responses suitable for students.`,

  code: `You are a code analysis AI that helps developers understand code.
Provide clear technical explanations and insights about code.`,

  flashcard: `You are an AI that generates educational flashcards.
Create question-answer pairs that help students learn effectively.`,

  quiz: `You are an educational quiz generator.
Create clear, fair quiz questions with correct answers.`,

  general: `You are a helpful AI assistant.`,

  // Conversational help for education apps
  educationChat: `You are an educational assistant helping teachers and students with ELL (English Language Learner) education.
You provide helpful, encouraging responses tailored to the user's context.
For teachers: Help with understanding proficiency levels, creating learning materials, and supporting diverse learners.
For students: Provide patient explanations, encourage questions, and adapt to their language level.
Keep responses concise but thorough. Use simple language when appropriate.`,

  // Bookmark description generation
  describe: `You are a helpful assistant that generates concise, informative descriptions for web bookmarks.
Given a URL and title, generate a 1-2 sentence description that captures what the resource is about.
Focus on being accurate and useful. If you're unsure, describe what you can infer from the title and URL.
Output only the description, nothing else.`
};

/**
 * Prometheus metrics endpoint
 */
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', promClient.register.contentType);
  res.end(await promClient.register.metrics());
});

/**
 * Health check endpoint - shows 3-tier backend status
 */
app.get('/health', async (req, res) => {
  // Get Redis cache status
  let cacheStatus = { enabled: CACHE_ENABLED, status: 'disabled' };
  if (redis && CACHE_ENABLED) {
    try {
      await redis.ping();
      cacheStatus = { enabled: true, status: 'connected', ttl: CACHE_TTL };
    } catch (err) {
      cacheStatus = { enabled: true, status: 'disconnected', error: err.message };
    }
  }

  const health = {
    status: 'ok',
    gateway: 'healthy',
    preference: BACKEND_PREFERENCE,
    cache: cacheStatus,
    kafka: {
      enabled: process.env.KAFKA_ENABLED !== 'false',
      connected: isKafkaConnected(),
    },
    observability: {
      lensLoop: {
        enabled: isLensLoopConfigured(),
        proxy: LENS_LOOP_PROXY || 'not configured',
        project: LENS_LOOP_PROJECT,
        description: 'LLM request tracing and analytics'
      }
    },
    backends: {
      // Tier 1: HuggingFace Inference API
      huggingface: {
        tier: 1,
        configured: isHuggingFaceConfigured(),
        model: isHuggingFaceConfigured() ? HUGGINGFACE_MODEL : null,
        status: isHuggingFaceConfigured() ? 'checking...' : 'not_configured',
        description: 'HuggingFace Inference API (fast, reliable)'
      },
      // Tier 2: VPS CPU
      local: {
        tier: 2,
        model: LOCAL_MODEL,
        url: LOCAL_URL,
        status: 'checking...',
        description: 'Llama 3.2 3B on VPS CPU (always available)'
      },
      // Tier 3: RunPod GPU
      runpod: {
        tier: 3,
        configured: isRunPodConfigured(),
        model: isRunPodConfigured() ? RUNPOD_MODEL : null,
        status: isRunPodConfigured() ? 'checking...' : 'not_configured',
        description: 'RTX 4090 via RunPod Serverless (paid fallback)'
      },
      // External: Groq (free tier for education apps)
      groq: {
        tier: 'external',
        configured: isGroqConfigured(),
        model: isGroqConfigured() ? GROQ_MODEL : null,
        apps: GROQ_APPS,
        status: isGroqConfigured() ? 'checking...' : 'not_configured',
        description: 'Groq Cloud - free tier for code-talk, educationelly, bookmarks apps'
      }
    }
  };

  // Check Tier 1: HuggingFace health
  if (isHuggingFaceConfigured()) {
    try {
      // Simple health check - just verify API is reachable
      const response = await fetch(`https://router.huggingface.co/hf-inference/models/${HUGGINGFACE_MODEL}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HUGGINGFACE_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ inputs: 'test', parameters: { max_new_tokens: 1 } }),
        signal: AbortSignal.timeout(5000)
      });
      if (response.ok || response.status === 503) {
        // 503 means model is loading, which is still "available"
        health.backends.huggingface.status = response.status === 503 ? 'loading' : 'healthy';
      } else {
        health.backends.huggingface.status = 'unhealthy';
      }
    } catch (error) {
      health.backends.huggingface.status = 'unavailable';
    }
  }

  // Check Tier 2: VPS CPU health
  try {
    const response = await fetch(`${LOCAL_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    health.backends.local.status = response.ok ? 'healthy' : 'unhealthy';
  } catch (error) {
    health.backends.local.status = 'unavailable';
  }

  // Check Tier 3: RunPod health
  if (isRunPodConfigured()) {
    try {
      const response = await fetch(`${RUNPOD_BASE_URL}/health`, {
        headers: { 'Authorization': `Bearer ${RUNPOD_API_KEY}` },
        signal: AbortSignal.timeout(5000)
      });
      if (response.ok) {
        const data = await response.json();
        health.backends.runpod.status = 'healthy';
        health.backends.runpod.workers = data.workers;
      } else {
        health.backends.runpod.status = 'unhealthy';
      }
    } catch (error) {
      health.backends.runpod.status = 'unavailable';
    }
  }

  // Check Groq health (external API)
  if (isGroqConfigured()) {
    try {
      // Groq uses OpenAI-compatible API - check models endpoint
      const response = await fetch(`${GROQ_API_URL}/models`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` },
        signal: AbortSignal.timeout(5000)
      });
      health.backends.groq.status = response.ok ? 'healthy' : 'unhealthy';
    } catch (error) {
      health.backends.groq.status = 'unavailable';
      health.backends.groq.error = error.message;
    }
  }

  // Add embedding backends to health check
  // Tier 1: VPS CPU Triton (always available), Tier 2: Local GPU Triton (optional)
  health.embedding = {
    vpsCpu: {
      tier: 1,
      model: EMBEDDING_MODEL,
      url: EMBEDDING_FALLBACK_URL,
      status: 'checking...',
      description: 'VPS CPU Triton (always available)'
    },
    localGpu: {
      tier: 2,
      configured: isEmbeddingPrimaryConfigured(),
      model: EMBEDDING_MODEL,
      url: EMBEDDING_PRIMARY_URL || 'not configured',
      status: 'checking...',
      description: 'Local GPU Triton (optional fallback)'
    }
  };

  // Check Tier 1 embedding: VPS CPU Triton
  try {
    const response = await fetch(`${EMBEDDING_FALLBACK_URL}/v2/health/ready`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    health.embedding.vpsCpu.status = response.ok ? 'healthy' : 'unhealthy';
  } catch (error) {
    health.embedding.vpsCpu.status = 'unavailable';
    health.embedding.vpsCpu.error = error.message;
  }

  // Check Tier 2 embedding: Local GPU Triton
  if (isEmbeddingPrimaryConfigured()) {
    try {
      const response = await fetch(`${EMBEDDING_PRIMARY_URL}/v2/health/ready`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      health.embedding.localGpu.status = response.ok ? 'healthy' : 'unhealthy';
    } catch (error) {
      health.embedding.localGpu.status = 'offline';
      health.embedding.localGpu.note = 'Local GPU/tunnel may be down';
    }
  } else {
    health.embedding.localGpu.status = 'not_configured';
  }

  // Determine active embedding backend (Tier 1: VPS CPU, Tier 2: Local GPU)
  if (health.embedding.vpsCpu.status === 'healthy') {
    health.activeEmbeddingBackend = 'vpsCpu';
    health.activeEmbeddingDescription = 'Using VPS CPU Triton (Tier 1)';
  } else if (health.embedding.localGpu.status === 'healthy') {
    health.activeEmbeddingBackend = 'localGpu';
    health.activeEmbeddingDescription = 'Using Local GPU Triton (Tier 2 fallback)';
  } else {
    health.activeEmbeddingBackend = 'none';
    health.activeEmbeddingDescription = 'No embedding backends available!';
  }

  // Add Anthropic (Claude) status to health check
  health.anthropic = {
    configured: isAnthropicConfigured(),
    model: isAnthropicConfigured() ? ANTHROPIC_MODEL : null,
    status: isAnthropicConfigured() ? 'ready' : 'not_configured',
    description: 'Claude API for complex reasoning (K8s analysis, debugging)'
  };

  // Determine active backend (what would be used for next request)
  // Tier order: HuggingFace → VPS CPU → RunPod
  if (health.backends.huggingface.status === 'healthy') {
    health.activeBackend = 'huggingface';
    health.activeDescription = 'Using HuggingFace Inference API (fast, cheap)';
  } else if (health.backends.local.status === 'healthy') {
    health.activeBackend = 'local';
    health.activeDescription = 'Using VPS CPU (slow but reliable)';
  } else if (health.backends.runpod.status === 'healthy') {
    health.activeBackend = 'runpod';
    health.activeDescription = 'Using RunPod RTX 4090 (paid)';
  } else {
    health.activeBackend = 'none';
    health.activeDescription = 'No backends available!';
  }

  // Overall status
  const healthyCount = [
    health.backends.huggingface.status,
    health.backends.local.status,
    health.backends.runpod.status
  ].filter(s => s === 'healthy').length;

  health.status = healthyCount >= 2 ? 'ok' : healthyCount === 1 ? 'degraded' : 'critical';

  res.json(health);
});

/**
 * POST /api/ai/generate
 * General-purpose text generation
 *
 * Body:
 * {
 *   "prompt": "Your prompt here",
 *   "app": "bookmarks|education|code|flashcard|quiz|general",
 *   "maxTokens": 512,
 *   "temperature": 0.7,
 *   "backend": "auto|local|runpod" (optional)
 * }
 */
app.post('/api/ai/generate', async (req, res) => {
  const startTime = Date.now();
  try {
    const {
      prompt,
      app = 'general',
      maxTokens = 512,
      temperature = 0.7,
      systemPrompt,
      backend
    } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const system = systemPrompt || SYSTEM_PROMPTS[app] || SYSTEM_PROMPTS.general;

    console.log(`[${app}] Generating: ${prompt.substring(0, 60)}...`);

    let result;

    // Route to Groq for specific apps (free tier)
    if (shouldUseGroq(app) && !backend) {
      console.log(`[${app}] Routing to Groq backend`);
      const messages = [];
      if (system) {
        messages.push({ role: 'system', content: system });
      }
      messages.push({ role: 'user', content: prompt });
      result = await instrumentedCall('groq', () => callGroq(messages, { maxTokens, temperature }));
    } else {
      result = await inference(prompt, {
        systemPrompt: system,
        maxTokens,
        temperature,
        forceBackend: backend
      });
    }

    console.log(`[${app}] ✓ Generation complete via ${result.backend} (${result.response.length} chars)`);
    emitAIEvent('generate', app, result, startTime);

    res.json({
      success: true,
      app,
      response: result.response,
      model: result.model,
      backend: result.backend,
      usage: result.usage
    });

  } catch (error) {
    console.error('Generate error:', error.message);
    emitAIError('generate', req.body?.app, error, startTime);
    res.status(500).json({
      error: 'Generation failed',
      message: error.message
    });
  }
});

/**
 * POST /api/ai/tags
 * Generate tags for bookmarks (hybrid approach: keyword extraction + optional AI)
 *
 * Body:
 * {
 *   "title": "Page title",
 *   "url": "https://example.com",
 *   "description": "Optional description",
 *   "useAI": false  // Optional: set to true for AI enhancement (slower)
 * }
 */
app.post('/api/ai/tags', async (req, res) => {
  const startTime = Date.now();
  try {
    const { title, url, description = '', useAI = false } = req.body;

    if (!title || !url) {
      return res.status(400).json({ error: 'Title and URL are required' });
    }

    // Fast keyword extraction (instant response)
    if (!useAI) {
      const tags = extractKeywordTags(title, url, description);
      console.log(`[bookmarks] ✓ Extracted ${tags.length} keyword tags:`, tags);

      return res.json({
        success: true,
        tags,
        title,
        url,
        method: 'keyword-extraction',
        speed: 'instant'
      });
    }

    // AI-enhanced generation (slower, optional)
    console.log(`[bookmarks] Using AI to generate tags for: ${title.substring(0, 50)}...`);

    try {
      const result = await inference(
        `Generate 3-5 relevant tags for this bookmark:\nTitle: ${title}\nURL: ${url}\n${description ? `Description: ${description}` : ''}\n\nOutput only comma-separated lowercase tags, nothing else.`,
        { systemPrompt: SYSTEM_PROMPTS.bookmarks, maxTokens: 50, temperature: 0.3 }
      );

      const tags = parseTags(result.response, title);
      console.log(`[bookmarks] ✓ AI generated ${tags.length} tags via ${result.backend}:`, tags);
      emitAIEvent('tags', 'bookmarks', result, startTime);

      return res.json({
        success: true,
        tags,
        title,
        url,
        method: 'ai-generation',
        backend: result.backend,
        model: result.model
      });
    } catch (aiError) {
      // Fallback to keyword extraction on AI failure
      const tags = extractKeywordTags(title, url, description);
      console.log(`[bookmarks] AI failed, using keyword fallback:`, tags);

      return res.json({
        success: true,
        tags,
        title,
        url,
        method: 'keyword-extraction-fallback',
        warning: aiError.message
      });
    }
  } catch (error) {
    console.error('Tag generation error:', error.message);

    // Fallback to keyword extraction on error
    try {
      const tags = extractKeywordTags(req.body.title, req.body.url, req.body.description || '');
      return res.json({
        success: true,
        tags,
        method: 'keyword-extraction-fallback',
        warning: error.message
      });
    } catch (fallbackError) {
      return res.status(500).json({
        error: 'Tag generation failed',
        message: error.message
      });
    }
  }
});

/**
 * POST /api/ai/explain-code
 * Explain code snippet
 *
 * Body:
 * {
 *   "code": "const x = 5;",
 *   "language": "javascript",
 *   "backend": "auto|local|runpod" (optional)
 * }
 */
app.post('/api/ai/explain-code', async (req, res) => {
  const startTime = Date.now();
  try {
    const { code, language = 'unknown', backend } = req.body;

    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }

    console.log(`[code] Explaining ${language} code (${code.length} chars)`);

    const result = await inference(
      `Explain this ${language} code concisely:\n\n\`\`\`${language}\n${code}\n\`\`\``,
      {
        systemPrompt: SYSTEM_PROMPTS.code,
        maxTokens: 600,
        temperature: 0.3,
        forceBackend: backend
      }
    );

    console.log(`[code] ✓ Explanation generated via ${result.backend}`);
    emitAIEvent('explain-code', 'code-talk', result, startTime);

    res.json({
      success: true,
      explanation: result.response,
      language,
      code: code.substring(0, 100) + (code.length > 100 ? '...' : ''),
      model: result.model,
      backend: result.backend
    });

  } catch (error) {
    console.error('Code explanation error:', error.message);
    emitAIError('explain-code', 'code-talk', error, startTime);
    res.status(500).json({
      error: 'Code explanation failed',
      message: error.message
    });
  }
});

/**
 * POST /api/ai/flashcard
 * Generate flashcard from content
 *
 * Body:
 * {
 *   "topic": "JavaScript Closures",
 *   "content": "A closure is a function that has access to variables in its outer scope",
 *   "backend": "auto|local|runpod" (optional)
 * }
 */
app.post('/api/ai/flashcard', async (req, res) => {
  const startTime = Date.now();
  try {
    const { topic, content, backend } = req.body;

    if (!topic || !content) {
      return res.status(400).json({ error: 'Topic and content are required' });
    }

    console.log(`[flashcard] Generating for topic: ${topic}`);

    const result = await inference(
      `Create a flashcard for learning.\n\nTopic: ${topic}\nContent: ${content}\n\nGenerate a clear question and answer. Format:\nQuestion: [your question]\nAnswer: [your answer]`,
      {
        systemPrompt: SYSTEM_PROMPTS.flashcard,
        maxTokens: 300,
        temperature: 0.5,
        forceBackend: backend
      }
    );

    // Parse question and answer from response
    const text = result.response;
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

    let question = lines.find(l => l.toLowerCase().startsWith('question:')) || lines[0] || `What is ${topic}?`;
    let answer = lines.find(l => l.toLowerCase().startsWith('answer:')) || lines[1] || content.substring(0, 100);

    question = question.replace(/^Question:\s*/i, '').replace(/^\d+\.\s*/, '');
    answer = answer.replace(/^Answer:\s*/i, '');

    console.log(`[flashcard] ✓ Flashcard generated via ${result.backend}`);
    emitAIEvent('flashcard', 'educationelly', result, startTime);

    res.json({
      success: true,
      topic,
      question,
      answer,
      model: result.model,
      backend: result.backend
    });

  } catch (error) {
    console.error('Flashcard generation error:', error.message);
    emitAIError('flashcard', 'educationelly', error, startTime);
    res.status(500).json({
      error: 'Flashcard generation failed',
      message: error.message
    });
  }
});

/**
 * POST /api/ai/quiz
 * Generate quiz questions
 *
 * Body:
 * {
 *   "topic": "JavaScript Arrays",
 *   "difficulty": "medium",
 *   "count": 3,
 *   "backend": "auto|local|runpod" (optional)
 * }
 */
app.post('/api/ai/quiz', async (req, res) => {
  const startTime = Date.now();
  try {
    const { topic, difficulty = 'medium', count = 3, backend } = req.body;

    if (!topic) {
      return res.status(400).json({ error: 'Topic is required' });
    }

    console.log(`[quiz] Generating ${count} ${difficulty} questions for: ${topic}`);

    const result = await inference(
      `Generate ${count} ${difficulty} difficulty quiz questions about: ${topic}\n\nFormat each question as:\nQ: [question]\nA: [correct answer]`,
      {
        systemPrompt: SYSTEM_PROMPTS.quiz,
        maxTokens: 600,
        temperature: 0.6,
        forceBackend: backend
      }
    );

    const questions = parseQuizQuestions(result.response);

    console.log(`[quiz] ✓ Generated ${questions.length} questions via ${result.backend}`);
    emitAIEvent('quiz', 'educationelly', result, startTime);

    res.json({
      success: true,
      topic,
      difficulty,
      count: questions.length,
      questions,
      model: result.model,
      backend: result.backend
    });

  } catch (error) {
    console.error('Quiz generation error:', error.message);
    emitAIError('quiz', 'educationelly', error, startTime);
    res.status(500).json({
      error: 'Quiz generation failed',
      message: error.message
    });
  }
});

/**
 * Multi-turn chat inference - accepts full messages array
 * Used for conversational endpoints where history matters
 */
async function chatInference(messages, options = {}) {
  const { maxTokens = 512, temperature = 0.7, forceBackend, skipCache = true } = options;

  const backend = forceBackend || BACKEND_PREFERENCE;

  // Force specific backend
  if (backend === 'huggingface') {
    if (!isHuggingFaceConfigured()) {
      throw new Error('HuggingFace requested but not configured');
    }
    return instrumentedCall('huggingface', () => callHuggingFace(messages, { maxTokens, temperature }), 'chat');
  }

  if (backend === 'runpod') {
    if (!isRunPodConfigured()) {
      throw new Error('RunPod requested but not configured');
    }
    return instrumentedCall('runpod', () => callRunPod(messages, { maxTokens, temperature }), 'chat');
  }

  if (backend === 'anthropic' || backend === 'claude') {
    if (!isAnthropicConfigured()) {
      // Anthropic not configured, try Groq as fallback
      if (isGroqConfigured()) {
        console.log('[chat] Anthropic not configured, using Groq fallback');
        return instrumentedCall('groq', () => callGroq(messages, { maxTokens, temperature }), 'chat');
      }
      throw new Error('Anthropic requested but not configured (Groq fallback also unavailable)');
    }
    // Try Anthropic, fallback to Groq on failure
    try {
      return await instrumentedCall('anthropic', () => callAnthropic(messages, { maxTokens, temperature }), 'chat');
    } catch (anthropicError) {
      if (isGroqConfigured()) {
        console.warn(`[chat] Anthropic failed: ${anthropicError.message}, falling back to Groq`);
        fallbackCounter.inc({ from_tier: 'anthropic', to_tier: 'groq', reason: 'error' });
        return await instrumentedCall('groq', () => callGroq(messages, { maxTokens, temperature }), 'chat');
      }
      throw anthropicError;
    }
  }

  if (backend === 'local') {
    return instrumentedCall('vps-cpu', () => callLocal(messages, { maxTokens, temperature }), 'chat');
  }

  // Auto mode: smart 3-tier fallback with health checks
  // Tier 1: HuggingFace (fast, reliable)
  if (isHuggingFaceConfigured()) {
    const hfHealthy = await checkBackendHealth('huggingface');
    backendGauge.set({ backend: 'huggingface' }, hfHealthy ? 1 : 0);
    if (hfHealthy) {
      try {
        console.log('[chat-auto] Trying HuggingFace (Tier 1)...');
        return await instrumentedCall('huggingface', () => callHuggingFace(messages, { maxTokens, temperature }), 'chat');
      } catch (error) {
        console.warn(`[chat-auto] HuggingFace failed: ${error.message}`);
        fallbackCounter.inc({ from_tier: 'huggingface', to_tier: 'vps-cpu', reason: 'error' });
        healthCache.huggingface = { status: 'unavailable', lastCheck: Date.now() };
      }
    } else {
      fallbackCounter.inc({ from_tier: 'huggingface', to_tier: 'vps-cpu', reason: 'unhealthy' });
    }
  }

  // Tier 2: VPS CPU (reliable fallback)
  const localHealthy = await checkBackendHealth('local');
  backendGauge.set({ backend: 'vps-cpu' }, localHealthy ? 1 : 0);
  if (localHealthy) {
    try {
      console.log('[chat-auto] Trying VPS CPU (Tier 2)...');
      return await instrumentedCall('vps-cpu', () => callLocal(messages, { maxTokens, temperature }), 'chat');
    } catch (error) {
      console.warn(`[chat-auto] VPS CPU failed: ${error.message}`);
      fallbackCounter.inc({ from_tier: 'vps-cpu', to_tier: 'runpod', reason: 'error' });
    }
  } else {
    fallbackCounter.inc({ from_tier: 'vps-cpu', to_tier: 'runpod', reason: 'unhealthy' });
  }

  // Tier 3: RunPod GPU (paid cloud fallback)
  if (isRunPodConfigured()) {
    try {
      console.log('[chat-auto] Trying RunPod GPU (Tier 3)...');
      return await instrumentedCall('runpod', () => callRunPod(messages, { maxTokens, temperature }), 'chat');
    } catch (error) {
      console.warn(`[chat-auto] RunPod failed: ${error.message}`);
      fallbackCounter.inc({ from_tier: 'runpod', to_tier: 'none', reason: 'error' });
    }
  }

  // All backends failed
  throw new Error('All chat inference backends failed');
}

/**
 * POST /api/ai/chat
 * Multi-turn conversational chat endpoint
 *
 * Body:
 * {
 *   "messages": [
 *     { "role": "user", "content": "Hello" },
 *     { "role": "assistant", "content": "Hi! How can I help?" },
 *     { "role": "user", "content": "Explain ELL proficiency levels" }
 *   ],
 *   "context": {
 *     "app": "educationelly",
 *     "userRole": "teacher",
 *     "gradeLevel": 5,
 *     "ellStatus": "LEP"
 *   },
 *   "maxTokens": 512,
 *   "temperature": 0.7,
 *   "backend": "auto"
 * }
 */
app.post('/api/ai/chat', async (req, res) => {
  const startTime = Date.now();
  try {
    const {
      messages = [],
      context = {},
      maxTokens = 512,
      temperature = 0.7,
      backend
    } = req.body;

    if (!messages || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required' });
    }

    // Check if messages already contain a system prompt (e.g., from POP dashboard)
    const hasExistingSystemPrompt = messages.some(m => m.role === 'system');

    let fullMessages;

    if (hasExistingSystemPrompt) {
      // Use the system prompt from the caller (e.g., POP's K8s expert prompt)
      // This preserves context-aware prompts built by the frontend
      console.log(`[chat] Using caller-provided system prompt`);
      fullMessages = messages;
    } else {
      // Build system prompt based on context (for apps that don't send their own)
      let systemPrompt = SYSTEM_PROMPTS.educationChat;

      if (context.app === 'intervalai' || context.app === 'spaced-repetition') {
        systemPrompt = `You are a learning assistant for a spaced repetition study app.
Help users understand concepts they're struggling with, provide hints without giving away answers,
and encourage effective learning habits. Be concise and supportive.`;
      } else if (context.app === 'code-talk') {
        systemPrompt = SYSTEM_PROMPTS.code;
      } else if (context.userRole === 'teacher') {
        systemPrompt += `\n\nYou are speaking with a teacher. Focus on pedagogical strategies and professional insights.`;
      } else if (context.userRole === 'student') {
        const gradeLevel = context.gradeLevel || 'unknown';
        systemPrompt += `\n\nYou are speaking with a student (grade level: ${gradeLevel}). Use age-appropriate language and be encouraging.`;
      }

      // Add context info if provided
      if (context.ellStatus) {
        systemPrompt += `\nStudent ELL status: ${context.ellStatus}`;
      }
      if (context.nativeLanguage) {
        systemPrompt += `\nStudent native language: ${context.nativeLanguage}`;
      }

      // Build full messages array with system prompt
      fullMessages = [
        { role: 'system', content: systemPrompt },
        ...messages
      ];
    }

    console.log(`[chat] Processing ${messages.length} messages (context: ${context.app || 'general'})`);

    let result;

    // Route to Groq for specific apps (free tier alternative to Claude)
    if (shouldUseGroq(context.app)) {
      console.log(`[chat] Routing ${context.app} to Groq backend`);
      try {
        result = await callGroq(fullMessages, { maxTokens, temperature });
      } catch (groqError) {
        // Fallback to Anthropic if Groq rate limits (429) or fails
        if (groqError.message.includes('429') || groqError.message.includes('rate limit')) {
          console.log(`[chat] Groq rate limited, falling back to Anthropic`);
          fallbackCounter.inc({ from_tier: 'groq', to_tier: 'anthropic', reason: 'rate_limit' });
          result = await callAnthropic(fullMessages, { maxTokens, temperature });
        } else {
          throw groqError;
        }
      }
    } else {
      // Use standard inference chain for other apps
      result = await chatInference(fullMessages, {
        maxTokens,
        temperature,
        forceBackend: backend
      });
    }

    console.log(`[chat] ✓ Response generated via ${result.backend} (${result.response.length} chars)`);
    emitAIEvent('chat', context.app || 'general', result, startTime);

    res.json({
      success: true,
      response: result.response,
      model: result.model,
      backend: result.backend,
      usage: result.usage
    });

  } catch (error) {
    console.error('Chat error:', error.message);
    emitAIError('chat', req.body?.context?.app, error, startTime);
    res.status(500).json({
      error: 'Chat failed',
      message: error.message
    });
  }
});

/**
 * POST /api/ai/describe
 * Generate description for a bookmark URL
 *
 * Body:
 * {
 *   "title": "Page title",
 *   "url": "https://example.com/article",
 *   "existingDescription": "Optional existing description to enhance",
 *   "backend": "auto"
 * }
 */
app.post('/api/ai/describe', async (req, res) => {
  const startTime = Date.now();
  try {
    const { title, url, existingDescription, backend } = req.body;

    if (!title || !url) {
      return res.status(400).json({ error: 'Title and URL are required' });
    }

    console.log(`[describe] Generating description for: ${title.substring(0, 50)}...`);

    // Extract domain and path hints
    let urlHints = '';
    try {
      const urlObj = new URL(url);
      const domain = urlObj.hostname.replace(/^www\./, '');
      const pathParts = urlObj.pathname.split('/').filter(Boolean);
      if (pathParts.length > 0) {
        urlHints = `Domain: ${domain}, Path hints: ${pathParts.slice(0, 3).join('/')}`;
      } else {
        urlHints = `Domain: ${domain}`;
      }
    } catch (e) {
      urlHints = `URL: ${url}`;
    }

    let prompt = `Generate a brief, informative description for this web bookmark:\n\nTitle: ${title}\n${urlHints}`;

    if (existingDescription) {
      prompt += `\n\nExisting description (improve or expand): ${existingDescription}`;
    }

    prompt += '\n\nProvide a 1-2 sentence description:';

    const result = await inference(prompt, {
      systemPrompt: SYSTEM_PROMPTS.describe,
      maxTokens: 150,
      temperature: 0.4,
      forceBackend: backend
    });

    // Clean up the response
    let description = result.response.trim();
    // Remove any leading quotes or formatting
    description = description.replace(/^["']|["']$/g, '').trim();

    console.log(`[describe] ✓ Description generated via ${result.backend}`);
    emitAIEvent('describe', 'bookmarks', result, startTime);

    res.json({
      success: true,
      description,
      title,
      url,
      model: result.model,
      backend: result.backend
    });

  } catch (error) {
    console.error('Description generation error:', error.message);

    // Fallback: generate a simple description from title
    const fallbackDesc = `Resource about ${req.body.title || 'this topic'}.`;

    res.json({
      success: true,
      description: fallbackDesc,
      title: req.body.title,
      url: req.body.url,
      method: 'fallback',
      warning: error.message
    });
  }
});

/**
 * Check if embedding primary backend is configured
 */
function isEmbeddingPrimaryConfigured() {
  return !!EMBEDDING_PRIMARY_URL;
}

/**
 * Check embedding backend health (Triton KServe V2)
 */
async function checkEmbeddingHealth(backend) {
  const now = Date.now();
  const cacheKey = backend === 'primary' ? 'embeddingPrimary' : 'embeddingFallback';
  const cache = healthCache[cacheKey];

  // Return cached status if fresh
  if (cache && (now - cache.lastCheck) < HEALTH_CACHE_TTL) {
    return cache.status === 'healthy';
  }

  const url = backend === 'primary' ? EMBEDDING_PRIMARY_URL : EMBEDDING_FALLBACK_URL;
  if (!url) {
    healthCache[cacheKey] = { status: 'not_configured', lastCheck: now };
    return false;
  }

  try {
    const response = await fetch(`${url}/v2/health/ready`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });

    const healthy = response.ok;
    healthCache[cacheKey] = { status: healthy ? 'healthy' : 'unhealthy', lastCheck: now };
    return healthy;
  } catch (error) {
    healthCache[cacheKey] = { status: 'unavailable', lastCheck: now };
    return false;
  }
}

/**
 * Call Triton embedding server (KServe V2 protocol)
 * Returns embeddings for input text(s)
 */
async function callTritonEmbedding(url, texts) {
  // Ensure texts is an array
  const inputTexts = Array.isArray(texts) ? texts : [texts];

  console.log(`[Embedding] Calling Triton at ${url} with ${inputTexts.length} text(s)`);

  // KServe V2 inference request format for embeddings
  // Shape must be [batch_size, 1] as model expects [-1, 1]
  // Data format: each text wrapped in its own array for the 2D shape
  const response = await fetch(`${url}/v2/models/${EMBEDDING_MODEL}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      inputs: [
        {
          name: 'text',
          shape: [inputTexts.length, 1],
          datatype: 'BYTES',
          data: inputTexts.map(t => [t])
        }
      ]
    }),
    signal: AbortSignal.timeout(30000)
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Triton embedding error ${response.status}: ${error}`);
  }

  const data = await response.json();

  // Extract embeddings from Triton response
  const embeddings = data.outputs?.[0]?.data || data.outputs?.[0]?.contents?.fp32_contents || [];

  // If single text, return single embedding
  if (!Array.isArray(texts)) {
    return {
      embedding: embeddings.slice(0, 1024), // bge-base is 768-dim, bge-large is 1024
      dimensions: embeddings.length / inputTexts.length
    };
  }

  // Return batch embeddings
  const embeddingDim = embeddings.length / inputTexts.length;
  const batchEmbeddings = [];
  for (let i = 0; i < inputTexts.length; i++) {
    batchEmbeddings.push(embeddings.slice(i * embeddingDim, (i + 1) * embeddingDim));
  }

  return {
    embeddings: batchEmbeddings,
    dimensions: embeddingDim
  };
}

/**
 * Unified embedding function - 2-tier fallback
 * Priority: VPS CPU Triton (always available) → Local GPU Triton (optional)
 */
async function getEmbeddings(texts, options = {}) {
  const { forceBackend, skipCache = false } = options;
  const inputTexts = Array.isArray(texts) ? texts : [texts];

  // Check cache for single text requests
  if (!skipCache && inputTexts.length === 1) {
    const cacheKey = getCacheKey(inputTexts[0], { type: 'embedding', model: EMBEDDING_MODEL });
    const cached = await getFromCache(cacheKey);
    if (cached) {
      console.log(`[Embedding] Cache HIT for: ${inputTexts[0].substring(0, 40)}...`);
      return { ...cached, fromCache: true };
    }
  }

  // Force specific backend
  if (forceBackend === 'vps' || forceBackend === 'fallback') {
    const result = await instrumentedCall('vps-cpu-embed', () => callTritonEmbedding(EMBEDDING_FALLBACK_URL, texts), 'embedding');
    return { ...result, backend: 'VPS CPU Triton', model: EMBEDDING_MODEL };
  }

  if (forceBackend === 'local-gpu' || forceBackend === 'primary') {
    if (!isEmbeddingPrimaryConfigured()) {
      throw new Error('Local GPU embedding backend not configured');
    }
    const result = await instrumentedCall('local-gpu-embed', () => callTritonEmbedding(EMBEDDING_PRIMARY_URL, texts), 'embedding');
    return { ...result, backend: 'Local GPU Triton', model: EMBEDDING_MODEL };
  }

  // Auto mode: 2-tier fallback
  // Tier 1: VPS CPU Triton (always available)
  const vpsHealthy = await checkEmbeddingHealth('fallback');
  backendGauge.set({ backend: 'vps-cpu-embed' }, vpsHealthy ? 1 : 0);
  if (vpsHealthy) {
    try {
      console.log('[Embedding] Trying VPS CPU Triton (Tier 1)...');
      const result = await instrumentedCall('vps-cpu-embed', () => callTritonEmbedding(EMBEDDING_FALLBACK_URL, texts), 'embedding');

      // Cache single text results
      if (!skipCache && inputTexts.length === 1) {
        const cacheKey = getCacheKey(inputTexts[0], { type: 'embedding', model: EMBEDDING_MODEL });
        await setInCache(cacheKey, result, 86400); // 24 hour cache for embeddings
      }

      return { ...result, backend: 'VPS CPU Triton (Tier 1)', model: EMBEDDING_MODEL };
    } catch (error) {
      console.warn(`[Embedding] VPS CPU failed: ${error.message}`);
      fallbackCounter.inc({ from_tier: 'vps-cpu-embed', to_tier: 'local-gpu-embed', reason: 'error' });
      healthCache.embeddingFallback = { status: 'unavailable', lastCheck: Date.now() };
    }
  } else {
    console.log('[Embedding] VPS CPU unavailable, skipping Tier 1');
    fallbackCounter.inc({ from_tier: 'vps-cpu-embed', to_tier: 'local-gpu-embed', reason: 'unhealthy' });
  }

  // Tier 2: Local GPU Triton (fallback)
  if (isEmbeddingPrimaryConfigured()) {
    const localGpuHealthy = await checkEmbeddingHealth('primary');
    backendGauge.set({ backend: 'local-gpu-embed' }, localGpuHealthy ? 1 : 0);
    if (localGpuHealthy) {
      try {
        console.log('[Embedding] Falling back to Local GPU Triton (Tier 2)...');
        const result = await instrumentedCall('local-gpu-embed', () => callTritonEmbedding(EMBEDDING_PRIMARY_URL, texts), 'embedding');

        // Cache single text results
        if (!skipCache && inputTexts.length === 1) {
          const cacheKey = getCacheKey(inputTexts[0], { type: 'embedding', model: EMBEDDING_MODEL });
          await setInCache(cacheKey, result, 86400);
        }

        return { ...result, backend: 'Local GPU Triton (Tier 2)', model: EMBEDDING_MODEL };
      } catch (error) {
        console.warn(`[Embedding] Local GPU failed: ${error.message}`);
        healthCache.embeddingPrimary = { status: 'unavailable', lastCheck: Date.now() };
      }
    }
  }

  throw new Error('All embedding backends failed');
}

/**
 * POST /api/ai/embed
 * Generate embeddings for text(s)
 *
 * Body:
 * {
 *   "text": "Single text to embed" OR
 *   "texts": ["Text 1", "Text 2", ...],
 *   "backend": "auto|primary|fallback" (optional)
 * }
 */
app.post('/api/ai/embed', async (req, res) => {
  const startTime = Date.now();
  try {
    const { text, texts, backend } = req.body;

    const input = texts || text;
    if (!input || (Array.isArray(input) && input.length === 0)) {
      return res.status(400).json({ error: 'Text or texts array is required' });
    }

    const isBatch = Array.isArray(input);
    console.log(`[embed] Generating embeddings for ${isBatch ? input.length + ' texts' : 'single text'}`);

    const result = await getEmbeddings(input, { forceBackend: backend });

    console.log(`[embed] ✓ Embeddings generated via ${result.backend} (${result.dimensions} dimensions)`);
    emitAIEvent('embed', 'general', result, startTime);

    res.json({
      success: true,
      ...(isBatch ? { embeddings: result.embeddings } : { embedding: result.embedding }),
      dimensions: result.dimensions,
      model: result.model,
      backend: result.backend,
      fromCache: result.fromCache || false
    });

  } catch (error) {
    console.error('Embedding error:', error.message);
    emitAIError('embed', 'general', error, startTime);
    res.status(500).json({
      error: 'Embedding generation failed',
      message: error.message
    });
  }
});

/**
 * GET /
 * API info
 */
app.get('/', (req, res) => {
  res.json({
    name: 'Shared AI Gateway',
    version: '3.5.0',
    description: '3-Tier LLM + 2-Tier Embedding GPU Fallback System with Anthropic Claude + Redis Caching',
    backends: {
      'tier1_huggingface': {
        configured: isHuggingFaceConfigured(),
        model: isHuggingFaceConfigured() ? HUGGINGFACE_MODEL : null,
        description: 'HuggingFace Inference API (fast, reliable)'
      },
      'tier2_vpsCpu': {
        model: LOCAL_MODEL,
        description: 'Llama 3.2 3B on VPS CPU (always available)'
      },
      'tier3_runpod': {
        configured: isRunPodConfigured(),
        model: isRunPodConfigured() ? RUNPOD_MODEL : null,
        description: 'RTX 4090 via RunPod Serverless (paid cloud fallback)'
      },
      'anthropic': {
        configured: isAnthropicConfigured(),
        model: isAnthropicConfigured() ? ANTHROPIC_MODEL : null,
        description: 'Claude API for complex reasoning (request with backend: "anthropic")'
      }
    },
    preference: BACKEND_PREFERENCE,
    embedding: {
      'tier1_vpsCpu': {
        model: EMBEDDING_MODEL,
        url: EMBEDDING_FALLBACK_URL,
        description: 'VPS CPU Triton (always available)'
      },
      'tier2_localGpu': {
        configured: isEmbeddingPrimaryConfigured(),
        model: EMBEDDING_MODEL,
        url: EMBEDDING_PRIMARY_URL || 'not configured',
        description: 'Local GPU Triton (optional fallback)'
      }
    },
    endpoints: {
      'POST /api/ai/generate': 'General text generation',
      'POST /api/ai/tags': 'Generate bookmark tags (keyword or AI)',
      'POST /api/ai/explain-code': 'Explain code snippets',
      'POST /api/ai/flashcard': 'Generate flashcards',
      'POST /api/ai/quiz': 'Generate quiz questions',
      'POST /api/ai/chat': 'Multi-turn conversational chat (with context)',
      'POST /api/ai/describe': 'Generate bookmark descriptions',
      'POST /api/ai/embed': 'Generate text embeddings (single or batch)',
      'GET /health': 'Health check with backend status',
      'GET /metrics': 'Prometheus metrics endpoint'
    },
    usage: {
      backend_param: 'Add "backend": "huggingface|local|runpod|anthropic|auto" to force a specific backend',
      auto_mode: 'Default "auto" tries backends in order: huggingface → local (VPS CPU) → runpod',
      anthropic_mode: 'Use "backend": "anthropic" or "claude" for complex reasoning tasks (K8s analysis, debugging)'
    }
  });
});

// Utility: Extract keyword tags from bookmark metadata (instant, no AI)
function extractKeywordTags(title, url, description) {
  const tags = new Set();

  // Common stop words to exclude
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'using', 'learn'
  ]);

  // Extract from URL domain
  try {
    const urlObj = new URL(url);
    const domain = urlObj.hostname.replace(/^www\./, '').split('.')[0];
    if (domain && domain.length > 3 && !stopWords.has(domain.toLowerCase())) {
      tags.add(domain.toLowerCase());
    }
  } catch (e) {
    // Invalid URL, skip
  }

  // Extract from title
  const titleWords = title
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter(word =>
      word.length > 3 &&
      !stopWords.has(word) &&
      !/^\d+$/.test(word)  // Exclude pure numbers
    );

  titleWords.forEach(word => tags.add(word));

  // Extract from description
  if (description) {
    const descWords = description
      .toLowerCase()
      .replace(/[^\w\s-]/g, ' ')
      .split(/\s+/)
      .filter(word =>
        word.length > 4 &&  // Longer words from description
        !stopWords.has(word) &&
        !/^\d+$/.test(word)
      );

    descWords.slice(0, 3).forEach(word => tags.add(word));
  }

  // Convert to array and limit to 5-8 tags
  const tagArray = Array.from(tags).slice(0, 8);

  // Ensure at least 3 tags
  if (tagArray.length < 3 && titleWords.length > 0) {
    // Add shorter title words if needed
    const shortTitleWords = title
      .toLowerCase()
      .replace(/[^\w\s-]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2 && !stopWords.has(word));

    shortTitleWords.forEach(word => {
      if (tagArray.length < 5) {
        tagArray.push(word);
      }
    });
  }

  return [...new Set(tagArray)].slice(0, 5);  // Deduplicate and limit to 5
}

// Utility: Parse tags from AI response
function parseTags(text, fallbackTitle) {
  const tags = [];

  try {
    // Try to find JSON array
    const jsonMatch = text.match(/\[([^\]]+)\]/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return parsed.slice(0, 5);
    }

    // Try comma-separated
    const commaSplit = text.split(',').map(t =>
      t.trim().toLowerCase()
        .replace(/['"]/g, '')
        .replace(/^[-•*]\s*/, '')
    );

    if (commaSplit.length > 0) {
      return commaSplit.filter(t => t && t.length > 2).slice(0, 5);
    }

    // Extract words as fallback
    const words = text.match(/\b[a-z]+\b/gi) || [];
    const uniqueWords = [...new Set(words.map(w => w.toLowerCase()))];
    return uniqueWords.filter(w => w.length > 3).slice(0, 5);

  } catch (error) {
    console.warn('Tag parsing fallback:', error.message);
    // Last resort: use title words
    const titleWords = fallbackTitle.match(/\b[a-z]+\b/gi) || [];
    return titleWords.filter(w => w.length > 4).slice(0, 3);
  }
}

// Utility: Parse quiz questions
function parseQuizQuestions(text) {
  const questions = [];
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

  let currentQ = null;

  for (const line of lines) {
    if (line.match(/^Q(\d*)[:\.]?\s*/i)) {
      if (currentQ && currentQ.question) {
        questions.push(currentQ);
      }
      currentQ = { question: line.replace(/^Q(\d*)[:\.]?\s*/i, ''), answer: '' };
    } else if (line.match(/^A(\d*)[:\.]?\s*/i) && currentQ) {
      currentQ.answer = line.replace(/^A(\d*)[:\.]?\s*/i, '');
      questions.push(currentQ);
      currentQ = null;
    }
  }

  if (currentQ && currentQ.question) {
    questions.push(currentQ);
  }

  return questions.slice(0, 5);
}

// Start server
app.listen(PORT, async () => {
  // Try to connect to Redis on startup
  if (redis && CACHE_ENABLED) {
    try {
      await redis.connect();
    } catch (err) {
      // Already connected or will connect lazily
    }
  }

  console.log(`
╔══════════════════════════════════════════════════════════╗
║   Shared AI Gateway v3.5 - LLM + Embeddings + Claude     ║
╠══════════════════════════════════════════════════════════╣
║   Port: ${PORT.toString().padEnd(48)}║
║   Mode: ${BACKEND_PREFERENCE.padEnd(49)}║
║   Cache: ${(CACHE_ENABLED ? `Enabled (TTL: ${CACHE_TTL}s)` : 'Disabled').padEnd(48)}║
║   Redis: ${REDIS_URL.substring(0, 47).padEnd(47)}║
╠══════════════════════════════════════════════════════════╣
║   OBSERVABILITY (Lens Loop)                              ║
╠══════════════════════════════════════════════════════════╣
║   Tracing: ${(isLensLoopConfigured() ? 'Enabled ✓' : 'Disabled').padEnd(45)}║
${isLensLoopConfigured() ? `║   Proxy: ${LENS_LOOP_PROXY.substring(0, 47).padEnd(47)}║\n║   Project: ${LENS_LOOP_PROJECT.padEnd(45)}║\n` : ''}╠══════════════════════════════════════════════════════════╣
║   LLM BACKENDS (3-Tier Fallback)                         ║
╠══════════════════════════════════════════════════════════╣
║   Tier 1: HuggingFace Inference API                      ║
║     Configured: ${(isHuggingFaceConfigured() ? 'Yes' : 'No').padEnd(40)}║
${isHuggingFaceConfigured() ? `║     Model: ${HUGGINGFACE_MODEL.substring(0, 45).padEnd(45)}║\n` : ''}╠══════════════════════════════════════════════════════════╣
║   Tier 2: VPS CPU (Always Available)                     ║
║     URL: ${LOCAL_URL.padEnd(47)}║
║     Model: ${LOCAL_MODEL.padEnd(45)}║
╠══════════════════════════════════════════════════════════╣
║   Tier 3: RunPod GPU (RTX 4090 Serverless)               ║
║     Configured: ${(isRunPodConfigured() ? 'Yes' : 'No').padEnd(40)}║
${isRunPodConfigured() ? `║     Model: ${RUNPOD_MODEL.substring(0, 45).padEnd(45)}║\n` : ''}╠══════════════════════════════════════════════════════════╣
║   ANTHROPIC (Claude - Premium Reasoning)                 ║
╠══════════════════════════════════════════════════════════╣
║   Configured: ${(isAnthropicConfigured() ? 'Yes ✓' : 'No').padEnd(42)}║
${isAnthropicConfigured() ? `║   Model: ${ANTHROPIC_MODEL.padEnd(47)}║\n` : ''}╠══════════════════════════════════════════════════════════╣
║   EMBEDDING BACKENDS (2-Tier Fallback)                   ║
╠══════════════════════════════════════════════════════════╣
║   Tier 1: VPS CPU Triton (Always Available)              ║
║     URL: ${EMBEDDING_FALLBACK_URL.substring(0, 47).padEnd(47)}║
║     Model: ${EMBEDDING_MODEL.padEnd(45)}║
╠══════════════════════════════════════════════════════════╣
║   Tier 2: Local GPU Triton (Optional)                    ║
║     Configured: ${(isEmbeddingPrimaryConfigured() ? 'Yes' : 'No').padEnd(40)}║
${isEmbeddingPrimaryConfigured() ? `║     URL: ${EMBEDDING_PRIMARY_URL.substring(0, 47).padEnd(47)}║\n` : ''}
╚══════════════════════════════════════════════════════════╝
  `);
});

process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down...');
  await closeKafka();
  process.exit(0);
});
