import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { createHash } from 'crypto';
import Redis from 'ioredis';

dotenv.config();

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
      return JSON.parse(cached);
    }
  } catch (err) {
    console.warn('[Cache] Get error:', err.message);
  }
  return null;
}

/**
 * Store response in cache
 */
async function setInCache(key, value, ttl = CACHE_TTL) {
  if (!redis || !CACHE_ENABLED) return;
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
// BACKEND CONFIGURATION - 3-Tier GPU Fallback System
// =============================================================================
// Priority 1: Local GPU (your GTX 1080 via Ollama + Cloudflare tunnel)
// Priority 2: RunPod GPU (cloud RTX 4090 - serverless, pay-per-use)
// Priority 3: VPS CPU (llama-3b - always available fallback)
// =============================================================================

// Tier 1: Local GPU - Ollama via Cloudflare tunnel (optional, can go offline)
const LOCAL_GPU_URL = process.env.LOCAL_GPU_URL; // e.g., https://gpu.yourdomain.com
const LOCAL_GPU_MODEL = process.env.LOCAL_GPU_MODEL || 'llama3.1:8b-instruct-q4_K_M';

// Tier 2: RunPod GPU - Llama 3.1 8B on RTX 4090 (cloud, serverless)
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : null;
const RUNPOD_MODEL = process.env.RUNPOD_MODEL || 'meta-llama/Llama-3.1-8B-Instruct';

// Tier 3: VPS CPU - Llama 3.2 3B via llama.cpp server (always available)
const LOCAL_URL = process.env.LOCAL_URL || 'http://llama-3b-service:8080';
const LOCAL_MODEL = 'llama-3.2-3b-instruct';

// Backend preference: 'auto' (smart fallback), 'local-gpu', 'runpod', 'local'
const BACKEND_PREFERENCE = process.env.BACKEND_PREFERENCE || 'auto';

// Health check cache (avoid hammering backends)
const healthCache = {
  localGpu: { status: 'unknown', lastCheck: 0 },
  runpod: { status: 'unknown', lastCheck: 0 },
  local: { status: 'unknown', lastCheck: 0 }
};
const HEALTH_CACHE_TTL = 30000; // 30 seconds

// Legacy support
const INFERENCE_URL = process.env.INFERENCE_URL || LOCAL_URL;

/**
 * Check if Local GPU (Ollama) is configured
 */
function isLocalGpuConfigured() {
  return !!LOCAL_GPU_URL;
}

/**
 * Check if RunPod is configured
 */
function isRunPodConfigured() {
  return !!(RUNPOD_API_KEY && RUNPOD_ENDPOINT_ID);
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
      case 'localGpu':
        if (!isLocalGpuConfigured()) return false;
        url = `${LOCAL_GPU_URL}/api/tags`; // Ollama health endpoint
        break;
      case 'runpod':
        if (!isRunPodConfigured()) return false;
        url = `${RUNPOD_BASE_URL}/health`;
        break;
      case 'local':
        url = `${LOCAL_URL}/health`;
        break;
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
 * Call Local GPU (Ollama) via Cloudflare tunnel
 * Uses OpenAI-compatible chat format
 */
async function callLocalGpu(messages, options = {}) {
  if (!isLocalGpuConfigured()) {
    throw new Error('Local GPU not configured');
  }

  const { maxTokens = 1024, temperature = 0.7 } = options;

  console.log(`[LocalGPU] Calling ${LOCAL_GPU_MODEL} via Ollama`);

  const response = await fetch(`${LOCAL_GPU_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: LOCAL_GPU_MODEL,
      messages,
      stream: false,
      options: {
        num_predict: maxTokens,
        temperature
      }
    }),
    signal: AbortSignal.timeout(120000) // 2 min timeout
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Local GPU error ${response.status}: ${error}`);
  }

  const data = await response.json();
  const content = data.message?.content || '';

  console.log(`[LocalGPU] ✓ Response received (${content.length} chars)`);

  return {
    response: content.trim(),
    model: LOCAL_GPU_MODEL,
    backend: 'local-gpu',
    usage: {
      prompt_tokens: data.prompt_eval_count,
      completion_tokens: data.eval_count
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
  if (backend === 'local-gpu') {
    if (!isLocalGpuConfigured()) {
      throw new Error('Local GPU requested but not configured');
    }
    return callLocalGpu(messages, { maxTokens, temperature });
  }

  if (backend === 'runpod') {
    if (!isRunPodConfigured()) {
      throw new Error('RunPod requested but not configured');
    }
    return callRunPod(messages, { maxTokens, temperature });
  }

  if (backend === 'local') {
    return callLocal(messages, { maxTokens, temperature });
  }

  // Auto mode: smart 3-tier fallback with health checks
  // Tier 1: Local GPU (free, fastest when available)
  if (isLocalGpuConfigured()) {
    const localGpuHealthy = await checkBackendHealth('localGpu');
    if (localGpuHealthy) {
      try {
        console.log('[auto] Trying Local GPU (Tier 1)...');
        const gpuResult = await callLocalGpu(messages, { maxTokens, temperature });
        // Cache the result for low-temperature requests
        if (!skipCache && temperature <= 0.5) {
          const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
          await setInCache(cacheKey, gpuResult);
        }
        return gpuResult;
      } catch (error) {
        console.warn(`[auto] Local GPU failed: ${error.message}`);
        // Mark as unhealthy to skip on next request
        healthCache.localGpu = { status: 'unavailable', lastCheck: Date.now() };
      }
    } else {
      console.log('[auto] Local GPU unavailable, skipping Tier 1');
    }
  }

  // Tier 2: RunPod GPU (paid, but fast)
  if (isRunPodConfigured()) {
    try {
      console.log('[auto] Trying RunPod GPU (Tier 2)...');
      const runpodResult = await callRunPod(messages, { maxTokens, temperature });
      // Cache the result for low-temperature requests
      if (!skipCache && temperature <= 0.5) {
        const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
        await setInCache(cacheKey, runpodResult);
      }
      return runpodResult;
    } catch (error) {
      console.warn(`[auto] RunPod failed: ${error.message}`);
    }
  }

  // Tier 3: VPS CPU (always available fallback)
  console.log('[auto] Falling back to VPS CPU (Tier 3)...');
  const localResult = await callLocal(messages, { maxTokens, temperature });

  // Cache the result for low-temperature requests
  if (!skipCache && temperature <= 0.5) {
    const cacheKey = getCacheKey(prompt, { systemPrompt, maxTokens, backend });
    await setInCache(cacheKey, localResult);
  }

  return localResult;
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

  general: `You are a helpful AI assistant.`
};

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
    backends: {
      // Tier 1: Local GPU (your GTX 1080)
      localGpu: {
        tier: 1,
        configured: isLocalGpuConfigured(),
        model: isLocalGpuConfigured() ? LOCAL_GPU_MODEL : null,
        url: LOCAL_GPU_URL || 'not configured',
        status: isLocalGpuConfigured() ? 'checking...' : 'not_configured',
        description: 'GTX 1080 via Ollama + Cloudflare tunnel'
      },
      // Tier 2: RunPod GPU
      runpod: {
        tier: 2,
        configured: isRunPodConfigured(),
        model: isRunPodConfigured() ? RUNPOD_MODEL : null,
        status: isRunPodConfigured() ? 'checking...' : 'not_configured',
        description: 'RTX 4090 via RunPod Serverless'
      },
      // Tier 3: VPS CPU
      local: {
        tier: 3,
        model: LOCAL_MODEL,
        url: LOCAL_URL,
        status: 'checking...',
        description: 'Llama 3.2 3B on VPS CPU (always available)'
      }
    }
  };

  // Check Tier 1: Local GPU health
  if (isLocalGpuConfigured()) {
    try {
      const response = await fetch(`${LOCAL_GPU_URL}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      });
      if (response.ok) {
        const data = await response.json();
        health.backends.localGpu.status = 'healthy';
        health.backends.localGpu.models = data.models?.map(m => m.name) || [];
      } else {
        health.backends.localGpu.status = 'unhealthy';
      }
    } catch (error) {
      health.backends.localGpu.status = 'offline';
      health.backends.localGpu.note = 'Your computer may be off or tunnel disconnected';
    }
  }

  // Check Tier 2: RunPod health
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

  // Check Tier 3: VPS CPU health
  try {
    const response = await fetch(`${LOCAL_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    health.backends.local.status = response.ok ? 'healthy' : 'unhealthy';
  } catch (error) {
    health.backends.local.status = 'unavailable';
    health.backends.local.error = error.message;
  }

  // Determine active backend (what would be used for next request)
  if (health.backends.localGpu.status === 'healthy') {
    health.activeBackend = 'localGpu';
    health.activeDescription = 'Using your GTX 1080 (free, fast)';
  } else if (health.backends.runpod.status === 'healthy') {
    health.activeBackend = 'runpod';
    health.activeDescription = 'Using RunPod RTX 4090 (paid)';
  } else if (health.backends.local.status === 'healthy') {
    health.activeBackend = 'local';
    health.activeDescription = 'Using VPS CPU (slow but reliable)';
  } else {
    health.activeBackend = 'none';
    health.activeDescription = 'No backends available!';
  }

  // Overall status
  const healthyCount = [
    health.backends.localGpu.status,
    health.backends.runpod.status,
    health.backends.local.status
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

    const result = await inference(prompt, {
      systemPrompt: system,
      maxTokens,
      temperature,
      forceBackend: backend
    });

    console.log(`[${app}] ✓ Generation complete via ${result.backend} (${result.response.length} chars)`);

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
    res.status(500).json({
      error: 'Quiz generation failed',
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
    version: '3.1.0',
    description: '3-Tier GPU Fallback System with Redis Caching',
    backends: {
      'tier1_localGpu': {
        configured: isLocalGpuConfigured(),
        model: isLocalGpuConfigured() ? LOCAL_GPU_MODEL : null,
        description: 'GTX 1080 via Ollama (free, uses your PC when online)'
      },
      'tier2_runpod': {
        configured: isRunPodConfigured(),
        model: isRunPodConfigured() ? RUNPOD_MODEL : null,
        description: 'RTX 4090 via RunPod Serverless (paid, cloud GPU)'
      },
      'tier3_local': {
        model: LOCAL_MODEL,
        description: 'Llama 3.2 3B on VPS CPU (always available fallback)'
      }
    },
    preference: BACKEND_PREFERENCE,
    endpoints: {
      'POST /api/ai/generate': 'General text generation',
      'POST /api/ai/tags': 'Generate bookmark tags (keyword or AI)',
      'POST /api/ai/explain-code': 'Explain code snippets',
      'POST /api/ai/flashcard': 'Generate flashcards',
      'POST /api/ai/quiz': 'Generate quiz questions',
      'GET /health': 'Health check with 3-tier backend status'
    },
    usage: {
      backend_param: 'Add "backend": "local-gpu|runpod|local|auto" to force a specific backend',
      auto_mode: 'Default "auto" tries backends in order: local-gpu → runpod → local'
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
║   Shared AI Gateway v3.1 - 3-Tier GPU + Redis Cache      ║
╠══════════════════════════════════════════════════════════╣
║   Port: ${PORT.toString().padEnd(48)}║
║   Mode: ${BACKEND_PREFERENCE.padEnd(49)}║
║   Cache: ${(CACHE_ENABLED ? `Enabled (TTL: ${CACHE_TTL}s)` : 'Disabled').padEnd(48)}║
║   Redis: ${REDIS_URL.substring(0, 47).padEnd(47)}║
╠══════════════════════════════════════════════════════════╣
║   Tier 1: Local GPU (GTX 1080 via Ollama)                ║
║     Configured: ${(isLocalGpuConfigured() ? 'Yes' : 'No').padEnd(40)}║
${isLocalGpuConfigured() ? `║     URL: ${LOCAL_GPU_URL.substring(0, 47).padEnd(47)}║\n║     Model: ${LOCAL_GPU_MODEL.padEnd(45)}║\n` : ''}╠══════════════════════════════════════════════════════════╣
║   Tier 2: RunPod GPU (RTX 4090 Serverless)               ║
║     Configured: ${(isRunPodConfigured() ? 'Yes' : 'No').padEnd(40)}║
${isRunPodConfigured() ? `║     Model: ${RUNPOD_MODEL.substring(0, 45).padEnd(45)}║\n` : ''}╠══════════════════════════════════════════════════════════╣
║   Tier 3: VPS CPU (Always Available)                     ║
║     URL: ${LOCAL_URL.padEnd(47)}║
║     Model: ${LOCAL_MODEL.padEnd(45)}║
╚══════════════════════════════════════════════════════════╝
  `);
});
