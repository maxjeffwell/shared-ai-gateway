# Shared AI Gateway

Node.js/Express API gateway (v3.5) providing LLM inference and embeddings to all portfolio applications. Features a multi-tier fallback system, Redis caching, Prometheus metrics, and LLM observability via Langfuse.

**Port:** 8002 | **Image:** `maxjeffwell/shared-ai-gateway`

## Architecture

```
                         ┌──────────────────────────────────┐
                         │      Shared AI Gateway           │
                         │      (Express, port 8002)        │
                         └──────┬───────────────┬───────────┘
                                │               │
                    ┌───────────▼──┐     ┌──────▼──────────┐
                    │  LLM Backends│     │  Embedding      │
                    │  (fallback)  │     │  Backends       │
                    └──────────────┘     └─────────────────┘
                    │                    │
  Tier 0: OVMS LLM (Intel iGPU)         Tier 1: OVMS e5-large (Intel iGPU)
  Tier 3: HuggingFace Inference API     Tier 2: OVMS (fallback URL)
  Tier 3b: Groq (free tier)
  Tier 4: RunPod GPU Serverless
  Claude: Premium (complex reasoning)
```

## LLM Fallback Tiers

| Tier | Backend | Model | When Used |
|:-----|:--------|:------|:----------|
| 0 | OVMS LLM (Intel iGPU, in-cluster) | qwen2.5-1.5b-instruct | Default primary (free) |
| 3 | HuggingFace Inference | Llama-3.1-8B-Instruct | If configured |
| 3b | Groq | openai/gpt-oss-120b | Free cloud fallback; auto for code-talk, educationelly, bookmarks |
| 4 | RunPod Serverless | Llama-3.1-8B-Instruct | Paid last resort |
| — | Anthropic Claude | claude-sonnet-4-20250514 | Explicit request or complex tasks |

In `auto` mode, the gateway health-checks each tier and falls back down the chain. Groq is used automatically for specific apps. Claude can be requested explicitly via `backend: "anthropic"`. The Ollama tiers (local GPU tunnel, VPS CPU llama.cpp) were removed 2026-10-02.

`GET /health` reports `ok` while OVMS LLM is healthy, `degraded` when traffic is falling back to remote/paid tiers, and `critical` when no backend is up.

## API Endpoints

### Text Generation

| Endpoint | Purpose | Key Params |
|:---------|:--------|:-----------|
| `POST /api/ai/generate` | General text generation | `prompt`, `app`, `maxTokens`, `temperature`, `backend` |
| `POST /api/ai/chat` | Multi-turn conversation | `messages`, `maxTokens`, `temperature`, `backend` |
| `POST /api/ai/tags` | Bookmark tag generation | `title`, `url`, `description` |
| `POST /api/ai/describe` | Bookmark descriptions | `url`, `title` |
| `POST /api/ai/explain-code` | Code explanation | `code`, `language` |
| `POST /api/ai/flashcard` | Flashcard generation | `topic`, `content` |
| `POST /api/ai/quiz` | Quiz generation | `topic`, `difficulty`, `count` |

### Embeddings

| Endpoint | Purpose | Key Params |
|:---------|:--------|:-----------|
| `POST /api/ai/embed` | Generate text embeddings | `texts` (array) |

2-tier fallback: `EMBEDDING_PRIMARY_URL` → `EMBEDDING_FALLBACK_URL` (both OVMS e5-large today).

### System

| Endpoint | Purpose |
|:---------|:--------|
| `GET /health` | Backend status, cache status, observability config |
| `GET /metrics` | Prometheus metrics |

## Caching

Redis caching with SHA256 keys from prompt + options:

- **TTL:** 1 hour (configurable)
- **Skips cache for:** temperature > 0.5, empty/short responses (< 5 chars)
- **Graceful degradation:** gateway continues if Redis is unavailable

## Prometheus Metrics

| Metric | Type | Labels |
|:-------|:-----|:-------|
| `gateway_requests_total` | Counter | backend, endpoint, status |
| `gateway_request_duration_seconds` | Histogram | backend, endpoint |
| `gateway_fallback_total` | Counter | from, to |
| `gateway_cache_total` | Counter | hit/miss |
| `gateway_backend_healthy` | Gauge | backend |

## Observability

LLM requests are traced through **LiteLLM** → **Langfuse** for full request/response logging, token usage, latency, and cost tracking.

## Environment Variables

```bash
# Server
PORT=8002
BACKEND_PREFERENCE=auto  # auto | huggingface | runpod | anthropic

# OVMS LLM (Tier 0)
OVMS_LLM_URL=http://ovms-llm.ovms:8000/v3
OVMS_LLM_MODEL=qwen2.5-1.5b-instruct

# HuggingFace (Tier 3)
HUGGINGFACE_API_KEY=
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

# RunPod GPU (Tier 4)
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
RUNPOD_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Anthropic
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Groq
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b

# Embeddings
EMBEDDING_PRIMARY_URL=  # local GPU Triton (Tier 1, e.g. https://embeddings.el-jefe.me)
EMBEDDING_FALLBACK_URL=http://triton-embeddings:8000  # VPS CPU Triton (Tier 2)
EMBEDDING_MODEL=bge_embeddings

# Redis
REDIS_URL=redis://redis:6379
CACHE_ENABLED=true
CACHE_TTL=3600

# Observability
LITELLM_URL=https://litellm.el-jefe.me
```

## Development

```bash
npm install
npm run dev    # nodemon
npm start      # production
```

## Kubernetes Deployment

Deployed to the K3s cluster via ArgoCD with:

- **Gateway:** 1 replica, 100m–500m CPU, 256–512Mi memory
- **LiteLLM Proxy:** Sidecar deployment on port 4000 (OpenAI-compatible interface for Claude/Groq)
- **Network Policy:** Only pods with `portfolio: "true"` label can reach the gateway
- **Health Probes:** Liveness (30s interval) and readiness (10s interval) on `/health`

## CI/CD

GitHub Actions builds on push to `main`, pushes to Docker Hub with `latest` and SHA tags, using Doppler for secrets management.

## Client Applications

| App | Features Used |
|:----|:-------------|
| **Bookmarked** | Tag generation, descriptions, embeddings (pgvector search) |
| **educationELLy** | Flashcards, quizzes, educational chat |
| **Code Talk** | Code explanation, general chat |
| **IntervalAI** | Spaced repetition content generation |
