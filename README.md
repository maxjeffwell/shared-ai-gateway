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
  Tier 1: HuggingFace Inference API     Tier 1: VPS CPU Triton (bge_embeddings)
  Tier 2: VPS CPU (llama.cpp)           Tier 2: Local GPU Triton (via tunnel)
  Tier 3: RunPod GPU Serverless
  Groq:   Free tier (select apps)
  Claude: Premium (complex reasoning)
```

## LLM Fallback Tiers

| Tier | Backend | Model | When Used |
|:-----|:--------|:------|:----------|
| 1 | HuggingFace Inference | Mistral-7B-Instruct-v0.3 | Default primary |
| 2 | VPS CPU (llama.cpp) | llama-3.2-3b-instruct | Always-available fallback |
| 3 | RunPod Serverless | Llama-3.1-8B-Instruct | GPU when available |
| — | Groq | llama-3.3-70b-versatile | Auto for code-talk, educationelly, bookmarks |
| — | Anthropic Claude | claude-sonnet-4-20250514 | Explicit request or complex tasks |

In `auto` mode, the gateway health-checks each tier and falls back down the chain. Groq is used automatically for specific apps. Claude can be requested explicitly via `backend: "anthropic"`.

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

2-tier fallback: VPS CPU Triton → Local GPU Triton.

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
BACKEND_PREFERENCE=auto  # auto | local | runpod | anthropic | groq

# HuggingFace (Tier 1)
HUGGINGFACE_API_KEY=
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# VPS CPU - llama.cpp (Tier 2)
LOCAL_URL=http://llama-3b-service:8080

# RunPod GPU (Tier 3)
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
RUNPOD_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Anthropic
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Groq
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

# Embeddings
EMBEDDING_FALLBACK_URL=http://triton-embeddings:8000
EMBEDDING_PRIMARY_URL=  # optional local GPU Triton
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
