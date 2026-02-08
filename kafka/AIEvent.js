import { randomUUID } from 'crypto';

/**
 * Build an AI gateway event from an endpoint result.
 * Metadata only — no prompt/response text for privacy.
 */
export function createAIEvent({ endpoint, app, result, latencyMs }) {
  return {
    eventId: randomUUID(),
    timestamp: Date.now(),
    endpoint,
    app: app || 'unknown',
    backend: result?.backend || 'unknown',
    model: result?.model || null,
    status: 'success',
    latencyMs: Math.round(latencyMs),
    usage: {
      promptTokens: result?.usage?.prompt_tokens || 0,
      completionTokens: result?.usage?.completion_tokens || 0,
    },
    fromCache: result?.fromCache || false,
  };
}

/**
 * Build an error event.
 */
export function createAIErrorEvent({ endpoint, app, error, latencyMs }) {
  return {
    eventId: randomUUID(),
    timestamp: Date.now(),
    endpoint,
    app: app || 'unknown',
    backend: 'unknown',
    model: null,
    status: 'error',
    latencyMs: Math.round(latencyMs),
    usage: { promptTokens: 0, completionTokens: 0 },
    fromCache: false,
  };
}
