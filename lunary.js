import { randomUUID } from 'crypto';
import fetch from 'node-fetch';

// Lunary direct integration for traces and threads
// LiteLLM handles LLM-level runs via callbacks; this module handles
// parent agent traces and chat thread events that LiteLLM doesn't create.

const LUNARY_API_URL = process.env.LUNARY_API_URL;
const LUNARY_PUBLIC_KEY = process.env.LUNARY_PUBLIC_KEY;

const isConfigured = () => !!LUNARY_API_URL && !!LUNARY_PUBLIC_KEY;

export function generateRunId() {
  return randomUUID();
}

/**
 * Send events to Lunary's ingest API (fire-and-forget)
 * Errors are logged but never thrown to avoid breaking the main request flow
 */
async function sendEvents(events) {
  if (!isConfigured()) return;

  try {
    const payload = Array.isArray(events) ? events : [events];
    const response = await fetch(`${LUNARY_API_URL}/v1/runs/ingest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${LUNARY_PUBLIC_KEY}`,
      },
      body: JSON.stringify({ events: payload }),
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      console.warn(`[Lunary] Ingest failed (${response.status}): ${body.substring(0, 200)}`);
    }
  } catch (error) {
    console.warn(`[Lunary] Ingest error: ${error.message}`);
  }
}

/**
 * Start a parent agent trace for a gateway request.
 * LiteLLM's LLM run becomes a child of this trace via parent_run_id.
 */
export async function startTrace({ traceId, app, userId, input }) {
  return sendEvents({
    type: 'agent',
    event: 'start',
    runId: traceId,
    name: `${app || 'gateway'}-chat`,
    timestamp: new Date().toISOString(),
    tags: [app, 'gateway'].filter(Boolean),
    userId,
    input,
    metadata: { app, source: 'shared-ai-gateway' },
  });
}

/**
 * End a parent agent trace with the response
 */
export async function endTrace({ traceId, output, tokensUsage }) {
  return sendEvents({
    type: 'agent',
    event: 'end',
    runId: traceId,
    timestamp: new Date().toISOString(),
    output,
    tokensUsage,
  });
}

/**
 * Mark a trace as errored
 */
export async function errorTrace({ traceId, error }) {
  return sendEvents({
    type: 'agent',
    event: 'error',
    runId: traceId,
    timestamp: new Date().toISOString(),
    error: { message: error.message, stack: error.stack },
  });
}

/**
 * Track a chat message in a Lunary thread.
 * Lunary auto-creates the thread container on first message.
 *
 * Apps must pass a stable threadId to group messages into conversations.
 * Without threadId, thread tracking is skipped.
 */
export async function trackChatMessage({ threadId, role, content, app, userId }) {
  if (!threadId) return;

  return sendEvents({
    type: 'chat',
    event: 'chat',
    runId: generateRunId(),
    parentRunId: threadId,
    timestamp: new Date().toISOString(),
    message: { role, content },
    userId,
    tags: [app, 'chat'].filter(Boolean),
    threadTags: [app, 'chat'].filter(Boolean),
  });
}

/**
 * Build metadata object for LiteLLM proxy requests.
 * LiteLLM extracts this and forwards to Lunary callback,
 * linking the LLM run as a child of our parent agent trace.
 */
export function buildLiteLLMMetadata({ traceId, app, userId, threadId }) {
  if (!traceId) return undefined;

  const metadata = {
    parent_run_id: traceId,
    tags: [app, 'gateway'].filter(Boolean),
  };

  if (userId) metadata.trace_user_id = userId;
  if (threadId) metadata.thread_id = threadId;
  if (app) metadata.app = app;

  return metadata;
}

export function isLunaryConfigured() {
  return isConfigured();
}
