import { Kafka, logLevel } from 'kafkajs';

let kafka = null;
let producer = null;

const KAFKA_TOPIC = process.env.KAFKA_AI_EVENTS_TOPIC || 'ai.gateway.events';

/**
 * Initialize Kafka producer with graceful degradation.
 * Returns null if connection fails (gateway continues without events).
 */
export async function initProducer() {
  try {
    const brokers = process.env.KAFKA_BROKERS
      ? process.env.KAFKA_BROKERS.split(',')
      : ['vertex-kafka-kafka-bootstrap.microservices.svc:9092'];
    const clientId = process.env.KAFKA_CLIENT_ID || 'shared-ai-gateway';

    kafka = new Kafka({
      clientId,
      brokers,
      logLevel: logLevel.ERROR,
      retry: { initialRetryTime: 100, retries: 8 },
    });

    producer = kafka.producer({
      allowAutoTopicCreation: false,
      transactionTimeout: 30000,
    });

    await producer.connect();
    console.log('[Kafka] Producer connected', { clientId, brokers });
    return producer;
  } catch (error) {
    console.warn('[Kafka] Producer unavailable:', error.message);
    producer = null;
    return null;
  }
}

/**
 * Send AI event to Kafka. Fire-and-forget — never blocks AI requests.
 */
export function sendAIEvent(event) {
  if (!producer) return;

  producer.send({
    topic: KAFKA_TOPIC,
    messages: [{
      key: event.app || 'unknown',
      value: JSON.stringify(event),
      timestamp: Date.now().toString(),
    }],
  }).catch((err) => {
    console.warn('[Kafka] Failed to send event:', err.message);
  });
}

/**
 * Disconnect producer for graceful shutdown.
 */
export async function closeKafka() {
  if (!producer) return;
  try {
    await producer.disconnect();
    console.log('[Kafka] Producer disconnected');
  } catch (error) {
    console.error('[Kafka] Error closing producer:', error.message);
  }
}

/**
 * Check if producer is connected.
 */
export function isKafkaConnected() {
  return producer !== null;
}
