# Shared AI Gateway

Node.js/Express API gateway that provides AI features to all portfolio applications.

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Your Apps      │      │  Shared AI       │      │  portfolio-ai-engine│
│  (Node.js)      │─────▶│  Gateway         │─────▶│  (Python/OpenVINO)  │
│                 │ HTTP │  (Express)       │ HTTP │  TinyLlama-1.1B-INT8│
│ - Bookmarked    │      │  Port 8002       │      │  Port 8001          │
│ - educationELLy │      └──────────────────┘      └─────────────────────┘
│ - code-talk     │
└─────────────────┘
```

## Features

- **Tag Generation**: Generate relevant tags for bookmarks
- **Code Explanation**: Explain code snippets in natural language
- **Flashcard Generation**: Create educational flashcards
- **Quiz Generation**: Generate quiz questions
- **General Text Generation**: Flexible AI text generation

## API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "gateway": "healthy",
  "backend": {
    "status": "ok",
    "model_loaded": true
  }
}
```

### `POST /api/ai/generate`
General-purpose text generation

**Request:**
```json
{
  "prompt": "Explain React hooks",
  "app": "education",
  "maxTokens": 200,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "app": "education",
  "response": "React hooks are functions that let you...",
  "model": "TinyLlama-1.1B-INT8"
}
```

### `POST /api/ai/tags`
Generate tags for bookmarks

**Request:**
```json
{
  "title": "JavaScript Array Methods Guide",
  "url": "https://example.com/js-arrays",
  "description": "Comprehensive guide to map, filter, reduce"
}
```

**Response:**
```json
{
  "success": true,
  "tags": ["javascript", "arrays", "tutorial", "programming", "web-development"]
}
```

### `POST /api/ai/explain-code`
Explain code snippets

**Request:**
```json
{
  "code": "const items = arr.map(x => x * 2);",
  "language": "javascript"
}
```

**Response:**
```json
{
  "success": true,
  "explanation": "This code creates a new array by doubling each element...",
  "language": "javascript"
}
```

### `POST /api/ai/flashcard`
Generate flashcards

**Request:**
```json
{
  "topic": "JavaScript Closures",
  "content": "A closure is a function that has access to variables in its outer scope"
}
```

**Response:**
```json
{
  "success": true,
  "topic": "JavaScript Closures",
  "question": "What is a closure in JavaScript?",
  "answer": "A function that has access to variables in its outer scope"
}
```

### `POST /api/ai/quiz`
Generate quiz questions

**Request:**
```json
{
  "topic": "React Hooks",
  "difficulty": "medium",
  "count": 3
}
```

**Response:**
```json
{
  "success": true,
  "topic": "React Hooks",
  "questions": [
    {
      "question": "What hook is used for managing state?",
      "answer": "useState"
    }
  ]
}
```

## Environment Variables

```bash
PORT=8002
INFERENCE_URL=http://portfolio-ai-engine:8001
```

## Development

```bash
npm install
npm run dev
```

## Docker

```bash
docker build -t shared-ai-gateway:latest .
docker run -p 8002:8002 shared-ai-gateway:latest
```

## Integration Example

```javascript
// In your Node.js app
const response = await fetch('http://shared-ai-gateway:8002/api/ai/tags', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: 'My Bookmark',
    url: 'https://example.com',
    description: 'A great resource'
  })
});

const { tags } = await response.json();
console.log(tags); // ['resource', 'example', 'bookmark']
```

## Supported Applications

- `general`: General-purpose assistance
- `bookmarks`: Bookmark tagging
- `education`: Educational content
- `code`: Code analysis
- `flashcard`: Flashcard generation
- `quiz`: Quiz generation

## Model

- **Model**: TinyLlama-1.1B-INT8
- **Runtime**: OpenVINO (CPU optimized)
- **Quantization**: INT8 for fast CPU inference
- **Backend**: Python FastAPI service
