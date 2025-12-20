import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fetch from 'node-fetch';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8002;

// OpenVINO inference backend URL
const INFERENCE_URL = process.env.INFERENCE_URL || 'http://portfolio-ai-engine:8001';

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
 * Health check endpoint
 */
app.get('/health', async (req, res) => {
  try {
    const response = await fetch(`${INFERENCE_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error(`Backend unhealthy: ${response.status}`);
    }

    const backendHealth = await response.json();

    res.json({
      status: 'ok',
      gateway: 'healthy',
      backend: backendHealth,
      inference_url: INFERENCE_URL
    });
  } catch (error) {
    console.error('Health check failed:', error.message);
    res.status(503).json({
      status: 'error',
      gateway: 'healthy',
      backend: 'unhealthy',
      error: error.message
    });
  }
});

/**
 * POST /api/ai/generate
 * General-purpose text generation
 *
 * Body:
 * {
 *   "prompt": "Your prompt here",
 *   "app": "bookmarks|education|code|flashcard|quiz|general",
 *   "maxTokens": 200,
 *   "temperature": 0.7
 * }
 */
app.post('/api/ai/generate', async (req, res) => {
  try {
    const {
      prompt,
      app = 'general',
      maxTokens = 200,
      temperature = 0.7,
      systemPrompt
    } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    // Build full prompt with system context
    const system = systemPrompt || SYSTEM_PROMPTS[app] || SYSTEM_PROMPTS.general;
    const fullPrompt = `${system}\n\n${prompt}`;

    console.log(`[${app}] Generating: ${prompt.substring(0, 60)}...`);

    const response = await fetch(`${INFERENCE_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: fullPrompt,
        max_new_tokens: maxTokens
      }),
      signal: AbortSignal.timeout(90000)
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.status}`);
    }

    const data = await response.json();

    // Clean up response
    let text = data.response || '';

    // Remove the original prompt if it's echoed back
    if (text.startsWith(fullPrompt)) {
      text = text.substring(fullPrompt.length).trim();
    } else if (text.startsWith(prompt)) {
      text = text.substring(prompt.length).trim();
    }

    console.log(`[${app}] ✓ Generation complete (${text.length} chars)`);

    res.json({
      success: true,
      app,
      response: text,
      model: 'TinyLlama-1.1B-INT8'
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

    const prompt = `Title: JavaScript Array Methods\nTags: javascript, arrays, programming, tutorial, web\n\nTitle: ${title}\nTags:`;

    const response = await fetch(`${INFERENCE_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 40
      }),
      signal: AbortSignal.timeout(90000)
    });

    if (!response.ok) {
      // Fallback to keyword extraction on AI failure
      const tags = extractKeywordTags(title, url, description);
      console.log(`[bookmarks] AI failed, using keyword fallback:`, tags);

      return res.json({
        success: true,
        tags,
        title,
        url,
        method: 'keyword-extraction-fallback',
        warning: 'AI generation failed, used keyword extraction'
      });
    }

    const data = await response.json();
    let text = data.response || '';

    // Parse tags from AI response
    const tags = parseTags(text, title);

    console.log(`[bookmarks] ✓ AI generated ${tags.length} tags:`, tags);

    res.json({
      success: true,
      tags,
      title,
      url,
      method: 'ai-generation'
    });

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
 *   "language": "javascript"
 * }
 */
app.post('/api/ai/explain-code', async (req, res) => {
  try {
    const { code, language = 'unknown' } = req.body;

    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }

    const prompt = `${SYSTEM_PROMPTS.code}

Explain this ${language} code concisely:

\`\`\`${language}
${code}
\`\`\`

Explanation:`;

    console.log(`[code] Explaining ${language} code (${code.length} chars)`);

    const response = await fetch(`${INFERENCE_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 300
      }),
      signal: AbortSignal.timeout(90000)
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.status}`);
    }

    const data = await response.json();
    let explanation = data.response || '';

    // Clean up
    if (explanation.startsWith(prompt)) {
      explanation = explanation.substring(prompt.length).trim();
    }

    console.log(`[code] ✓ Explanation generated`);

    res.json({
      success: true,
      explanation,
      language,
      code: code.substring(0, 100) + (code.length > 100 ? '...' : '')
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
 *   "content": "A closure is a function that has access to variables in its outer scope"
 * }
 */
app.post('/api/ai/flashcard', async (req, res) => {
  try {
    const { topic, content } = req.body;

    if (!topic || !content) {
      return res.status(400).json({ error: 'Topic and content are required' });
    }

    const prompt = `${SYSTEM_PROMPTS.flashcard}

Create a flashcard for learning.

Topic: ${topic}
Content: ${content}

Generate a clear question and answer for a flashcard.

Question:`;

    console.log(`[flashcard] Generating for topic: ${topic}`);

    const response = await fetch(`${INFERENCE_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 150
      }),
      signal: AbortSignal.timeout(90000)
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.status}`);
    }

    const data = await response.json();
    let text = data.response || '';

    // Try to parse question and answer
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

    let question = lines[0] || `What is ${topic}?`;
    let answer = lines.find(l => l.toLowerCase().startsWith('answer:')) ||
                 lines[1] ||
                 content.substring(0, 100);

    // Clean up
    question = question.replace(/^Question:\s*/i, '').replace(/^\d+\.\s*/, '');
    answer = answer.replace(/^Answer:\s*/i, '');

    console.log(`[flashcard] ✓ Flashcard generated`);

    res.json({
      success: true,
      topic,
      question,
      answer
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
 *   "count": 3
 * }
 */
app.post('/api/ai/quiz', async (req, res) => {
  try {
    const { topic, difficulty = 'medium', count = 3 } = req.body;

    if (!topic) {
      return res.status(400).json({ error: 'Topic is required' });
    }

    const prompt = `${SYSTEM_PROMPTS.quiz}

Generate ${count} ${difficulty} difficulty quiz questions about: ${topic}

Format each question as:
Q: [question]
A: [correct answer]

Questions:`;

    console.log(`[quiz] Generating ${count} ${difficulty} questions for: ${topic}`);

    const response = await fetch(`${INFERENCE_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 400
      }),
      signal: AbortSignal.timeout(90000)
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.status}`);
    }

    const data = await response.json();
    let text = data.response || '';

    // Parse questions
    const questions = parseQuizQuestions(text);

    console.log(`[quiz] ✓ Generated ${questions.length} questions`);

    res.json({
      success: true,
      topic,
      difficulty,
      count: questions.length,
      questions
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
    version: '1.0.0',
    model: 'TinyLlama-1.1B-INT8 (OpenVINO)',
    endpoints: {
      'POST /api/ai/generate': 'General text generation',
      'POST /api/ai/tags': 'Generate bookmark tags',
      'POST /api/ai/explain-code': 'Explain code snippets',
      'POST /api/ai/flashcard': 'Generate flashcards',
      'POST /api/ai/quiz': 'Generate quiz questions',
      'GET /health': 'Health check'
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
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════╗
║   Shared AI Gateway                    ║
║   Port: ${PORT.toString().padEnd(32)}║
║   Backend: ${INFERENCE_URL.padEnd(26)}║
╚════════════════════════════════════════╝
  `);
});
