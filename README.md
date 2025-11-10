# L3 Module 10.4: Conversational RAG with Memory

Multi-turn dialogue system with dual-level memory (verbatim + summarized), spaCy-based reference resolution, and Redis session persistence.

**Level 3 | TVH Baseline | Windows-First**

## Overview

This module implements a production-ready conversational RAG system that:

- **Dual-level memory**: Short-term (last 5 turns verbatim) + long-term (LLM-summarized)
- **Reference resolution**: Resolves pronouns ("it", "that", etc.) with 80-90% accuracy
- **Session persistence**: Redis-backed with 7-day TTL for 10K+ concurrent conversations
- **Token management**: Automatic summarization when approaching context limits
- **Production-ready**: FastAPI wrapper with health checks and metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversational RAG                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌─────────────────────┐         │
│  │ User Query       │─────▶│ Reference Resolver  │         │
│  │ "How tall is it?"│      │ (spaCy NLP)         │         │
│  └──────────────────┘      └──────────┬──────────┘         │
│                                       │                      │
│                            Resolve "it" → "Eiffel Tower"    │
│                                       │                      │
│                                       ▼                      │
│  ┌─────────────────────────────────────────────────┐        │
│  │       Conversation Memory Manager                │        │
│  ├─────────────────────────────────────────────────┤        │
│  │  Short-term Buffer (Last 5 turns, verbatim)     │        │
│  │  ┌──────────────────────────────────────┐       │        │
│  │  │ Turn 1: User: "Tell me about..."     │       │        │
│  │  │ Turn 2: Asst: "The Eiffel Tower..." │       │        │
│  │  │ Turn 3: User: "How tall is it?"     │       │        │
│  │  └──────────────────────────────────────┘       │        │
│  │                                                  │        │
│  │  Long-term Summary (LLM-compressed)            │        │
│  │  ┌──────────────────────────────────────┐       │        │
│  │  │ "Earlier discussed French landmarks, │       │        │
│  │  │  construction history..."            │       │        │
│  │  └──────────────────────────────────────┘       │        │
│  └─────────────────────────────────────────────────┘        │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────┐        │
│  │ LLM (GPT-4o-mini) with full context             │        │
│  │ → Generate contextual response                   │        │
│  └─────────────────────────────────────────────────┘        │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────┐        │
│  │ Session Manager (Redis)                          │        │
│  │ → Save conversation state with 7-day TTL        │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Install spaCy model for reference resolution
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Optional: Start Redis

```bash
# Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or use existing Redis instance
# Update REDIS_HOST and REDIS_PORT in .env
```

### 4. Run the API Server

**Windows (PowerShell - Recommended):**
```powershell
powershell -File scripts\run_api.ps1
```

**Linux/Mac:**
```bash
PYTHONPATH=$PWD python app.py
```

Visit http://localhost:8000/docs for interactive API documentation.

### 5. Or Use the Python Module

```python
from src.l3_m10_conversational_rag_memory import ConversationalRAG
from config import get_clients, Config

# Initialize
clients = get_clients()
rag = ConversationalRAG(
    llm_client=clients["openai"],
    redis_client=clients["redis"],
    short_term_size=Config.SHORT_TERM_BUFFER_SIZE
)

# Query with session
session_id = "user-123"
response = rag.query("Tell me about the Eiffel Tower", session_id=session_id)
print(response)

# Follow-up with reference
response = rag.query("How tall is it?", session_id=session_id)
# "it" is automatically resolved to "Eiffel Tower"
print(response)

# Check memory stats
stats = rag.get_memory_stats()
print(stats)
```

### 6. Run Tests

**Windows (PowerShell - Recommended):**
```powershell
powershell -File scripts\run_tests.ps1
```

**Linux/Mac:**
```bash
PYTHONPATH=$PWD pytest -v tests/
```

## API Endpoints

### POST /query

Process conversational query with memory.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "session_id": "demo-session",
    "create_session": true
  }'
```

**Response:**
```json
{
  "response": "Python is a high-level programming language...",
  "session_id": "demo-session",
  "memory_stats": {
    "short_term_turns": 2,
    "has_long_term_summary": false,
    "estimated_tokens": 150
  },
  "skipped": false
}
```

### GET /session/{session_id}

Get session statistics.

```bash
curl "http://localhost:8000/session/demo-session"
```

### POST /session/reset

Reset memory for a session.

```bash
curl -X POST "http://localhost:8000/session/reset" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-session"}'
```

### GET /health

Health check with configuration validation.

## Environment Variables

All configuration is managed through `.env` file. See `.env.example` for a complete template.

### Required
- `OPENAI_API_KEY` - Your OpenAI API key for LLM calls

### Optional
- `ANTHROPIC_API_KEY` - Alternative LLM provider
- `REDIS_HOST` - Redis server host (default: `localhost`)
- `REDIS_PORT` - Redis server port (default: `6379`)
- `REDIS_DB` - Redis database number (default: `0`)
- `REDIS_PASSWORD` - Redis password (if required)
- `REDIS_SESSION_TTL` - Session TTL in seconds (default: `604800` = 7 days)

### Memory Configuration
- `SHORT_TERM_BUFFER_SIZE` - Number of recent turns to keep verbatim (default: `5`)
- `MAX_CONTEXT_TOKENS` - Token limit before summarization (default: `8000`)
- `SUMMARY_MODEL` - Model for summarization (default: `gpt-4o-mini`)

### Other
- `DEFAULT_MODEL` - Default LLM model (default: `gpt-4o-mini`)
- `SPACY_MODEL` - spaCy NLP model (default: `en_core_web_sm`)

## Offline Mode

For development and testing without API keys or Redis:

**Windows:**
```powershell
$env:OFFLINE="true"
python app.py
```

**Linux/Mac:**
```bash
OFFLINE=true python app.py
```

In offline mode:
- API endpoints return mock responses
- No external API calls are made
- Session persistence is disabled
- All operations are logged but skipped

This is useful for:
- Local development without API costs
- CI/CD testing pipelines
- Demonstrations without live services

## How It Works

### 1. Dual-Level Memory

**Short-term buffer**: Last 5 turns stored verbatim for exact recall
- Fast retrieval
- Preserves conversational context
- Used for reference resolution

**Long-term memory**: Older turns compressed via LLM summarization
- Prevents context overflow
- Maintains key facts and entities
- Triggers automatically when buffer exceeds threshold

**Migration flow**:
```
Turns 1-10 added → Turn 11 triggers migration
→ Turns 1-6 summarized → Moved to long-term
→ Turns 7-11 remain in short-term buffer
```

### 2. Reference Resolution

Uses spaCy NLP to detect and resolve:
- **Pronouns**: it, that, this, these, those, them, they
- **Named entities**: Barack Obama, Google, Python
- **Noun chunks**: "the latest smartphone", "renewable energy"

**Example**:
```
Context: "The Eiffel Tower is in Paris."
Query: "How tall is it?"
Resolved: "How tall is the Eiffel Tower?"
```

**Accuracy**: 80-90% on simple cases, 60-70% on ambiguous references

### 3. Session Management

Redis persistence with:
- JSON serialization of conversation state
- 7-day default TTL (configurable)
- Automatic expiry for inactive sessions
- Load/save operations for fault tolerance

## Common Failures & Fixes

### 1. Memory Overflow (>20 turns)

**Symptom**: Quality degradation, token limit errors

**Fix**:
- Automatic summarization triggers at `MAX_CONTEXT_TOKENS`
- Adjust `SHORT_TERM_BUFFER_SIZE` to keep fewer verbatim turns
- Monitor `memory_stats.estimated_tokens`

### 2. Wrong Antecedent Resolution

**Symptom**: "it" resolves to incorrect entity

**Example**:
```
Context: "Tesla and Ford make EVs. Ford has long history."
Query: "Tell me about its founder"
Wrong: its → Ford (should be context-dependent)
```

**Fix**:
- Provide more specific queries
- Current system uses simple heuristic (most recent entity)
- Production would need neural coreference resolution

### 3. Session Expiry Mid-Conversation

**Symptom**: Redis session expires during active use

**Fix**:
- Increase `REDIS_SESSION_TTL` (default: 7 days)
- Implement session refresh on each query
- Monitor session creation/expiry rates

### 4. Token Limit Exceeded

**Symptom**: Summary generation itself consumes too many tokens

**Fix**:
- Reduce `MAX_CONTEXT_TOKENS` threshold
- Use more aggressive summarization (shorter summaries)
- Implement sliding window instead of summarization

### 5. Cross-Contamination

**Symptom**: User sessions share memory (isolation failure)

**Fix**:
- Ensure unique `session_id` per user
- Test with `tests_smoke.py::test_cross_contamination`
- Verify Redis key isolation (`session:{id}`)

## Decision Card

### ✅ Choose Conversational Memory When:

- Users ask follow-up questions (60-70% of production queries)
- Conversation spans 3+ turns
- Reference resolution improves answer quality
- Session persistence needed for fault tolerance

### ❌ Avoid When:

- Pure lookup/search queries dominate (no context needed)
- Highly regulated data requiring zero storage
- Budget constraints prohibit per-query LLM costs
- <3 turn conversations (stateless RAG sufficient)

### 🔄 Alternative Solutions:

1. **Stateless RAG**: No memory, sufficient for isolated queries
2. **Client-side memory**: Browser storage, reduces server load but loses persistence
3. **Managed platforms**: ChatGPT Assistants API, outsource complexity
4. **PostgreSQL-backed**: High-scale option with better querying

## Production Considerations

### Scaling

- **Latency**: +50-100ms per query for reference resolution
- **Cost**: $0.03 per 1K tokens (conversation length matters)
- **Throughput**: Redis supports 10K+ concurrent sessions with proper tuning

### Cost Breakdown (5,000 conversations/day)

- API calls (GPT-4o-mini summaries): ~$150/month
- Redis storage: ~$20/month
- Infrastructure: ~$50-100/month

**Total**: ~$220-270/month

### Monitoring (Non-Negotiable)

Track these metrics:
- Reference resolution accuracy
- Session creation/expiry rates
- Token consumption per conversation
- Redis memory utilization
- Query latency (p50, p95, p99)

Enable Prometheus metrics at `/metrics` endpoint.

## Troubleshooting

### spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

### Redis connection failed

```bash
# Check Redis is running
docker ps | grep redis

# Test connection
redis-cli ping
# Should return: PONG
```

### OpenAI API errors

- Verify `OPENAI_API_KEY` in `.env`
- Check API quota/billing: https://platform.openai.com/usage
- Monitor rate limits (10K RPM for tier 1)

### Memory not persisting

- Verify Redis is configured in `.env`
- Check `session_id` is provided in queries
- Confirm Redis TTL hasn't expired (default: 7 days)

## Project Structure

```
.
├── app.py                      # FastAPI wrapper (thin, /health + router)
├── config.py                   # Configuration management
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment template
├── .gitignore                  # Python/Jupyter ignores
├── pyproject.toml              # Project metadata + pytest config
├── LICENSE                     # License file
├── README.md                   # This file
├── src/
│   └── l3_m10_conversational_rag_memory/
│       └── __init__.py         # Core implementation (all logic)
├── notebooks/
│   └── L3_M10_Conversational_RAG_with_Memory.ipynb  # Tutorial
├── tests/
│   └── test_m10_conversational_rag_memory.py  # Smoke tests
├── configs/
│   ├── example.json            # Config schema documentation
│   └── example_data.json       # Sample conversation data
└── scripts/
    ├── run_api.ps1             # Windows PowerShell - run API
    ├── run_tests.ps1           # Windows PowerShell - run tests
    └── demo_run.py             # Interactive CLI demo
```

## Next Steps

- **Module 10.5**: Agentic workflows with tool orchestration
- **Module 11**: Production monitoring and observability
- **Module 12**: Advanced coreference resolution with neural models

## References

- [OpenAI Chat Completions](https://platform.openai.com/docs/guides/chat)
- [spaCy NLP](https://spacy.io/)
- [Redis Python Client](https://redis-py.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Production Reminder**: "Production systems require 3-4x infrastructure over development" - monitor religiously to catch reference resolution failures at scale.
