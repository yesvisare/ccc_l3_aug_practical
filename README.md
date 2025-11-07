# Module 9.1: Query Decomposition & Planning

> Advanced retrieval techniques for handling complex multi-part queries in RAG systems

## Overview

This module addresses a critical production limitation: **standard RAG pipelines struggle with complex multi-part queries** (15-20% of production traffic). Current simple retrieval approaches yield quality scores of 2.1/5 on complex queries versus 4.2/5 on simple ones.

**Query decomposition improves complex query accuracy from 2.1/5 to 4.0/5** by breaking queries into atomic sub-queries, building dependency graphs, executing retrievals in parallel, and synthesizing coherent answers.

### Key Capabilities

- ✅ Decompose complex queries into atomic sub-queries (95%+ accuracy)
- ✅ Build dependency graphs determining optimal retrieval order
- ✅ Implement async parallel execution reducing latency by 60% for independent queries
- ✅ Synthesize coherent answers from multiple retrieval results
- ✅ **Critically:** Recognize when NOT to use decomposition (80% of simple queries)

### Trade-offs

- ⚠️ Adds 200-500ms overhead (NOT suitable for simple queries)
- ⚠️ Higher LLM costs ($0.01-0.02 per complex query vs $0.001 for simple)
- ⚠️ Increased complexity in debugging multi-step failures
- ⚠️ 3-4× retrieval costs from multiple sub-queries

## Quickstart

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Basic Usage

```python
import asyncio
from l2_m9_query_decomposition_planning import QueryDecompositionPipeline
from config import Config

# Mock retrieval function (replace with your actual retrieval)
async def my_retrieval(query: str) -> str:
    # Your vector search / retrieval logic here
    return f"Retrieved results for: {query}"

async def main():
    pipeline = QueryDecompositionPipeline(
        api_key=Config.OPENAI_API_KEY,
        retrieval_function=my_retrieval,
        enable_fallback=True
    )

    # Process a complex query
    result = await pipeline.process_query(
        "What are the performance differences between PostgreSQL and MySQL, "
        "and which one has better JSON support?"
    )

    print(f"Answer: {result['answer']}")
    print(f"Method: {result['method']}")
    print(f"Latency: {result['latency_ms']:.0f}ms")

asyncio.run(main())
```

### 3. Run the FastAPI Server

```bash
# Start the API server
python app.py

# Test the health endpoint
curl http://localhost:8000/health

# Process a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare PostgreSQL and MySQL performance and JSON support"}'
```

### 4. Explore the Jupyter Notebook

```bash
jupyter notebook L2_M9_Query_Decomposition_Planning.ipynb
```

## How It Works

```
┌─────────────────┐
│ Complex Query   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 1. Decomposer (GPT-4)   │  Temperature=0.0, JSON output
│    - Atomic sub-queries │  Max 6 sub-queries
│    - Dependencies       │  Validation & fallback
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 2. Dependency Graph     │  NetworkX DiGraph
│    - Build DAG          │  Circular dependency check
│    - Execution levels   │  Parallel opportunities
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 3. Parallel Executor    │  Async/await patterns
│    - Semaphore (max=5)  │  Resource limiting
│    - Timeouts (30s)     │  Error isolation
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 4. Answer Synthesizer   │  LLM-based aggregation
│    - Combine results    │  Conflict resolution
│    - Context mgmt (<4K) │  Temperature=0.3
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│ Coherent Answer │
└─────────────────┘
```

## Common Failures & Fixes

### 1. Too Granular Decomposition (10+ sub-queries)

**Symptom:** `DecompositionError: Too granular decomposition: 10 sub-queries exceeds limit of 6`

**Fix:**
- Simplify the original query
- Increase `MAX_SUB_QUERIES` in config (with caution)
- Consider if the query is actually multiple separate questions

### 2. Circular Dependencies

**Symptom:** `DependencyError: Circular dependencies detected - cannot execute`

**Fix:**
- LLM decomposition created invalid dependency graph
- Retry decomposition (temperature=0.0 should be deterministic but may vary)
- Manually validate the decomposition logic

### 3. Parallel Execution Timeouts

**Symptom:** `TimeoutError` during retrieval execution

**Fix:**
- Reduce `MAX_CONCURRENT_RETRIEVALS` (default: 5)
- Increase `RETRIEVAL_TIMEOUT_SEC` (default: 30)
- Optimize your retrieval function
- Check vector database rate limits

### 4. Context Overflow

**Symptom:** `SynthesisError: Context overflow: N retrievals exceeded 4000 token limit`

**Fix:**
- Reduce document chunk size in retrieval
- Increase `MAX_CONTEXT_TOKENS` (default: 4000)
- Implement result truncation before synthesis
- Use summarization on sub-results

### 5. Answer Synthesis Conflicts

**Symptom:** Contradictory information in sub-answers

**Fix:**
- Review sub-query decomposition quality
- Implement custom conflict resolution logic
- Add source tracking to identify conflicts
- Manual intervention for edge cases

## Decision Card: When to Use Query Decomposition

### ✅ Use When:

- Query has **2-4 distinct semantic parts**
- Sub-queries are **largely independent** or have clear dependencies
- **Latency budget ≥700ms** (accounts for overhead)
- **Accuracy improvement worth cost increase**
- Handling **15-20% complex queries** in production traffic

### ❌ Don't Use When:

- **Simple direct questions** (80%+ of traffic) - Adds unnecessary latency
- **Real-time/latency-sensitive apps** (<500ms requirement)
- **Very high query volume** (>100K queries/day) on limited budget
- **Highly domain-specific queries** (medical, legal) - May need fine-tuning

### 📊 Performance Comparison

| Metric | Simple Query | Complex w/o Decomp | Complex w/ Decomp |
|--------|--------------|-------------------|-------------------|
| **Latency** | 200ms | 250ms | **800ms** |
| **Quality** | 4.2/5 | **2.1/5** | **4.0/5** |
| **Cost** | $0.001 | $0.001 | **$0.020** |
| **Use Case** | 80% traffic | ❌ Poor quality | ✅ Complex only |

### 🎯 Key Decision Rule

Deploy when **15-20% complex query volume justifies the cost and latency trade-off**. For 80% simple queries, standard retrieval remains optimal.

## Alternative Solutions

Before implementing query decomposition, consider:

1. **Single-Shot Retrieval with Better Prompting**
   - Simplest approach, no code changes
   - Better system prompts and context
   - Use when: Query complexity is manageable with better prompting

2. **Query Expansion (Not Decomposition)**
   - Middle-ground approach
   - Generate semantic variations without full decomposition
   - Use when: Need better coverage without multiple retrievals

3. **Managed Query Understanding Service**
   - Zero implementation effort
   - Vendor-dependent (Google Vertex AI, Azure)
   - Use when: Want turnkey solution, can accept vendor lock-in

4. **Fine-Tuned Decomposition Model**
   - Advanced option for specialized domains
   - Lower inference cost after training
   - Use when: High volume (>100K/day) justifies training investment

## Troubleshooting

### No API Key

```
⚠️ Skipping API calls (no OPENAI_API_KEY)
```

**Fix:** Set `OPENAI_API_KEY` in your `.env` file or environment.

### Import Errors

```
ModuleNotFoundError: No module named 'networkx'
```

**Fix:** `pip install -r requirements.txt`

### Async Event Loop Issues

```
RuntimeError: This event loop is already running
```

**Fix:** Use `await` in async context, or `asyncio.run()` at top level.

### Rate Limiting

```
openai.RateLimitError: Rate limit exceeded
```

**Fix:**
- Reduce `MAX_CONCURRENT_RETRIEVALS`
- Add retry logic with exponential backoff
- Upgrade OpenAI API tier

## Production Deployment Checklist

- [ ] **Fallback enabled** - Set `ENABLE_FALLBACK=true` for graceful degradation
- [ ] **Rate limiting** - Implement rate limiting on decomposition calls
- [ ] **Circuit breakers** - Add timeouts for stuck async operations
- [ ] **Logging** - Log all failed sub-queries with full context
- [ ] **Monitoring** - Track:
  - Decomposition success rate
  - Sub-query retrieval quality per dependency level
  - Synthesis conflict frequency
  - End-to-end latency (p50, p95, p99)
  - Cost per query
- [ ] **Cost controls** - Set budget alerts for OpenAI API usage
- [ ] **A/B testing** - Compare decomposition vs. simple retrieval on sample traffic

## Running Tests

```bash
# Run smoke tests
python tests_smoke.py

# Expected output:
# ✓ Config loads successfully
# ✓ Core decomposition returns plausible shape
# ✓ Network paths gracefully skip without keys
```

## Project Structure

```
.
├── l2_m9_query_decomposition_planning.py  # Core implementation
├── config.py                               # Configuration management
├── app.py                                  # FastAPI entrypoint
├── requirements.txt                        # Dependencies
├── .env.example                            # Environment template
├── example_data.json                       # Sample queries & test data
├── L2_M9_Query_Decomposition_Planning.ipynb  # Interactive tutorial
├── tests_smoke.py                          # Basic tests
└── README.md                               # This file
```

## Cost Breakdown

Based on GPT-4 Turbo pricing (as of 2024):

- **Decomposition**: $0.01-0.015 per complex query
- **Multiple Retrievals**: 3-4× your base retrieval cost
- **Answer Synthesis**: $0.005 per query
- **Total**: ~$0.02 per complex query vs $0.001 for simple

For 10,000 complex queries/day: **~$200/day or $6,000/month** (decomposition only, excluding retrieval infrastructure).

## Scaling Considerations

- **Concurrent retrieval limits** - Most vector DBs limit concurrent connections (e.g., Pinecone: 100/sec)
- **LLM API rate limiting** - OpenAI Tier 1: 500 RPM, Tier 2: 5000 RPM
- **Memory overhead** - Each async task consumes ~10-50KB memory
- **Token context limits** - GPT-4 Turbo: 128K input, but synthesis quality degrades beyond 4K

## Next Modules

- **Module 9.2**: Query Rewriting & Expansion
- **Module 9.3**: Hybrid Search Techniques
- **Module 10**: Multi-Modal RAG

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review example_data.json for common scenarios
- Open an issue with full error logs and config

---

**Remember:** Query decomposition is a powerful tool for complex queries, but it's not a silver bullet. Use it judiciously when the trade-offs align with your requirements.
