# Module 9: Hypothetical Document Embeddings (HyDE)

## Overview

HyDE addresses vocabulary mismatch in retrieval systems by generating hypothetical document-style answers before embedding, rather than directly embedding user queries. This improves precision by 15-40% for conceptual queries from non-expert users.

**Key Insight:** User queries and document answers live in different semantic spaces. HyDE bridges this gap by transforming questions into document-style answers first.

**Example:**
- **Query:** "What are the tax implications of stock options?"
- **Hypothesis:** "Stock option taxation follows IRS code section 422 for ISOs and 83 for NSOs. Upon exercise, income recognition depends on holding period..."
- **Result:** Better matches with formal compliance documents

## Quickstart

### 1. Installation

```bash
# Clone and navigate to workspace
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Minimum required: OPENAI_API_KEY
# Optional: PINECONE_API_KEY, PINECONE_INDEX_NAME
```

### 3. Run Examples

```bash
# CLI examples (requires OPENAI_API_KEY)
python l2_m9_hypothetical_document_embeddings.py

# Run smoke tests
python tests_smoke.py

# Start FastAPI server
python app.py
# Then visit http://localhost:8000/docs
```

### 4. Jupyter Notebook

```bash
jupyter notebook L2_M9_Hypothetical_Document_Embeddings_\(HyDE\).ipynb
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    HyDE Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User Query                                               │
│     "What are tax implications of stock options?"            │
│                        │                                     │
│                        ▼                                     │
│  2. Generate Hypothesis (LLM)                               │
│     "Stock option taxation follows IRC 422..."               │
│     [500-800ms, $0.001-0.005]                               │
│                        │                                     │
│                        ▼                                     │
│  3. Embed Hypothesis                                         │
│     [1536-dimensional vector]                                │
│     [100-150ms, $0.0001]                                    │
│                        │                                     │
│                        ▼                                     │
│  4. Vector Search (Pinecone)                                │
│     [Retrieve top-K similar documents]                       │
│     [50-100ms]                                              │
│                        │                                     │
│                        ▼                                     │
│  5. Ranked Results                                           │
│     [Documents matching hypothesis]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Adaptive Routing:**
- Conceptual queries → HyDE (15-40% precision gain)
- Factoid queries → Traditional (no latency overhead)
- Hybrid → Both methods, merged results

## Common Failures & Fixes

### 1. Generic Hypotheses
**Symptom:** Vague hypotheses like "It works by following a process..."

**Root Cause:** Query too broad or LLM lacks domain context

**Fix:**
```python
# Provide domain context and example documents
generator.generate_hypothesis(
    query=query,
    domain_context="Financial documents use IRC sections and tax terminology",
    example_documents=[doc1, doc2]
)
```

### 2. Precision Drop on Factoid Queries
**Symptom:** "When was X?" queries return wrong results

**Root Cause:** HyDE generates hallucinated or generic answers for factoids

**Fix:** Use adaptive routing (automatic in AdaptiveHyDERetriever)
```python
retriever = AdaptiveHyDERetriever(...)
result = retriever.retrieve(query)  # Auto-routes based on query type
```

### 3. Timeout Errors
**Symptom:** 504 Gateway Timeout, p95 latency >2s

**Root Cause:** OpenAI API latency spikes under load

**Fix:**
- Implement hypothesis caching (Redis)
- Use async generation with proper timeouts
- Set application timeout to 3-5s (not 1.5s)

### 4. Poor Domain Quality
**Symptom:** Wrong hypotheses for specialized domains

**Root Cause:** GPT-4 lacks your domain expertise

**Fix:** Use RAG-augmented hypothesis generation
```python
# First retrieve context, then generate hypothesis with that context
context = traditional_retrieval(query)
hypothesis = generate_with_context(query, context)
```

### 5. Wrong Routing Decisions
**Symptom:** Classification accuracy <80%

**Root Cause:** Regex patterns too brittle

**Fix:** Upgrade to LLM-based classification (50-100ms overhead, 90% accuracy)

## Decision Card

### ✅ BENEFIT
Bridges vocabulary mismatch; 15-40% precision gain for conceptual queries; works without retraining embeddings; effective for compliance/legal/technical domains where formal and informal language differ.

### ❌ LIMITATION
Adds 500-1000ms latency (cannot be eliminated); costs $0.001-0.005 per query (10-50x traditional); reduces precision on factoid queries; only benefits 20-30% of queries; fails on highly specialized niche domains.

### 💰 COST
- **Implementation:** 6-8 hours development + 4-6 hours monitoring
- **Operational:** $100-2000/month OpenAI (1K-100K queries/day); $70-500/month Pinecone; $20-50/month Redis (optional caching)
- **Complexity:** 300+ lines of code, 4 new components

### 🤔 USE WHEN
- Building knowledge base for non-expert users
- Queries are conceptual ("What/How/Why")
- Severe vocabulary mismatch (informal ↔ formal)
- Latency budget >700ms p95
- Budget allows $0.001-0.005 per query
- Domain general enough for GPT-4

### 🚫 AVOID WHEN
- Latency requirement <500ms → use fine-tuned embeddings
- Queries primarily factoid → traditional retrieval sufficient
- Budget <$0.001 per query → use query expansion or hybrid BM25
- Highly specialized domain → fine-tuned embeddings better
- Can afford fine-tuning ($500 one-time vs $1500/month ongoing)

## Troubleshooting

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY='sk-your-key-here'
# Or add to .env file
```

### "Pinecone connection failed"
This is expected if you don't have Pinecone configured. The module will:
- Still generate hypotheses
- Skip vector search (return empty results)
- Return 200 OK with `"skipped": true`

### "Hypothesis generation too slow"
Check p95 latency:
- Expected: 400-800ms
- If >1500ms: Implement caching or switch to gpt-3.5-turbo (10x cheaper)

### "Precision worse with HyDE"
Check query types:
- Run classification on sample queries
- If >50% are factoid, disable HyDE for those
- Use `AdaptiveHyDERetriever` for automatic routing

### Cost too high
Optimizations:
1. **Caching:** Save 40-50% (easy win)
2. **Adaptive routing:** Only use HyDE for 20-30% of queries → 70% cost reduction
3. **Cheaper model:** Use gpt-3.5-turbo ($0.0001 vs $0.001)
4. **At scale:** Consider fine-tuned embeddings (Alternative 2)

## Module Structure

```
ccc_l3_aug_practical/
├── l2_m9_hypothetical_document_embeddings.py  # Main module
├── app.py                                      # FastAPI wrapper
├── config.py                                   # Configuration
├── requirements.txt                            # Dependencies
├── .env.example                                # Config template
├── example_data.json                           # Sample queries
├── tests_smoke.py                              # Basic tests
├── README.md                                   # This file
└── L2_M9_Hypothetical_Document_Embeddings_(HyDE).ipynb  # Tutorial
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Query (Adaptive Routing)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the tax implications?", "top_k": 5}'
```

### Generate Hypothesis
```bash
curl -X POST http://localhost:8000/hypothesis \
  -H "Content-Type: application/json" \
  -d '{"query": "What are stock options?"}'
```

### Classify Query
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "When was the deadline?"}'
```

## Performance Benchmarks

| Metric | Traditional | HyDE | Hybrid |
|--------|-------------|------|--------|
| Latency (p50) | 120ms | 650ms | 700ms |
| Latency (p95) | 200ms | 1200ms | 1500ms |
| Cost per query | $0.0001 | $0.001-0.005 | $0.0015-0.006 |
| Precision (conceptual) | 0.62 | 0.78 | 0.81 |
| Precision (factoid) | 0.85 | 0.65 | 0.83 |

## Alternative Approaches

1. **Query Expansion** (Free, fast, simple synonyms)
2. **Fine-Tuned Embeddings** ($500 one-time, no latency overhead)
3. **Hybrid BM25+Dense** (Proven, faster, cheaper)

See notebook for detailed comparison.

## Next Module

**M9.4: Advanced Reranking Strategies**
- Ensemble cross-encoders
- MMR diversity
- Recency boosting
- User preference learning

## Support

- Issues: Report at project GitHub
- Documentation: See Jupyter notebook for detailed walkthrough
- Practathon: Complete challenges in notebook Section 11

---

**Author:** Module 9 Implementation Team
**Version:** 1.0.0
**Last Updated:** 2025-01-07
