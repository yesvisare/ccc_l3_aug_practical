# Module 9.4: Advanced Reranking Strategies

## Overview

This module implements four complementary strategies for improving search result quality in production RAG systems beyond basic single cross-encoder models:

1. **Ensemble Reranking with Voting** - Multiple cross-encoder models reduce bias and improve accuracy by 8-12%
2. **Maximal Marginal Relevance (MMR)** - Balances relevance against diversity to avoid redundant results
3. **Temporal/Recency Boosting** - Applies time-aware scoring for queries requiring recent information
4. **User Preference Learning** - Personalizes results based on implicit user feedback

## Key Insight

> "A single cross-encoder makes one judgment call. It doesn't know about recency, optimizes for similarity not diversity, and has no idea what this particular user cares about."

Advanced reranking solves these limitations through ensemble confidence, diversity algorithms, temporal awareness, and personalization—but only when production metrics justify the added complexity.

## Environment Variables

The following environment variables are used by this module:

- **OPENAI_API_KEY** — Required for model-backed reranking. If unset, the API and notebook run in OFFLINE mode and skip external model calls.

See `.env.example` for the complete list of configurable parameters including model names, ensemble weights, MMR lambda, temporal decay settings, and performance budgets.

## Quickstart

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Basic Usage

```python
from src.l3_m9_advanced_reranking.l3_m9_advanced_reranking_strategies import (
    EnsembleReranker,
    MMRReranker,
    TemporalReranker,
    PersonalizationReranker,
    Document
)

# Create documents
documents = [
    Document(
        id="doc1",
        text="PyTorch 2.0 was released in 2023...",
        metadata={"timestamp": "2023-03-15T10:00:00Z"},
        score=0.8
    ),
    # ... more documents
]

# 1. Ensemble reranking
ensemble = EnsembleReranker(
    model_names=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
    aggregation="weighted"
)
result = ensemble.rerank("latest ML frameworks", documents)

# 2. MMR for diversity
mmr = MMRReranker(lambda_param=0.7)
result = mmr.rerank(documents, top_k=5)

# 3. Temporal boosting
temporal = TemporalReranker(decay_days=30, boost_factor=1.5)
result = temporal.rerank("latest frameworks", documents)

# 4. Personalization
personalization = PersonalizationReranker(min_interactions=100)
result = personalization.rerank(documents, user_profile)
```

### Run the API

**Windows (PowerShell):**
```powershell
# Start FastAPI server
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
```

**Linux/Mac:**
```bash
# Start FastAPI server
PYTHONPATH=$PWD uvicorn app:app --reload --port 8000
```

### Run Tests

**Windows (PowerShell):**
```powershell
# Run test suite
powershell -c "$env:PYTHONPATH='$PWD'; pytest -q"
```

**Linux/Mac:**
```bash
PYTHONPATH=$PWD pytest -v tests/
```

### Test the API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest machine learning frameworks",
    "documents": [
      {
        "id": "doc1",
        "text": "PyTorch 2.0 released...",
        "metadata": {"timestamp": "2023-03-15T10:00:00Z"},
        "score": 0.8
      }
    ],
    "top_k": 3,
    "strategies": ["ensemble", "mmr", "temporal"]
  }'
```

### Run Tests

```bash
pytest tests_smoke.py -v
```

## How It Works

### Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────┐
│                    Initial Documents                         │
│                     (with scores)                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              ENSEMBLE RERANKING                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Model 1  │  │ Model 2  │  │ Model 3  │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │             │             │                          │
│       └─────────────┴─────────────┘                          │
│                     │                                        │
│           Aggregate (weighted/voting/confidence)            │
│                     │                                        │
│              Latency: 200-400ms                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL BOOSTING                               │
│  • Detect temporal keywords in query                        │
│  • Apply exponential decay: boost × exp(-age/decay_days)   │
│  • Multiply relevance scores                                │
│                                                             │
│              Latency: 5ms                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              PERSONALIZATION                                 │
│  • Extract features (source, depth, length, type)           │
│  • Predict user preference score                            │
│  • Combine with base relevance                              │
│                                                             │
│              Latency: 15-30ms                               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              MMR DIVERSITY                                   │
│  • Iteratively select documents                             │
│  • score = λ×relevance - (1-λ)×max_similarity              │
│  • Balance relevance vs diversity                           │
│                                                             │
│              Latency: 10-20ms                               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Ranked Results                      │
│                     (top_k documents)                        │
└─────────────────────────────────────────────────────────────┘
```

### Ensemble Aggregation Methods

1. **Weighted Average**: Direct score combination with configurable weights
2. **Voting (Borda Count)**: Rank-based voting system
3. **Confidence Fusion**: Magnitude-based weighting by prediction confidence

### MMR Formula

```
MMR_score = λ × relevance - (1-λ) × max_similarity_to_selected

where:
- λ = 1.0: pure relevance (no diversity)
- λ = 0.0: pure diversity (may sacrifice relevance)
- λ = 0.7: recommended balance
```

### Temporal Decay Formula

```
recency_multiplier = boost_factor × exp(-decay_rate × age_days)

where:
- decay_rate = ln(2) / decay_days (half-life)
- boost_factor = maximum boost for recent documents (default: 1.5)
```

## Performance Budgets

| Strategy          | Latency Budget | Use Case                           |
|-------------------|----------------|------------------------------------|
| Ensemble          | 200-400ms      | High-stakes queries, need accuracy |
| MMR               | 10-20ms        | Diverse perspectives required      |
| Temporal          | 5ms            | Time-sensitive queries             |
| Personalization   | 15-30ms        | User-specific relevance            |
| **Combined**      | **<200ms P95** | Full pipeline with all strategies  |

## Common Failures & Fixes

### 1. Ensemble Overconfidence

**Symptom**: All models agree on incorrect ranking

**Cause**: Models trained on similar data with similar biases

**Fix**:
```python
# Use diverse model architectures
ensemble = EnsembleReranker(
    model_names=[
        "cross-encoder/ms-marco-MiniLM-L-6-v2",   # Fast, small
        "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Larger, slower
        "cross-encoder/ms-marco-electra-base"     # Different architecture
    ],
    aggregation="voting"  # Use voting instead of weighted
)
```

### 2. MMR Sacrificing Relevance

**Symptom**: Top results are diverse but not relevant

**Cause**: Lambda parameter too low (over-prioritizing diversity)

**Fix**:
```python
# Increase lambda to favor relevance
mmr = MMRReranker(lambda_param=0.8)  # Was 0.5, now 0.8
```

### 3. Recency Bias Overwhelming

**Symptom**: Recent but less relevant documents rank higher

**Cause**: Boost factor too high or decay too slow

**Fix**:
```python
# Reduce boost factor and/or increase decay rate
temporal = TemporalReranker(
    decay_days=15,      # Was 30, now faster decay
    boost_factor=1.2    # Was 1.5, now more conservative
)
```

### 4. Personalization Overfitting

**Symptom**: User gets same type of results, missing new content

**Cause**: Model memorizing individual quirks

**Fix**:
```python
# Require more interactions before personalizing
personalization = PersonalizationReranker(
    min_interactions=200  # Was 100, now requires more data
)

# In reranking, blend with base relevance
personalized_score = doc.score * (0.8 + 0.2 * pref_score)  # Less aggressive
```

### 5. Latency Bottlenecks

**Symptom**: Combined pipeline exceeds 200ms SLA

**Cause**: Sequential model execution, too many strategies

**Fix**:
```python
# Disable ensemble or use single lightweight model
reranker = AdvancedReranker(
    enable_ensemble=False,  # Skip most expensive step
    enable_mmr=True,
    enable_temporal=True,
    enable_personalization=True
)

# Or use only one fast model in ensemble
ensemble = EnsembleReranker(
    model_names=["cross-encoder/ms-marco-TinyBERT-L-6"],
    aggregation="weighted"
)
```

## Decision Card

### When to Use Advanced Reranking

✅ **USE when:**
- First-pass retrieval has ≥60% precision
- Query complexity requires multiple perspectives
- User requests diverse results
- Time-sensitive information matters
- Personalization improves user engagement
- Production traffic >1,000 queries/day
- Accuracy improvements justify latency cost

❌ **DO NOT USE when:**
- First-pass retrieval has <60% precision (fix retrieval first!)
- Simple, unambiguous queries
- Low traffic (<1,000 queries/day)
- Latency budget <100ms total
- No user interaction data for personalization
- Cost/complexity exceeds value

### Strategy Selection Guide

| Strategy          | When to Apply                                      | Skip If                                |
|-------------------|----------------------------------------------------|----------------------------------------|
| Ensemble          | High-stakes queries, need confidence               | Latency critical, simple queries       |
| MMR               | User wants diverse perspectives                    | Single answer expected                 |
| Temporal          | Query contains temporal keywords                   | Evergreen content queries              |
| Personalization   | User has ≥100 interactions                         | New user, no history                   |

## Troubleshooting

### Models Not Loading

```bash
# Check sentence-transformers installation
pip install sentence-transformers --upgrade

# Test model loading
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

### API Returns "skipped: true"

**Cause**: Models not available or missing dependencies

**Fix**: Check logs for specific error, install missing packages

### High Latency

1. Profile each strategy:
```python
result = reranker.rerank(query, documents)
print(result.debug_info["pipeline_steps"])
```

2. Disable slow strategies or reduce model count

### Personalization Not Applied

**Cause**: User has insufficient interactions

**Fix**: Check `interaction_count` in user profile meets minimum threshold

## Configuration

See `.env.example` for all configurable parameters:

- `RERANKER_MODEL_1`, `RERANKER_MODEL_2`, `RERANKER_MODEL_3`: Model identifiers
- `ENSEMBLE_WEIGHT_1`, `ENSEMBLE_WEIGHT_2`, `ENSEMBLE_WEIGHT_3`: Model weights (sum to 1.0)
- `MMR_LAMBDA`: Balance between relevance (1.0) and diversity (0.0)
- `RECENCY_DECAY_DAYS`: Half-life for exponential decay
- `RECENCY_BOOST_FACTOR`: Multiplicative boost for recent documents
- `MIN_USER_INTERACTIONS`: Minimum interactions for personalization

## Next Steps

**Previous Module**: M9.3 - HyDE (Hypothetical Document Embeddings)

**Next Module**: M10 - Production Monitoring and Observability

**Related Modules**:
- M1.4 - Basic Cross-Encoder Reranking
- M9.1 - Query Decomposition
- M9.2 - Multi-Hop Retrieval

## References

- [Cross-Encoder Models (Sentence Transformers)](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Maximal Marginal Relevance Paper](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
- [Temporal Ranking in Information Retrieval](https://en.wikipedia.org/wiki/Learning_to_rank#Temporal_factors)

## License

MIT License - See LICENSE file for details
