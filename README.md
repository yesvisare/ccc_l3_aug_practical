# Module 11.4: Vector Index Sharding

Production-grade vector database sharding for multi-tenant SaaS applications. Distributes tenants across multiple Pinecone indexes using consistent hashing to overcome single-index limitations.

## Overview

This module implements sharding for vector databases when namespace-based isolation from earlier modules (M11.1-M11.3) reaches its limits. Use sharding **only when** metrics justify the operational complexity.

**When to Shard:**
- \>80 tenants
- \>1M total vectors
- Approaching 100 namespace limit per index
- Single index P95 latency >500ms

**When NOT to Shard:**
- <80 tenants (namespace isolation sufficient)
- Highly uneven tenant sizes (creates hot shards)
- Frequent cross-tenant queries (sharding penalizes these)
- Team lacks distributed systems experience

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Services

```bash
cp .env.example .env
# Edit .env with your API keys:
# - PINECONE_API_KEY
# - OPENAI_API_KEY
# - REDIS_HOST (optional, falls back to in-memory)
```

### 3. Run Tests

```bash
pytest tests_smoke.py -v
```

### 4. Start API

```bash
python app.py
# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Explore Notebook

```bash
jupyter notebook L2_M11_Vector_Index_Sharding.ipynb
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Request                           │
│                 (tenant-001, query_text)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │     Shard Manager          │
         │  (Consistent Hashing)      │
         │  hash = mmh3(tenant_id)    │
         │  shard = hash % num_shards │
         └────────────┬───────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌────────┐   ┌────────┐   ┌────────┐
   │Shard 0 │   │Shard 1 │   │Shard 2 │
   │────────│   │────────│   │────────│
   │tenant-0│   │tenant-1│   │tenant-2│
   │tenant-3│   │tenant-4│   │tenant-5│
   └────────┘   └────────┘   └────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
              ┌──────────────┐
              │    Redis     │
              │ (Assignment  │
              │   Tracking)  │
              └──────────────┘
```

### Key Components

1. **ShardManager**: Routes tenants deterministically using MurmurHash3
   - Caches assignments in Redis
   - Supports explicit assignment for rebalancing
   - Maintains routing consistency across restarts

2. **ShardedRAG**: Coordinates queries across shards
   - Single-tenant: Hits one shard (~350ms P95)
   - Cross-shard: Aggregates all shards (slower, admin only)

3. **Monitoring**: Tracks shard health against thresholds
   - Vector count: Rebalance at 300K/shard
   - Namespace count: Alert at 18/20 capacity
   - P95 latency: Warn at 500ms

## Common Failures & Fixes

### 1. Hot Shard Problem

**Symptom:** One shard has significantly more vectors/namespaces than others, causing degraded performance.

**Cause:** Large tenants concentrated on single shard due to hash distribution or growth over time.

**Fix:**
```python
# Explicitly reassign heavy tenant to underutilized shard
shard_manager.assign_tenant_to_shard("large-tenant-id", target_shard_id=2)
```

**Prevention:**
- Monitor shard distribution regularly (`GET /metrics`)
- Set alerts at 18/20 namespace threshold
- Plan rebalancing during low-traffic windows

### 2. Routing Inconsistency

**Symptom:** Same tenant routes to different shards across requests.

**Cause:** Redis connection lost, cache eviction, or inconsistent `num_shards` config.

**Fix:**
```python
# Verify Redis connectivity
redis_client.ping()

# Check consistent num_shards across all services
assert Config.NUM_SHARDS == 4  # Must match across all instances
```

**Prevention:**
- Use Redis persistence (RDB/AOF)
- Validate config on startup
- Log shard assignments for audit trail

### 3. Cross-Shard Query Timeout

**Symptom:** Admin queries across all shards timeout or return partial results.

**Cause:** High latency aggregating results from multiple indexes.

**Fix:**
```python
# Reduce top_k per shard or add timeout handling
result = rag.query_cross_shard(query_text, top_k=3)  # Lower from default 5

# Or implement async with timeout
async with timeout(5.0):
    result = await async_cross_shard_query(...)
```

**Prevention:**
- Limit cross-shard queries to admin/analytics only
- Cache frequent cross-shard results
- Consider separate analytics index

### 4. Rebalancing Disruption

**Symptom:** Tenant queries fail during rebalancing operations.

**Cause:** Moving tenant data between shards without blue-green strategy.

**Fix:**
```python
# Blue-green rebalancing:
# 1. Write to both old and new shard
# 2. Backfill data to new shard
# 3. Switch reads to new shard
# 4. Clean up old shard

# Update assignment only after backfill completes
shard_manager.assign_tenant_to_shard(tenant_id, new_shard_id)
```

**Prevention:**
- Plan rebalancing as zero-downtime operation
- Test with staging data first
- Monitor error rates during migration

## Decision Card

Use this decision framework before implementing sharding:

| Metric | Namespace Isolation | Sharded Architecture |
|--------|-------------------|---------------------|
| **Max Tenants** | ~100 | 1000s |
| **Query Latency (P95)** | 200-400ms | Single: 350ms, Cross: 800ms+ |
| **Operational Complexity** | Low | High |
| **Cross-Tenant Queries** | Fast | Slow (avoid) |
| **Rebalancing** | Not needed | Required skill |
| **Team Skills Required** | Basic Python, Pinecone | Distributed systems, async patterns |
| **Cost** | 1 index fee | N × index fees |
| **When to Use** | <80 tenants, <1M vectors | >80 tenants, hitting limits |

**Recommendation:** Start with namespace isolation (M11.1-M11.3). Migrate to sharding only when metrics prove necessity. Most SaaS applications never need sharding.

## Troubleshooting

### Redis Connection Fails

**Error:** `redis.exceptions.ConnectionError: Error connecting to Redis`

**Solution:**
```bash
# Start local Redis
docker run -d -p 6379:6379 redis:latest

# Or update .env for remote Redis
REDIS_HOST=your-redis-host.com
REDIS_PORT=6379
REDIS_PASSWORD=your-password
```

### Pinecone Index Not Found

**Error:** `pinecone.core.client.exceptions.NotFoundException: Index 'tenant-shard-0' not found`

**Solution:**
```python
# Create shard indexes before use
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
for i in range(4):
    pc.create_index(
        name=f"tenant-shard-{i}",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
```

### Uneven Shard Distribution

**Issue:** All tenants hashing to same shard.

**Diagnosis:**
```bash
curl http://localhost:8000/metrics
# Check distribution.shard_X.tenant_count
```

**Solution:**
- Verify `num_shards` > 1
- Check tenant_id format (ensure variation)
- Manually rebalance if needed

### High Latency on Single-Tenant Queries

**Issue:** P95 latency >500ms for single-tenant queries.

**Diagnosis:**
```python
# Check shard vector count
health = monitor_shard_health(shard_manager, pinecone_client)
for shard in health["shards"]:
    print(f"{shard['index_name']}: {shard['total_vectors']} vectors")
```

**Solution:**
- If shard >300K vectors: Trigger rebalancing
- If specific tenant large: Move to dedicated shard
- Consider pod-based index for large tenants

## API Reference

### Endpoints

**GET /health**
```json
{
  "status": "ok",
  "services": {"pinecone": true, "redis": true, "openai": true},
  "config": {"num_shards": 4}
}
```

**POST /ingest**
```json
{
  "tenant_id": "tenant-001",
  "documents": [
    {"id": "doc1", "text": "...", "metadata": {}}
  ]
}
```

**POST /query**
```json
{
  "tenant_id": "tenant-001",
  "query_text": "search query",
  "top_k": 5
}
```

**GET /metrics**
```json
{
  "distribution": {"0": {"tenant_count": 25, "tenants": [...]}},
  "health": {"needs_rebalancing": false, "alerts": []}
}
```

## Next Steps

- **Module 11.5:** Advanced rebalancing strategies
- **Module 12:** Multi-region deployment
- **Module 13:** Cost optimization and capacity planning

## References

- [Pinecone Sharding Best Practices](https://docs.pinecone.io/)
- [Consistent Hashing Explained](https://en.wikipedia.org/wiki/Consistent_hashing)
- Redis persistence: [RDB vs AOF](https://redis.io/docs/manual/persistence/)
