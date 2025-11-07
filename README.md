# Module 11.3: Resource Management & Throttling

## Overview

This module implements **per-tenant resource quotas and fair scheduling** to prevent the "noisy neighbor" problem in multi-tenant SaaS systems. One tenant's excessive usage shouldn't degrade service for others.

### What You'll Build

- ✅ Per-tenant quota tracking (queries, tokens, storage)
- ✅ Fair queue with round-robin scheduling
- ✅ Throttling middleware for FastAPI
- ✅ Background queue workers
- ✅ Atomic quota checking (race-condition free)
- ✅ Resource-weighted quotas

### Real-World Impact

- **Prevents noisy neighbors:** Saves 20-40% infrastructure costs
- **Predictable performance:** Maintains 2-3s p95 latency under load
- **Cost control:** Prevents $10K+ surprise bills

## Quickstart

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your Redis configuration
```

### 2. Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or use managed Redis (Railway, Render, etc.)
```

### 3. Run the Module

```bash
# Test the module directly
python l2_m11_resource_management_throttling.py

# Or run the FastAPI app
python app.py

# Or explore the Jupyter notebook
jupyter notebook L2_M11_Resource_Management_Throttling.ipynb
```

### 4. Basic Usage

```python
import redis
from l2_m11_resource_management_throttling import QuotaManager, QuotaType

# Initialize
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
qm = QuotaManager(r)

# Set tenant tier
qm.set_tenant_tier("tenant_123", "pro")

# Check quota
under_quota, current, limit = qm.check_quota(
    "tenant_123",
    QuotaType.QUERIES_HOURLY
)

# Record usage
qm.record_query("tenant_123", tokens_used=1000)

# Get status
status = qm.get_quota_status("tenant_123")
```

## How It Works

### Architecture Diagram

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Tenant ID      │  Extract from header/token
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Quota Check    │  Redis: atomic check-and-increment
└──────┬──────────┘
       │
       ├─ Under quota ──────────────┐
       │                            ▼
       │                    ┌───────────────┐
       │                    │   Process     │
       │                    │   Request     │
       │                    └───────────────┘
       │
       └─ Over quota ───────────────┐
                                    ▼
                            ┌───────────────┐
                            │   Fair Queue  │ Round-robin per tenant
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │ Queue Worker  │ Process when capacity available
                            └───────────────┘
```

### Components

#### 1. QuotaManager
Tracks usage across multiple dimensions:
- Queries per hour/day/month
- Token consumption (OpenAI)
- Storage usage

Uses Redis for atomic counting with automatic TTL expiration.

#### 2. FairTenantQueue
Implements fair FIFO scheduling with round-robin tenant selection to prevent starvation.

#### 3. ThrottlingMiddleware
Checks quotas before processing requests. Under-quota requests proceed immediately; over-quota requests enter queue or receive 429 responses.

#### 4. QueueWorker
Background process that drains queues when capacity becomes available, maintaining fair service.

## Common Failures & Fixes

### Failure 1: Noisy Neighbor Despite Quotas

**Problem:** Tenant stays under query quota but uses expensive operations (GPT-4 with 8K context).

**Fix:** Use resource-weighted quotas
```python
from l2_m11_resource_management_throttling import ResourceWeightedQuota

weighted = ResourceWeightedQuota(r)
weight = weighted.calculate_query_weight(request_data)
# GPT-4 with large context = 40-80x weight vs GPT-3.5
```

### Failure 2: Race Condition Bypass

**Problem:** Concurrent requests bypass quota checks (check-then-act race).

**Fix:** Use atomic operations
```python
from l2_m11_resource_management_throttling import AtomicQuotaManager

atomic_qm = AtomicQuotaManager(r)
success, current, limit = atomic_qm.atomic_check_and_increment(
    tenant_id, QuotaType.QUERIES_HOURLY, increment=1
)
```

### Failure 3: Queue Starvation

**Problem:** Enterprise tenant with 100 queued requests delays free tenant's single request.

**Fix:** Queue-depth-aware scheduling (already implemented in FairTenantQueue).

### Failure 4: Unbounded Queue Growth

**Problem:** Traffic spike fills queue, consuming memory.

**Fix:** Set global queue limits
```python
queue = FairTenantQueue(r, max_queue_size=100)
# Also set MAX_TOTAL_QUEUE_SIZE=5000 in .env
```

### Failure 5: Emergency Quota Increase Requires Deploy

**Problem:** Sales closes deal, customer needs quota increase NOW.

**Fix:** Database-backed quota configuration (see `config.py` for admin API integration).

## Decision Card

### ✅ USE WHEN

- 50-500 tenants on shared infrastructure
- Experiencing noisy neighbor complaints
- Need predictable cost control
- Can accept 10-20ms latency overhead

### 🚫 AVOID WHEN

- **<50 tenants** → Use trust-based approach (no quotas)
- **Need <50ms latency SLA** → Use reserved capacity
- **Highly spiky traffic (10x+)** → Use hard limits + auto-scaling
- **Team <5 people** → Too complex to manage

### 💰 COST

- **Initial:** 12-16 hours implementation
- **Ongoing:** $300-600/month infrastructure + 8-12 hours/month management
- **Complexity:** 3 new failure modes to monitor

## Alternative Solutions

### 1. No Quotas (Trust-Based)
**Best for:** <50 tenants, all paying customers
**Pros:** Zero complexity
**Cons:** One tenant can impact all

### 2. Hard Limits (No Queuing)
**Best for:** 50-200 tenants with clear tiers
**Pros:** Simple (100 lines vs 600)
**Cons:** Poor UX (hard rejections)

### 3. Dynamic Throttling
**Best for:** 100+ tenants with variable usage
**Pros:** Better resource utilization
**Cons:** Complex to tune

### 4. Reserved Capacity
**Best for:** Enterprise customers ($10K+/month)
**Pros:** Guaranteed performance
**Cons:** High cost ($500-2000/tenant/month)

## Troubleshooting

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping

# Should return: PONG

# If not, start Redis
docker run -d -p 6379:6379 redis:7-alpine
```

### Quota Not Enforcing

```python
# Check quota configuration
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
tier = r.get("tenant:YOUR_TENANT_ID:tier")
print(f"Tenant tier: {tier}")

# Verify quota key exists
key = "quota:YOUR_TENANT_ID:queries_hourly:2025-01-15-14"
value = r.get(key)
print(f"Current usage: {value}")
```

### Queue Not Processing

```bash
# Check queue depth
redis-cli LLEN queue:tenant:YOUR_TENANT_ID

# Check active tenants
redis-cli SMEMBERS queue:active_tenants

# Check queue worker is running
ps aux | grep queue_worker
```

## API Endpoints (app.py)

### GET /health
Health check endpoint
```bash
curl http://localhost:8000/health
```

### POST /query
Submit a query (respects quotas)
```bash
curl -X POST http://localhost:8000/query \
  -H "X-Tenant-ID: tenant_123" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query"}'
```

### GET /quota/status
Get quota status for tenant
```bash
curl http://localhost:8000/quota/status \
  -H "X-Tenant-ID: tenant_123"
```

### GET /queue/stats
Get queue statistics
```bash
curl http://localhost:8000/queue/stats
```

### POST /admin/quota/update
Update tenant quota (requires admin key)
```bash
curl -X POST http://localhost:8000/admin/quota/update \
  -H "X-Admin-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_123",
    "tier": "enterprise"
  }'
```

## Testing

### Run Smoke Tests

```bash
pytest tests_smoke.py -v
```

### Load Testing

```python
# Test concurrent requests
import concurrent.futures
import requests

def make_request():
    return requests.post(
        "http://localhost:8000/query",
        headers={"X-Tenant-ID": "tenant_test"},
        json={"query": "Test"}
    )

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(make_request) for _ in range(200)]
    results = [f.result() for f in futures]

# First 100 should succeed, rest queued/rejected
successes = sum(1 for r in results if r.status_code == 200)
print(f"Successes: {successes}/200")
```

## Production Considerations

### Monitoring Metrics

Track these in Prometheus/Datadog:

```python
# Quota utilization per tenant
quota_utilization = current_usage / quota_limit  # Alert if >90%

# Queue depth
queue_depth_per_tenant  # Alert if >50
global_queue_depth  # Alert if >1000

# Queue wait time
average_queue_wait_seconds  # Alert if >60s

# Quota rejection rate
quota_rejection_rate = rejected / total  # Alert if >5%

# Redis memory
redis_memory_bytes  # Alert if >80% capacity
```

### Scaling

**At 500+ tenants:**
- Redis memory: 125MB → 500MB with overhead
- Consider Redis clustering
- Use Redis pipelining for batch operations

**At 1000+ req/sec:**
- Quota checks become bottleneck
- Cache quota configs (refresh every 60s)
- Use probabilistic counters for high-scale tracking

### Multi-Region

- Redis must be shared across regions OR
- Route all quota checks to single region (adds 50-100ms latency)
- Use Redis cluster with cross-region replication

## Next Module

**M11.4: Vector Index Sharding** - When your Pinecone namespace strategy breaks down and you need to shard across multiple indexes.

## License

Educational purposes - CCC Level 3

## Support

- Check the Jupyter notebook for detailed examples
- Review the source code in `l2_m11_resource_management_throttling.py`
- Test with `example_data.json` scenarios
