# Module 11.1: Multi-Tenant SaaS Architecture - Tenant Isolation Strategies

Production-grade tenant isolation for RAG systems supporting 100-500 customers with namespace-based and index-based isolation strategies.

## Overview

This module implements **defense-in-depth** tenant isolation using:

- **PostgreSQL Row-Level Security (RLS)**: Database-level enforcement preventing cross-tenant leakage even with application bugs
- **Namespace isolation**: Cost-efficient shared index for Free/Pro tiers (10-500 tenants)
- **Index isolation**: Dedicated Pinecone indexes for Enterprise tier (strongest guarantees)
- **Cost tracking**: Variable and fixed cost allocation per tenant
- **Performance isolation**: Resource quotas and rate limiting

### Key Architecture Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Layer Defense                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Application Code (TenantDataManager)                   │
│           - Mandatory tenant_id parameter on all queries         │
│           - Wrapper prevents queries without tenant context      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Vector Database Isolation                              │
│           - Namespace scoping (Free/Pro)                         │
│           - Dedicated indexes (Enterprise)                       │
│           - Metadata filtering (belt-and-suspenders)             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Database RLS (PostgreSQL)                              │
│           - Row-level policies enforce tenant_id filtering       │
│           - Prevents leakage even with SQL injection             │
└─────────────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run Smoke Tests

```bash
# Verify installation
python tests_smoke.py

# Or use pytest
pytest tests_smoke.py -v
```

### 3. CLI Demo

```bash
# Run standalone demo
python l2_m11_tenant_isolation_strategies.py
```

Expected output:
```
=== Module 11.1: Tenant Isolation Strategies Demo ===

1. Registering tenants...
   - Acme Corp: free tier, namespace=tenant_tenant-001
   - Beta LLC: pro tier, namespace=tenant_tenant-002
   - Enterprise Inc: enterprise tier, index=tenant-tenant-003
...
```

### 4. Start API Server

```bash
# Start FastAPI server
python app.py

# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 5. Explore Jupyter Notebook

```bash
jupyter notebook L2_M11_Tenant_Isolation_Strategies.ipynb
```

## How It Works

### Tenant Registration & Tier Assignment

```python
from l2_m11_tenant_isolation_strategies import TenantRegistry, TenantTier

registry = TenantRegistry()

# Free tier: shared index with namespace isolation
free_tenant = registry.register_tenant("acme-001", "Acme Corp", TenantTier.FREE)
# → namespace: "tenant_acme-001", max 1K docs, 100 queries/day

# Enterprise tier: dedicated index for strongest isolation
ent_tenant = registry.register_tenant("big-corp", "BigCorp Inc", TenantTier.ENTERPRISE)
# → dedicated_index: "tenant-big-corp", max 100K docs, 10K queries/day
```

### Data Operations with Mandatory Scoping

```python
from l2_m11_tenant_isolation_strategies import TenantDataManager

data_manager = TenantDataManager(registry)

# Upsert documents - tenant_id is MANDATORY
documents = [
    {"id": "doc1", "values": [0.1] * 384, "metadata": {"title": "Document 1"}}
]
result = data_manager.upsert_documents(
    tenant_id="acme-001",  # Cannot omit this!
    documents=documents
)

# Query with automatic namespace routing
results = data_manager.query_documents(
    tenant_id="acme-001",  # Enforced by method signature
    query_vector=[0.15] * 384,
    top_k=5
)
```

**Critical**: The `TenantDataManager` makes it **impossible** to query without a tenant_id. This prevents the #1 cause of cross-tenant leakage.

### Cost Tracking & Allocation

```python
from l2_m11_tenant_isolation_strategies import CostAllocationEngine

cost_engine = CostAllocationEngine()

# Track variable costs per query
cost = cost_engine.track_query_cost(
    tenant_id="acme-001",
    embed_tokens=500,   # ~$0.00005
    llm_tokens=1000     # ~$0.002
)
# Total: ~$0.00206

# Allocate fixed infrastructure costs monthly
allocated = cost_engine.allocate_fixed_costs(
    monthly_fixed_cost=115.0,  # $50 Pinecone + $30 DB + $35 monitoring
    allocation_basis={
        "acme-001": 20.0,   # 20% of total usage
        "beta-002": 50.0,   # 50% of total usage
        "big-corp": 30.0    # 30% of total usage
    }
)
# acme-001: $23.00, beta-002: $57.50, big-corp: $34.50
```

## Common Failures & Fixes

### 1. Cross-Tenant Data Leakage

**Problem**: Forgot namespace parameter in query code

```python
# WRONG - queries ALL tenants' data
results = pinecone_client.query(vector=[0.1] * 384, top_k=5)
```

**Solution**: Use `TenantDataManager` wrapper that enforces tenant scoping

```python
# CORRECT - impossible to query without tenant_id
results = data_manager.query_documents(
    tenant_id="acme-001",  # Required parameter
    query_vector=[0.1] * 384,
    top_k=5
)
```

### 2. Performance Degradation (40ms+ overhead)

**Problem**: Isolation checks add latency to every query

**Solutions**:
- ✅ Cache tenant configs (5-minute TTL) - eliminates DB lookup
- ✅ Batch cost tracking writes - don't record per-query in DB
- ✅ Remove redundant metadata filters for dedicated indexes

**Code fix**:
```python
# Before: 40ms overhead
config = db.query("SELECT * FROM tenants WHERE id = ?", tenant_id)  # Every query!

# After: 5ms overhead
config = cache.get(f"tenant:{tenant_id}", ttl=300)  # Cache hit
```

### 3. Cost Allocation Inaccuracy

**Problem**: Shared infrastructure costs unallocated → negative margins

**Verification**:
```python
allocated_total = sum(cost_engine.allocate_fixed_costs(...).values())
actual_bill = 115.0  # From cloud provider

# MUST be within ±5%
assert abs(allocated_total - actual_bill) / actual_bill < 0.05
```

**Solution**: Monthly reconciliation + proportional allocation by usage %

### 4. Namespace Exhaustion

**Problem**: Hit Pinecone limit of 100 namespaces at 101st customer

**Prevention**:
```python
# Monitor capacity continuously
current = registry.namespace_usage["shared-index-1"]
max_safe = 90  # Conservative limit

if current >= int(max_safe * 0.8):  # 72 namespaces
    logger.error("ALERT: 80% namespace capacity - provision new index!")

if current >= max_safe:
    raise RuntimeError("Namespace exhaustion - auto-provision new index")
```

### 5. Network Isolation Misconfiguration

**Problem**: VPC security group allows all traffic instead of whitelisted IPs

**Test**:
```bash
# From unauthorized IP - should FAIL
curl -X POST https://api.example.com/query \
  -H "Authorization: Bearer token" \
  -d '{"tenant_id": "acme-001", "query": "test"}'
# Expected: 403 Forbidden

# From authorized IP - should SUCCEED
curl -X POST https://api.example.com/query \
  -H "Authorization: Bearer token" \
  -d '{"tenant_id": "acme-001", "query": "test"}'
# Expected: 200 OK
```

## Decision Card

### When to Use Namespace Isolation

✅ **Choose namespace isolation when:**
- Running **10-500 customers** at $20-200/month each
- Data isolation required but not life-critical
- Cost efficiency matters (shared infrastructure)
- Can accept **15-25ms isolation overhead** per query

**Cost**: $3-7 per tenant/month at scale

### When to Use Alternatives

#### Alternative #1: Single-Tenant Deployments
- **Best for**: <20 enterprise customers
- **Required for**: HIPAA/FedRAMP compliance
- **Cost**: $500-2000/month per customer
- **Tradeoff**: True physical isolation but high operational overhead

#### Alternative #2: Tenant-per-Database
- **Best for**: 50-200 tenant sweet spot
- **Benefit**: Stronger isolation than namespaces
- **Challenge**: Connection pooling complexity
- **Cost**: $5-15 per tenant/month

#### Alternative #3: Hybrid Tiering
- **Best for**: Scaling from many small to few large customers
- Free/Pro on shared infrastructure
- Enterprise on dedicated instances
- **Tradeoff**: Managing two operational models

#### Alternative #4: Geo-Distributed Multi-Tenant
- **Best for**: GDPR data residency requirements
- **Cost**: 3-5x infrastructure costs
- Each region runs independent deployment
- Route tenants to home region automatically

## Production Checklist

**Before deployment:**

- [ ] Row-Level Security policies tested (cannot access cross-tenant data)
- [ ] Network isolation configured for Enterprise tenants
- [ ] Cost allocation verified (±5% of actual cloud spend)
- [ ] Namespace capacity monitoring with 80% alerts (72/90)
- [ ] Load tested at **2x expected traffic**
- [ ] Per-tenant backup/restore tested
- [ ] Cross-tenant data leakage test **FAILS** (security test must fail)
- [ ] API authentication enabled (no anonymous access)
- [ ] Input validation on all tenant_id parameters
- [ ] Logging captures tenant_id on all operations
- [ ] Monitoring dashboards per tenant (latency, errors, quota)

## Key Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| Per-tenant query latency P95 | >200ms | Investigate isolation overhead |
| Per-tenant error rate | >5% | Check quota limits, API errors |
| Namespace utilization | 72/90 (80%) | Provision new index |
| Cost per tenant | >70% of revenue | Migrate to dedicated index |
| Database connection pool | >80% | Scale database connections |

## Cost Structure (Monthly)

| Scale | Pinecone | PostgreSQL | Compute | Monitoring | **Per-Tenant** |
|-------|----------|------------|---------|------------|----------------|
| 10 tenants | $10 | $20 | $30 | $10 | **$7.00** |
| 100 tenants | $80 | $50 | $200 | $50 | **$3.80** |
| 500 tenants | $400 | $150 | $800 | $200 | **$3.10** |

**Optimization strategies:**
- Batch free/trial tenants on 1-2 indexes (save $100-200/month at 100 tenants)
- Use reserved instances for 40% compute savings at 500+ tenants
- Migrate to dedicated indexes when allocated cost exceeds 70% of subscription

## Critical Limitations

⚠️ **Important constraints:**

1. **Namespace isolation is not physical**: Theoretical Pinecone bugs could leak data across namespaces (hasn't occurred in practice)
2. **Noisy neighbor problems persist**: High-volume tenants can spike on shared indexes
3. **15-25ms overhead**: Isolation checks degrade performance vs single-tenant
4. **Scaling ceiling**: At 1000+ tenants, requires namespace-to-index migration strategy
5. **Not for regulated industries**: HIPAA/FedRAMP require physical isolation (dedicated deployments)

## Troubleshooting

### "Namespace exhaustion" error

**Cause**: Exceeded 90 namespace limit per index

**Fix**:
```python
# Check current usage
print(registry.namespace_usage)  # {'shared-index-1': 92}

# Provision new index and update config
Config.PINECONE_SHARED_INDEX = "shared-index-2"
```

### "Tenant not found" on query

**Cause**: Querying before tenant registration

**Fix**:
```python
# Always register before use
tenant = registry.get_tenant("acme-001")
if not tenant:
    tenant = registry.register_tenant("acme-001", "Acme Corp", TenantTier.FREE)
```

### High query latency (>200ms)

**Cause**: Uncached tenant config lookups

**Fix**: Enable caching in production
```python
# Add caching layer
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_tenant_cached(tenant_id: str):
    return registry.get_tenant(tenant_id)
```

### Cost allocation doesn't sum to 100%

**Cause**: Missing tenants in allocation basis

**Fix**:
```python
# Include ALL active tenants
all_tenants = registry.tenants.keys()
allocation_basis = {tid: get_usage_pct(tid) for tid in all_tenants}
assert sum(allocation_basis.values()) == 100.0
```

## API Reference

### REST Endpoints

```bash
# Health check
GET /health

# Create tenant
POST /tenants
Body: {"tenant_id": "acme-001", "tenant_name": "Acme Corp", "tier": "pro"}

# Get tenant
GET /tenants/{tenant_id}

# Upsert documents
POST /upsert
Body: {"tenant_id": "acme-001", "documents": [...]}

# Query documents
POST /query
Body: {"tenant_id": "acme-001", "query_text": "...", "top_k": 5}

# Get costs
GET /costs/{tenant_id}

# Allocate costs
POST /costs/allocate
Body: {"monthly_fixed_cost": 115.0, "allocation_basis": {...}}

# Test isolation (security test)
POST /security/test-isolation?tenant_a_id=...&tenant_b_id=...
```

## Next Steps

**Module 11.2**: Multi-Region Deployment Strategies
- Geo-distributed tenant routing
- Data residency compliance (GDPR)
- Cross-region replication
- Failover strategies

## Resources

- **Pinecone Namespaces**: https://docs.pinecone.io/docs/namespaces
- **PostgreSQL RLS**: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
- **Multi-Tenant SaaS Best Practices**: https://aws.amazon.com/saas/

## License

Educational use for CCC Level 3 training.
