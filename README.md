# Module 11.2: Tenant-Specific Customization

Database-backed multi-tenant configuration management for RAG pipelines. Enables SaaS products to serve diverse customer needs without code deployments.

## Overview

This module implements configuration-driven RAG pipelines where each tenant can customize:
- **Model Selection**: GPT-4, GPT-3.5, Claude variants
- **Retrieval Parameters**: top_k, alpha, reranking
- **Prompt Templates**: Safe variable injection with validation
- **Resource Limits**: Max tokens, temperature bounds
- **Branding**: Custom UI colors

**Key Pattern**: Database-backed configuration with Redis caching eliminates hardcoded tenant-specific if-statements that don't scale beyond 5-10 tenants.

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database and Redis URLs
```

### 3. Run Examples

```bash
# CLI examples
python l2_m11_tenant_specific_customization.py

# Start FastAPI server
python app.py

# API docs: http://localhost:8000/docs
```

### 4. Run Tests

```bash
pytest tests_smoke.py -v
```

### 5. Explore Jupyter Notebook

```bash
jupyter lab L2_M11_Tenant-Specific_Customization.ipynb
```

## How It Works

```
┌─────────────────┐
│  API Request    │
│  (tenant_id)    │
└────────┬────────┘
         │
         v
┌─────────────────┐     Cache Miss
│  Redis Cache    ├──────────────┐
│  (TTL: 300s)    │              │
└────────┬────────┘              │
         │ Cache Hit             v
         │              ┌─────────────────┐
         │              │   PostgreSQL    │
         │              │ tenant_configs  │
         │              └────────┬────────┘
         │                       │
         v                       v
┌─────────────────────────────────┐
│  Pydantic Validation            │
│  - Model whitelist              │
│  - Bounded parameters           │
│  - Injection prevention         │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  RAG Pipeline                   │
│  - Apply model settings         │
│  - Render prompt template       │
│  - Configure retrieval          │
└─────────────────────────────────┘
```

### Architecture Components

1. **Database Schema**: `tenant_configurations` table with JSONB columns for flexible storage
2. **Pydantic Models**: Type-safe validation with bounded numeric fields and model whitelists
3. **Repository Pattern**: Caching layer with TTL-based invalidation and fallback to defaults
4. **RAG Integration**: Applies configurations to model selection, temperature, and retrieval parameters
5. **Management API**: CRUD endpoints with audit trails

## Common Failures & Fixes

### 1. Configuration Conflicts
**Symptom**: Tenant A's GPT-4 requests with Tenant B's budget limits collide

**Fix**: Always validate tenant_id in request pipeline; implement tenant isolation checks

```python
# BAD: Shared global config
global_config = load_config()

# GOOD: Per-request tenant config
config = repository.get_config(request.tenant_id)
```

### 2. Default Override Failures
**Symptom**: Partial updates don't merge correctly with defaults

**Fix**: Use `merge=True` for partial updates, `merge=False` for full replacement

```python
# Partial update (merge with existing)
repository.update_config("tenant_id", {"temperature": 0.8}, merge=True)

# Full replacement (reset other fields to defaults)
repository.update_config("tenant_id", complete_config, merge=False)
```

### 3. Cache Staleness
**Symptom**: Configuration updates don't propagate to running processes

**Fix**: Explicit cache invalidation on updates; monitor cache hit rates

```python
# Repository automatically invalidates on update
repository.update_config(tenant_id, updates)  # Cache deleted internally

# Manual invalidation if needed
redis_client.delete(f"tenant_config:{tenant_id}")
```

### 4. No Rollback Mechanism
**Symptom**: Bad configurations break tenants until manual intervention

**Fix**: Implement configuration versioning (see Practathon challenge)

**Workaround**: Keep backups before updates
```python
# Store previous config
backup = repository.get_config(tenant_id).model_dump()
try:
    repository.update_config(tenant_id, new_config)
except Exception:
    repository.update_config(tenant_id, backup, merge=False)
```

### 5. Isolation Testing
**Symptom**: Hard to test tenant configs in parallel without cross-contamination

**Fix**: Use unique tenant IDs in tests; clean up after each test

```python
import uuid

def test_config():
    tenant_id = f"test_tenant_{uuid.uuid4()}"
    # Test logic
    repository.delete_config(tenant_id)  # Cleanup
```

## Decision Card (TVH v2.0)

### Use This Pattern When:
✅ Managing **10-100+ tenants** with varied needs
✅ Revenue models tied to **feature differentiation**
✅ Need **self-service configuration** capabilities
✅ Deployment cycle too slow for frequent changes
✅ Customer requests exceed product roadmap capacity

### Use Alternatives When:
❌ **Standardization** simplifies product (startups MVP)
❌ Configuration changes rare (**<monthly**)
❌ **Cost control** more important than customization
❌ Serving **<10 tenants** with similar needs
❌ Team lacks database/cache infrastructure expertise

### Trade-offs Accepted:
- **Database queries** for every request (mitigated by caching)
- **Eventually consistent** configuration updates
- **Feature flag complexity** for gradual rollouts
- **Storage costs** scale with tenant count

### When It Breaks:
🔴 **>1000 concurrent tenants** with frequent config changes
🔴 **Microsecond-level latency** requirements
🔴 Configurations requiring **real-time synchronization** across regions
🔴 **Cost explosions** if tenants request expensive models without plan enforcement

### Alternative Approaches:

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **LaunchDarkly** | Managed service, A/B testing | $50-200/month, vendor lock-in | Feature flags, experimentation |
| **Config-as-Code (YAML/Git)** | Version control, auditable | Requires deployments | Infrastructure-focused teams |
| **Single Configuration** | Simple, low overhead | No customization | Standardized products |
| **This Pattern** | Scalable, self-service | Database/cache required | Multi-tenant SaaS (10-100+ tenants) |

## API Reference

### GET /health
Health check with service availability

### GET /config/{tenant_id}
Get tenant configuration (returns default if not found)

### POST /config/{tenant_id}
Update tenant configuration
```json
{
  "config": {
    "model": "gpt-4",
    "temperature": 0.5,
    "top_k": 10
  },
  "merge": true
}
```

### DELETE /config/{tenant_id}
Delete tenant configuration (revert to defaults)

### GET /tenants
List all configured tenants

### POST /query
Execute RAG query with tenant config
```json
{
  "tenant_id": "tenant_startup_co",
  "query": "What is machine learning?"
}
```

## Validation Rules

### Model Whitelist
- `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### Parameter Bounds
- **Temperature**: 0.0 - 2.0
- **Top-K**: 1 - 20
- **Alpha**: 0.0 - 1.0 (0=keyword, 1=semantic)
- **Max Tokens**: 1 - 4000

### Security
- **Hex Colors**: `#RRGGBB` format only
- **Prompt Injection**: Blocks `ignore previous instructions`, `<script>`, `javascript:`

## Production Considerations

### Scaling
- Handles **100+ tenants** comfortably
- Caching layer critical above **500 tenants**
- Consider read replicas for >1000 tenants

### Cost Breakdown (Monthly)
- Database storage: ~$10
- Redis cache: ~$15
- Monitoring: ~$20
- **Total infrastructure**: ~$45 (excluding LLM costs)

### Monitoring Requirements
- Configuration load times (p50, p95, p99)
- Cache hit rates (target: >90%)
- Validation error rates
- Model cost per tenant

### Deployment Checklist
- [ ] Database migrations tested
- [ ] Validation rules prevent abuse
- [ ] Cache invalidation strategy documented
- [ ] Audit trail for compliance
- [ ] Rollback procedures for configuration changes
- [ ] Cost alerts per tenant
- [ ] Rate limiting per tenant tier

## Practathon Challenges

### 🟢 Easy (60-90 min)
Add a new configuration field for `max_retries` and validate it prevents negative values.

**Hints**:
1. Add field to `TenantConfig` with `ge=0` validator
2. Update `example_data.json`
3. Add test in `tests_smoke.py`

### 🟡 Medium (2-3 hrs)
Implement configuration versioning with rollback capability.

**Requirements**:
- Store history of configuration changes
- API endpoint: `POST /config/{tenant_id}/rollback?version=N`
- Limit history to last 10 versions

### 🔴 Hard (4-5 hrs)
Build multi-region configuration synchronization with conflict resolution.

**Requirements**:
- Propagate config changes across 3+ regions
- Handle conflicting updates (last-write-wins or merge strategies)
- Monitor replication lag

## Troubleshooting

### Issue: "Cache hit rate <50%"
**Cause**: TTL too short or frequent config changes
**Fix**: Increase `REDIS_TTL` or batch updates

### Issue: "Validation error: Model not approved"
**Cause**: Requesting model not in whitelist
**Fix**: Add to `Config.APPROVED_MODELS` or choose approved model

### Issue: "Database connection failed"
**Cause**: Invalid `DATABASE_URL`
**Fix**: Check `.env` and database server status; module falls back to in-memory store

### Issue: "Redis connection refused"
**Cause**: Redis server not running
**Fix**: Start Redis or set `REDIS_URL=` to disable caching

## Next Module

[Module 11.3: Resource Management & Throttling](../M11_3_Resource_Management_Throt.md)
Learn to enforce tenant-specific rate limits and resource quotas to prevent abuse and cost overruns.

## Key Takeaway

> "Database-driven configuration separates infrastructure from customization logic, enabling SaaS products to serve diverse customer needs without code deployments."

When you need to support 10+ tenants with different requirements, this pattern provides the flexibility to scale without sacrificing maintainability or deployment velocity.
