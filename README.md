# Module 13: Enterprise RAG SaaS - Complete Multi-Tenant Integration

A production-ready, multi-tenant Compliance Copilot SaaS platform integrating configuration management, tenant context propagation, orchestration, failure isolation, and resource attribution.

## Overview

This module demonstrates the complete integration of 12 previous modules (M1-M12) into a cohesive, production-ready system for 5-100 paying customers.

### Core Components

1. **Configuration Layer**: Pydantic + Dynaconf for flexible, cascading settings
2. **Tenant Context Propagation**: OpenTelemetry baggage + ContextVar for identity tracking
3. **Orchestration Pattern**: Central `ComplianceCopilotSaaS` class coordinates all workflows
4. **Resource Attribution**: Every operation tracked per tenant for accurate billing

### Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway                         │
│                    (Authentication)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           ComplianceCopilotSaaS Orchestrator                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Config Mgr   │  │ Usage Track  │  │ Vector Store │      │
│  │ (Cascade)    │  │ (Billing)    │  │ (Namespaced) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Tenant:      │ │ Tenant:      │ │ Tenant:      │
│ acme_corp    │ │ beta_inc     │ │ gamma_labs   │
│ (GPT-4)      │ │ (GPT-3.5)    │ │ (GPT-3.5)    │
│ Namespace A  │ │ Namespace B  │ │ Namespace C  │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run Tests

```bash
# Windows (PowerShell)
powershell -c "$env:PYTHONPATH='$PWD'; pytest -q"
# or
.\scripts\run_tests.ps1

# Unix/Linux/macOS
PYTHONPATH=. pytest -q
# or
./scripts/run_tests.sh
```

### 3. Start API Server

```bash
# Windows (PowerShell)
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
# or
.\scripts\run_api.ps1

# Unix/Linux/macOS
PYTHONPATH=. uvicorn app:app --reload
# or
./scripts/run_api.sh
```

API available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 4. Jupyter Notebook

```bash
jupyter notebook notebooks/L3_M13_Complete_SaaS_Build.ipynb
```

Execute cells sequentially to see:
- Configuration cascade in action
- Tenant context propagation
- Multi-tenant query execution
- Usage tracking & billing
- Failure modes & mitigations

## How It Works

### Configuration Cascade

System defaults → Tenant defaults → Query-level overrides

```python
# System default
config_mgr.set_system_defaults(model_tier=ModelTier.GPT35)

# Tenant override
config_mgr.update_tenant_config("acme_corp", model_tier=ModelTier.GPT4)

# Query-level override
await copilot.query(
    tenant_id="acme_corp",
    query_text="...",
    override_config={"retrieval_mode": RetrievalMode.AGENTIC}
)
```

### Tenant Context Propagation

Identity flows through all async operations:

```python
TenantContext.set_tenant("acme_corp")
# All subsequent operations inherit tenant context
await copilot.query(...)  # Automatically uses acme_corp context
```

Ensures:
- Namespace isolation in Pinecone (no cross-tenant data leakage)
- Accurate billing attribution
- Distributed tracing across services

### Resource Limits

Per-tenant quotas prevent abuse:

```python
resource_limits = ResourceLimits(
    max_queries_per_hour=100,
    max_tokens_per_query=4096,
    max_concurrent_requests=5
)
```

Enforced automatically during query execution.

### Failure Isolation

Single tenant issues don't impact others:

```python
# Tenant A hits rate limit
await copilot.query("tenant_a", "...")  # Raises exception

# Tenant B continues normally
await copilot.query("tenant_b", "...")  # ✓ Success
```

## Common Failures & Fixes

### 1. Cache Race Conditions (Cross-Tenant Leakage)

**Symptom:** Tenant A sees Tenant B's data

**Cause:** Shared cache without tenant isolation

**Fix:** Thread-safe caching with tenant-scoped locks
```python
async with self._lock:
    self._tenant_configs[tenant_id] = config
```

### 2. Cascading Rate Limits

**Symptom:** One heavy tenant blocks others

**Fix:** Per-tenant rate limiting
```python
self._rate_limits.setdefault(tenant_id, [])
recent = [t for t in self._rate_limits[tenant_id] if now - t < 60]
if len(recent) >= 100:
    raise Exception(f"Rate limit exceeded")
```

### 3. Connection Pool Exhaustion

**Symptom:** Timeouts during bulk operations

**Fix:** Connection pooling + request batching

### 4. OpenTelemetry Context Loss

**Symptom:** Tracing breaks mid-chain

**Fix:** Explicit context propagation
```python
baggage.set_baggage("tenant_id", tenant_id)
```

### 5. Async Billing Lag

**Symptom:** Operations complete but billing delayed

**Fix:** Background worker with retry queue

## Decision Card

### ✅ Use This Architecture When:

- **5-100 paying customers** (sweet spot for multi-tenancy)
- **>500ms P95 latency acceptable** (coordination overhead)
- **Need strong tenant isolation** (data privacy requirements)
- **Have DevOps expertise** (infrastructure management)
- **Market size >100** (justifies engineering investment)

### ❌ Avoid This When:

- **<5 customers** → Overhead unjustified, start single-tenant
- **<100 total market** → Overengineered for small opportunity
- **<500ms latency required** → Coordination overhead too high
- **No DevOps team** → Use managed platforms instead
- **MVP stage** → Start simpler, add multi-tenancy later

### 🔄 Alternative Approaches:

1. **MVP-first phasing**: Single-tenant → Add multi-tenancy incrementally
2. **Microservices**: Separate services per component for independent scaling
3. **Managed platforms**: Hosted RAG solutions (e.g., Pinecone Assistant)
4. **Tenant-per-instance**: Single-tenant SaaS copies for premium customers

## Cost Breakdown (Example Monthly)

| Component       | Min    | Max     | Notes                          |
|----------------|--------|---------|--------------------------------|
| Database       | $50    | $200    | PostgreSQL for tenant configs  |
| Vector Store   | $70    | $500    | Pinecone (varies by scale)     |
| LLM APIs       | $100   | $2,000  | OpenAI/Anthropic usage         |
| Observability  | $50    | $300    | OpenTelemetry + monitoring     |
| **Total**      | **$270** | **$3,000** | Per month                   |

## API Reference

### POST /query

Execute RAG query for a tenant.

**Request:**
```json
{
  "tenant_id": "acme_corp",
  "query": "What are our data encryption requirements?",
  "model_tier": "gpt-4",
  "retrieval_mode": "hybrid"
}
```

**Response:**
```json
{
  "answer": "Based on your policies...",
  "sources": [{"id": 0, "score": 0.9}],
  "metadata": {
    "tenant_id": "acme_corp",
    "model": "gpt-4",
    "tokens_used": 250,
    "latency_ms": 1800,
    "retrieval_mode": "hybrid"
  }
}
```

### POST /ingest

Ingest documents for a tenant.

**Request:**
```json
{
  "tenant_id": "acme_corp",
  "documents": [
    {
      "text": "ACME Data Protection Policy...",
      "metadata": {
        "source": "internal_policy",
        "category": "data_protection"
      }
    }
  ]
}
```

### GET /metrics/{tenant_id}

Retrieve usage metrics and costs.

**Response:**
```json
{
  "tenant_id": "acme_corp",
  "time_window_hours": 24,
  "total_queries": 150,
  "successful_queries": 148,
  "avg_latency_ms": 1250.5,
  "costs": {
    "total_tokens": 50000,
    "estimated_llm_cost": 0.5,
    "total_requests": 150,
    "success_rate": 0.987
  }
}
```

## Environment Variables

Configure these environment variables in `.env` (copy from `.env.example`):

```bash
# LLM API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Vector Store (Pinecone)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=compliance-copilot

# Database (PostgreSQL for tenant configs)
DATABASE_URL=postgresql://user:password@localhost:5432/compliance_copilot

# System Defaults
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_RETRIEVAL_MODE=basic
DEFAULT_MAX_TOKENS=4096

# Resource Limits
MAX_QUERIES_PER_HOUR=100
MAX_CONCURRENT_REQUESTS=5
MAX_DOCUMENTS_PER_TENANT=10000
QUERY_TIMEOUT_SECONDS=30.0

# Observability (OpenTelemetry)
OTLP_ENDPOINT=http://localhost:4317
LOG_LEVEL=INFO

# Application Settings
ENVIRONMENT=development
DEBUG=false

# FastAPI Settings
API_HOST=0.0.0.0
API_PORT=8000
```

## Troubleshooting

### "Service not initialized" error

**Cause:** Missing API keys or configuration

**Fix:**
1. Copy `.env.example` to `.env`
2. Fill in API keys
3. Restart server

### Cross-tenant data leakage

**Cause:** Missing tenant context

**Fix:** Always call `TenantContext.set_tenant()` before operations

### High latency (>1000ms P95)

**Causes:**
- Network issues with LLM/vector store
- Large document retrieval
- Unoptimized embeddings

**Fixes:**
1. Cache frequent queries
2. Reduce top_k for retrieval
3. Use faster model tier (GPT-3.5 vs GPT-4)

### Rate limit errors

**Cause:** Tenant exceeds quota

**Fix:** Increase limits in tenant config or upgrade tier

### Offline/Limited Mode

**Behavior:** The module runs in a "degraded" mode if API keys (`OPENAI_API_KEY`, `PINECONE_API_KEY`) are not set in `.env`.

The `config.py` file will return `None` for these clients, and the `app.py` logic will return a `{"skipped": true, "reason": "Service not initialized..."}` response for API calls.

**Use case:** This mode is useful for:
- Running tests without real API calls (smoke tests)
- Local development without API keys
- Educational purposes (simulated/mocked operations)

**To enable live API calls:**
1. Configure API keys in `.env`
2. Restart the API server
3. The system will automatically detect available services

## Practathon Challenges

### Easy (10-15 hours)
- 3 tenants with basic load testing
- 100 requests total
- Basic monitoring

### Medium (15-20 hours)
- Production-ready system
- Comprehensive testing (1000 req/hr)
- Full monitoring dashboard

### Hard (25-30 hours)
- Multi-region deployment
- Automatic failover
- 100+ tenant simulation
- P95 < 500ms at scale

## File Structure

```
ccc_l3_aug_practical/
├── app.py                                  # FastAPI entrypoint
├── config.py                               # Environment & client management
├── src/
│   └── l3_m13_complete_saas_build/
│       └── __init__.py                     # Core module (multi-tenant logic)
├── notebooks/
│   └── L3_M13_Complete_SaaS_Build.ipynb   # Jupyter notebook
├── tests/
│   └── test_m13_complete_saas_build.py    # Smoke tests
├── configs/
│   └── example.json                        # Configuration templates
├── scripts/
│   ├── run_api.ps1                         # Windows: Start API
│   ├── run_api.sh                          # Unix: Start API
│   ├── run_tests.ps1                       # Windows: Run tests
│   └── run_tests.sh                        # Unix: Run tests
├── requirements.txt                        # Dependencies
├── .env.example                            # Environment template
├── .gitignore                              # Git ignore rules
├── example_data.json                       # Sample data
└── README.md                               # This file
```

## Next Steps

1. **Production Deployment**
   - Set up PostgreSQL for tenant configs
   - Configure Pinecone index
   - Deploy to cloud (AWS/GCP/Azure)
   - Set up monitoring dashboards

2. **Security Hardening**
   - API key rotation
   - Request rate limiting
   - DDoS protection
   - Audit logging

3. **Observability**
   - OpenTelemetry integration
   - Custom metrics
   - Alert thresholds
   - Incident runbooks

4. **Performance Optimization**
   - Query caching
   - Connection pooling
   - Async processing
   - Load balancing

## Related Modules

- **Module 1-12**: Individual RAG components integrated here
- **Module 14**: (If applicable) Advanced monitoring & optimization

## License

[Your license here]

## Support

For issues or questions:
- GitHub Issues: [repo-url]/issues
- Documentation: [docs-url]
- Email: support@example.com
