# Module 12: Tenant Lifecycle Management (L3)

Complete implementation of tenant lifecycle operations for SaaS applications, including plan upgrades/downgrades, GDPR-compliant data exports, soft-deletion with retention, and reactivation workflows.

## Overview

This module provides production-ready components for managing the complete tenant lifecycle in multi-tenant SaaS systems. It implements:

- **State Machine**: 8-state lifecycle with validated transitions and audit logging
- **Plan Changes**: Upgrade/downgrade with billing integration and rollback
- **Data Export**: GDPR-compliant chunked exports with signed URLs
- **Soft Deletion**: 30-90 day retention with verification
- **Reactivation**: Win-back workflows with state conflict resolution

**Based on**: Module 12 (Tenant Lifecycle) from CCC Level 3 curriculum

## Quickstart

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Configuration & Environment Variables

See `.env.example` for all available configuration options. Core environment variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/tenant_lifecycle_db

# Redis Configuration (for Celery broker)
REDIS_URL=redis://localhost:6379/0

# Stripe API Configuration
STRIPE_API_KEY=sk_test_your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here

# Lifecycle Configuration
SOFT_DELETE_RETENTION_DAYS=30
DATA_EXPORT_CHUNK_SIZE_MB=50
MAX_CONCURRENT_LIFECYCLE_JOBS=10

# Storage Configuration (for data exports)
EXPORT_STORAGE_TYPE=local
EXPORT_STORAGE_PATH=/tmp/tenant_exports

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Feature Flags
ENABLE_AUTO_DOWNGRADE=true
ENABLE_REACTIVATION_WORKFLOW=true
ENABLE_METRICS=false
```

**Note**: The module works in **limited mode** if `STRIPE_API_KEY`, `DATABASE_URL`, or `REDIS_URL` are not configured. The `config.py` module returns `None` for unconfigured clients, and core logic gracefully skips live API calls (logging warnings instead of failing).

### 3. Run the API

**Windows (PowerShell):**
```powershell
# Using script
.\scripts\run_api.ps1

# Or manually
$env:PYTHONPATH = $PWD
uvicorn app:app --reload
```

**Linux/Mac (Bash):**
```bash
# Using script
./scripts/run_api.sh

# Or manually
export PYTHONPATH=$(pwd)
uvicorn app:app --reload
```

API will be available at http://localhost:8000
Interactive docs at http://localhost:8000/docs

### 4. Run the Notebook

```bash
# Launch Jupyter Lab
jupyter lab notebooks/L3_M12_Tenant_Lifecycle_Management.ipynb

# Or Jupyter Notebook
jupyter notebook notebooks/
```

### 5. Run Tests

**Windows (PowerShell):**
```powershell
.\scripts\run_tests.ps1
```

**Linux/Mac (Bash):**
```bash
./scripts/run_tests.sh
# Or directly
pytest -q tests/
```

## How It Works

### Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────┐
│                     Tenant Lifecycle Flow                    │
└─────────────────────────────────────────────────────────────┘

   ┌─────────┐
   │ ACTIVE  │ ◄──────────────────┐
   └────┬────┘                    │
        │                         │
        ├─► UPGRADING ────────────┤
        │                         │
        ├─► DOWNGRADING ──────────┤
        │                         │
        ├─► EXPORTING ────────────┘
        │
        ├─► SUSPENDED
        │       │
        │       └─► REACTIVATING ──► ACTIVE
        │
        └─► DELETING ──► DELETED
                          (30-90 day retention)

Plan Hierarchy: FREE → STARTER → PROFESSIONAL → ENTERPRISE
```

### State Transitions

| From State   | Valid Transitions                                      |
|--------------|-------------------------------------------------------|
| ACTIVE       | UPGRADING, DOWNGRADING, SUSPENDED, EXPORTING, DELETING |
| UPGRADING    | ACTIVE, SUSPENDED                                     |
| DOWNGRADING  | ACTIVE, SUSPENDED                                     |
| SUSPENDED    | REACTIVATING, DELETING                                |
| EXPORTING    | ACTIVE                                                |
| DELETING     | DELETED                                               |
| DELETED      | (none - use reactivation workflow)                    |
| REACTIVATING | ACTIVE, SUSPENDED                                     |

### Core Operations

#### 1. Upgrade Tenant

```python
from src.l3_m12_tenant_lifecycle_management import upgrade_tenant
from config import Config

result = upgrade_tenant(
    tenant_data={
        "tenant_id": "tenant_001",
        "name": "Acme Corp",
        "email": "admin@acme.com",
        "current_plan": "free",
        "state": "active"
    },
    to_plan="starter",
    plan_hierarchy=Config.PLAN_HIERARCHY,
    plan_limits=Config.PLAN_LIMITS
)
```

**Process**:
1. Validates plan hierarchy (free → starter is valid upgrade)
2. Provisions new resources **before** billing changes
3. Updates Stripe subscription with proration
4. Rolls back on failure to avoid partial upgrades

#### 2. Downgrade Tenant

```python
from src.l3_m12_tenant_lifecycle_management import downgrade_tenant

result = downgrade_tenant(
    tenant_data={...},
    to_plan="free",
    plan_hierarchy=Config.PLAN_HIERARCHY,
    plan_limits=Config.PLAN_LIMITS
)
```

**Process**:
1. Validates current usage fits new plan limits
2. Schedules Stripe changes for billing period end
3. Safely reduces resource allocations
4. Prevents data loss during downgrade

#### 3. Export Data (GDPR)

```python
from src.l3_m12_tenant_lifecycle_management import export_tenant_data

result = export_tenant_data(
    tenant_data={...},
    export_type="full"
)
# Returns: {"export_id": "...", "status": "queued", "estimated_completion": "..."}
```

**Process**:
1. Initiates background Celery job
2. Chunks data into manageable pieces (50MB default)
3. Generates ZIP archive with checksum
4. Creates signed URL with 7-day expiration

#### 4. Soft Delete

```python
from src.l3_m12_tenant_lifecycle_management import delete_tenant

result = delete_tenant(
    tenant_data={...},
    requested_by="admin@example.com"
)
# Returns: {"deletion_id": "...", "can_reactivate": true, "retention_days": 30}
```

**Process**:
1. Marks tenant as DELETED (soft-delete)
2. Sets retention period (30-90 days)
3. Schedules hard delete after retention expires
4. Allows reactivation during retention window

#### 5. Reactivate

```python
from src.l3_m12_tenant_lifecycle_management import reactivate_tenant

result = reactivate_tenant(
    tenant_data={...},
    reactivation_plan="starter"
)
```

**Process**:
1. Verifies tenant is within retention period
2. Restores soft-deleted data
3. Reactivates or creates Stripe subscription
4. Transitions state to ACTIVE

## Common Failures & Fixes

### 1. Upgrade Service Interruption

**Symptom**: Users experience downtime during plan upgrade

**Root Cause**: Resources provisioned after billing changes, creating gap in service

**Fix**: Always provision resources **before** updating billing
```python
# Correct order in execute_upgrade():
1. _provision_resources()  # First
2. _update_stripe_subscription()  # Second
3. Update tenant metadata
```

### 2. Incomplete Data Export

**Symptom**: Export job times out or returns partial data

**Root Cause**: Attempting to export large datasets in single operation

**Fix**: Use chunked exports with background jobs
```python
service = DataExportService(chunk_size_mb=50)  # Smaller chunks
result = service.initiate_export(tenant, export_type="full")
# Process in background with Celery
```

### 3. Deletion Verification Failures

**Symptom**: Data remains after hard delete completion

**Root Cause**: Deletion not verified across all data stores

**Fix**: Multi-stage verification in hard_delete()
```python
verification_steps = [
    {"step": "database_records", "deleted": True},
    {"step": "storage_files", "deleted": True},
    {"step": "stripe_subscription", "cancelled": True},
    {"step": "cache_entries", "deleted": True}
]
```

### 4. Reactivation State Conflicts

**Symptom**: Reactivation fails with "invalid state" error

**Root Cause**: Attempting to reactivate after retention period or from invalid state

**Fix**: Always check can_reactivate() first
```python
workflow = ReactivationWorkflow()
can_reactivate, reason = workflow.can_reactivate(tenant)
if not can_reactivate:
    return {"error": reason}
```

### 5. Retention Policy Violations

**Symptom**: Tenant reactivation fails because data was prematurely deleted

**Root Cause**: Hard delete executed before retention period ended

**Fix**: Automated retention enforcement
```python
# Use scheduled tasks to enforce retention
deleted_time = datetime.fromisoformat(tenant.deleted_at)
retention_limit = deleted_time + timedelta(days=retention_days)
if datetime.utcnow() < retention_limit:
    # Still within retention - allow reactivation
```

## Decision Card

### When to Use Automated Lifecycle

✅ **Use when**:
- 10+ tenants with frequent plan changes
- Regulated industry requiring compliance audit trails
- Self-service model demanded by market
- Need to scale lifecycle operations without CSM overhead

❌ **Don't use when**:
- <10 tenants (manual SQL scripts sufficient)
- Enterprise-only with <5 annual transitions
- High-touch relationships requiring personal coordination
- Instant exports required (<5 min response time)

### Alternatives

| Approach | When to Use | Trade-offs |
|----------|-------------|------------|
| **Manual CSM-Driven** | Enterprise-only, high-touch | Lower error risk but higher overhead |
| **Upgrade-Only** | Forbid downgrades | Simpler but impacts retention |
| **Hard Deletes** | Non-regulated B2C | Violates GDPR recovery rights |
| **Managed Platforms** | BuilderKit/Supabase | Less control but turnkey solution |

## Troubleshooting

### Offline/Limited Mode

**Description**: The module runs in a limited mode if API keys (`STRIPE_API_KEY`, `DATABASE_URL`, `REDIS_URL`) are not set in `.env`.

**Behavior**: The `config.py` file will return `None` for unconfigured clients, and the core logic/API will gracefully skip live API calls. For example:
- Upgrade/downgrade operations succeed but skip Stripe subscription updates (logged as warnings)
- Data exports work but skip cloud storage uploads
- Reactivation works but skips billing subscription creation

**When to Use**:
- Local development and testing without external services
- CI/CD pipelines running smoke tests
- Educational/demo environments
- Jupyter notebooks in offline mode (set `OFFLINE=true` environment variable)

**Note**: All operations return success responses with `{"skipped": true}` metadata when running in limited mode.

### Stripe Not Configured

**Symptom**: Operations succeed but billing not updated

**Cause**: STRIPE_API_KEY not set in .env

**Fix**:
```bash
# Add to .env
STRIPE_API_KEY=sk_test_your_key_here

# Restart API
python app.py
```

### Downgrade Blocked

**Symptom**: `"Current usage exceeds plan limits"` error

**Cause**: Tenant using more resources than target plan allows

**Fix**: Have tenant reduce usage first, or contact support for migration plan
```python
# Check current usage
tenant.current_usage  # {"users": 50, "storage_gb": 40}

# Target plan limits
plan_limits["starter"]  # {"users": 20, "storage_gb": 10}

# User must remove 30 users before downgrading
```

### Export Timeout

**Symptom**: Export job never completes

**Cause**: Dataset too large for configured timeout

**Fix**: Reduce chunk size or increase timeout
```bash
# In .env
DATA_EXPORT_CHUNK_SIZE_MB=25  # Reduce from 50
```

### Retention Period Confusion

**Symptom**: Can't determine when hard delete will occur

**Fix**: Check deletion record
```python
deletion_record = delete_tenant(tenant_data, "admin@example.com")
print(deletion_record["hard_delete_scheduled_at"])
# "2024-12-07T08:05:00Z"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upgrade` | POST | Upgrade tenant to higher plan |
| `/downgrade` | POST | Downgrade tenant to lower plan |
| `/export` | POST | Initiate GDPR data export |
| `/delete` | POST | Soft-delete tenant |
| `/reactivate` | POST | Reactivate suspended/deleted tenant |
| `/plans` | GET | Get available plans and limits |
| `/config` | GET | Get current configuration |
| `/metrics` | GET | Prometheus metrics (if enabled) |

**Full API docs**: http://localhost:8000/docs

## Cost Considerations

**Per 100 lifecycle events/month**:
- Celery workers: ~$50/month (2 workers on cloud VMs)
- Redis: ~$15/month (managed Redis instance)
- Storage (exports): ~$10/month (S3 or equivalent)
- Stripe API calls: Free (included in Stripe fees)

**Total**: ~$75/month for 100 events/month

## Monitoring

**Key Metrics**:
- State transition failures
- Upgrade/downgrade duration
- Export completion rate
- Deletion verification success
- Reactivation success rate

**Setup with Prometheus** (optional):
```bash
# Enable in .env
ENABLE_METRICS=true

# Metrics available at /metrics
curl http://localhost:8000/metrics
```

## Next Steps

- **Module 12.5**: Usage tracking and billing automation
- **Module 13**: Advanced monitoring and observability

## License

Educational use only - CCC Level 3 Module 12

## Support

For issues or questions about this module, refer to the course materials or augmented script: `augmented_m12_videom12_4_Tenant_Lif.md`
