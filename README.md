# Module 12: Usage Metering & Analytics

**Production-grade usage metering for multi-tenant SaaS applications**

## Overview

This module implements a complete usage metering system that bridges the gap between having a multi-tenant RAG API and being able to bill customers accurately or detect quota overages.

**Key Capabilities:**
- Event capture with tenant attribution at API endpoints
- Append-only storage in ClickHouse (columnar OLAP database)
- Materialized views for real-time aggregations (hourly, daily)
- Cost calculation with configurable pricing
- Quota enforcement and overage detection
- Monthly billing export for invoice generation

**Performance Characteristics:**
- **10-100x faster** than PostgreSQL for aggregations
- **<5ms overhead** per request (async buffering)
- Handles **10,000+ events/day** per tenant
- **36-month retention** for compliance

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your ClickHouse credentials (optional for local dev)
```

### 3. Run Tests

```bash
# Windows PowerShell
powershell -c "$env:PYTHONPATH='$PWD'; pytest -q"
# or
./scripts/run_tests.ps1

# Linux/Mac
export PYTHONPATH=$PWD && pytest -q
```

### 4. Start the API

```bash
# Windows PowerShell
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
# or
./scripts/run_api.ps1

# Linux/Mac
export PYTHONPATH=$PWD && uvicorn app:app --reload

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Explore the Jupyter Notebook

```bash
jupyter lab notebooks/L3_M12_Usage_Metering_Analytics.ipynb
# or
jupyter notebook notebooks/L3_M12_Usage_Metering_Analytics.ipynb
```

---

## Environment Variables

The module requires the following environment variables (see `.env.example`):

```bash
# ClickHouse Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=metering

# Buffering Configuration (for <5ms overhead)
BUFFER_SIZE=100
FLUSH_INTERVAL_SECONDS=1.0

# Fallback Storage (when ClickHouse unavailable)
FALLBACK_FILE_PATH=./usage_events_fallback.jsonl

# Pricing Configuration (per-unit costs in USD)
PRICE_PER_QUERY=0.01
PRICE_PER_1K_INPUT_TOKENS=0.003
PRICE_PER_1K_OUTPUT_TOKENS=0.015
PRICE_PER_GB_STORAGE=0.10

# Default Quotas (per tenant per day)
DEFAULT_QUOTA_QUERIES=1000
DEFAULT_QUOTA_TOKENS=100000

# Retention Policy (months)
RETENTION_MONTHS=36

# Grafana (optional - for dashboards)
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=

# Offline Mode (optional - for L3 notebook consistency)
OFFLINE=false
```

---

## How It Works

### Architecture Diagram (Text)

```
┌─────────────────┐
│  API Endpoint   │ ─┐
│ (Multi-Tenant)  │  │
└─────────────────┘  │
                     │ <5ms overhead
┌─────────────────┐  │
│ Usage Tracker   │ ◄┘
│ (Async Buffer)  │
│  100 events or  │
│  1-sec interval │
└────────┬────────┘
         │
         ├──► ClickHouse ──► Materialized Views
         │    (Primary)       (Hourly/Daily)
         │                         │
         │                         ├──► Grafana Dashboard
         │                         ├──► Quota Checker
         │                         └──► Billing Exporter
         │
         └──► Local JSONL
              (Fallback)
```

### Data Flow

1. **Event Capture**: API endpoints emit usage events with tenant_id
2. **Async Buffering**: Events queue in memory (non-blocking)
3. **Batch Insert**: Flush to ClickHouse every 100 events or 1 second
4. **Aggregation**: Materialized views pre-compute hourly/daily totals
5. **Querying**: Dashboards and billing queries hit aggregated data
6. **Fallback**: If ClickHouse down, write to local JSONL files

### Key Components

**1. ClickHouse Schema** (`src/l3_m12_usage_metering_analytics/__init__.py`)
- `usage_events`: Append-only event log, partitioned monthly
- `usage_hourly`: Materialized view for real-time aggregations
- `usage_daily`: Daily summaries per tenant
- `tenant_quotas`: Quota limits and consumption tracking

**2. Usage Tracker** (`src/l3_m12_usage_metering_analytics/__init__.py`)
- Async buffering with <5ms overhead
- Batch insertion (100 events or 1-second intervals)
- Fallback to local JSONL files if ClickHouse unavailable

**3. Cost Calculator** (`src/l3_m12_usage_metering_analytics/__init__.py`)
- Maps usage quantities to billable amounts
- Configurable pricing model (queries, tokens, storage)

**4. Quota Manager** (`src/l3_m12_usage_metering_analytics/__init__.py`)
- Real-time quota checking
- Overage detection with configurable enforcement

**5. Billing Exporter** (`src/l3_m12_usage_metering_analytics/__init__.py`)
- Monthly invoice generation
- Line-item breakdown by event type

---

## Common Failures & Fixes

### 1. Metering Data Loss
**Symptom:** Events not written if ClickHouse unavailable
**Fix:** System automatically falls back to local file buffering (`usage_events_fallback.jsonl`). Replay these events when ClickHouse recovers.

```bash
# Check for fallback file
ls -lh usage_events_fallback.jsonl

# Replay events (custom script needed)
python replay_fallback_events.py
```

### 2. Aggregation Bugs
**Symptom:** Usage calculation discrepancies between raw events and summaries
**Fix:** Regular reconciliation queries comparing event sums against materialized view totals

```sql
-- Reconciliation query
SELECT
  tenant_id,
  sum(quantity) as raw_total
FROM usage_events
WHERE toDate(timestamp) = today()
GROUP BY tenant_id;
```

### 3. Dashboard Performance Degradation
**Symptom:** Slow queries for large tenants with years of history
**Fix:** Already implemented via monthly partitioning and pre-aggregated views. For extremely large tenants, add tenant-specific partitioning.

### 4. Overage Detection Delays
**Symptom:** Tenants hit quotas without real-time notification
**Fix:** Check materialized view refresh settings. Consider sub-minute aggregation windows for critical quotas.

### 5. Billing Reconciliation Errors
**Symptom:** Invoiced amounts don't match usage records
**Fix:** Immutable event log allows auditing. Query raw events for the period:

```python
# Regenerate invoice from raw events
invoice = billing_exporter.export_monthly_invoice("tenant_id", 2025, 11)
```

---

## Decision Card

### ✅ Use This System When:

1. **50+ customers** requiring billing transparency
2. **Significant per-tenant usage variation** (10x+ differences)
3. **Usage-based pricing** is core to business model
4. **10,000+ events/day** across all tenants
5. **Audit requirements** demand complete event history

### ❌ Do NOT Use When:

1. **Early-Stage MVP (<50 customers)**
   - Alternative: Manual tracking or simple database queries
   - Reason: Infrastructure overhead unjustified

2. **Low-Volume Internal Tools (<100 queries/day)**
   - Alternative: Flat-rate or seat-based pricing
   - Reason: Billing complexity premature

3. **High-Frequency Trading (>10K requests/second)**
   - Alternative: Specialized stream processing (Kafka + Flink)
   - Reason: Event volume overwhelms typical metering

### 🎯 Quick Decision Test

**Use this system if:**
50+ tenants × variable usage × need billing transparency = **YES**

### 💰 Production Costs

- **Infrastructure:** $50-185/month (ClickHouse instance for 50 tenants)
- **Engineering:** 4-8 hours implementation + ongoing monitoring
- **Critical Metrics:** Event write latency, buffer flush rate, aggregation lag

---

## Troubleshooting

### ClickHouse Connection Failed

```
⚠️ ClickHouse connection failed: Connection refused
```

**Fix:**
1. Ensure ClickHouse is running: `docker ps | grep clickhouse`
2. Check connection settings in `.env`
3. System will fall back to local file storage automatically

### High Buffer Size Warning

```
WARNING: Buffer size exceeds 500 events
```

**Fix:**
1. Check ClickHouse write performance
2. Reduce `FLUSH_INTERVAL_SECONDS` in `.env`
3. Scale ClickHouse instance if needed

### Offline/Limited Mode

**Offline/Limited Mode**: The module runs in a limited mode if ClickHouse credentials are not set in `.env`. The `config.py` file will return `None` for the client, and the `UsageTracker` will automatically fall back to writing events to the local `usage_events_fallback.jsonl` file. API endpoints like `/quota/check` will return a 'skipped' response.

**To enable full mode:**
1. Set up ClickHouse (Docker: `docker run -d -p 9000:9000 clickhouse/clickhouse-server`)
2. Configure credentials in `.env`
3. Restart the API

### Invoice Generation Returns Empty

**Fix:**
1. Verify events exist for the period:
   ```sql
   SELECT count() FROM usage_events
   WHERE tenant_id = 'tenant_acme'
     AND toYYYYMM(timestamp) = 202511;
   ```
2. Check date range parameters (month is 1-indexed)

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Track Usage Event
```bash
curl -X POST http://localhost:8000/track \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_acme",
    "event_type": "query",
    "quantity": 1,
    "metadata": {"endpoint": "/api/search"}
  }'
```

### Check Quota
```bash
curl -X POST http://localhost:8000/quota/check \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "tenant_acme"}'
```

### Set Quota
```bash
curl -X POST http://localhost:8000/quota/set \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_acme",
    "queries_per_day": 1000,
    "tokens_per_day": 100000,
    "storage_gb": 10.0
  }'
```

### Generate Invoice
```bash
curl -X POST http://localhost:8000/invoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_acme",
    "year": 2025,
    "month": 11
  }'
```

---

## File Structure

```
.
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── .env.example                                      # Environment variables template
├── .gitignore                                        # Git ignore patterns
├── config.py                                         # Configuration management (root)
├── app.py                                            # FastAPI wrapper (root, no business logic)
├── example_data.json                                 # Sample events and quotas
├── LICENSE                                           # License file
├── src/
│   └── l3_m12_usage_metering_analytics/
│       └── __init__.py                               # Core module (all business logic)
├── notebooks/
│   └── L3_M12_Usage_Metering_Analytics.ipynb        # Interactive tutorial
├── tests/
│   └── test_m12_usage_metering_analytics.py         # Smoke tests
├── configs/
│   └── example.json                                  # Configuration examples
├── scripts/
│   ├── run_api.ps1                                   # Windows: Start API
│   └── run_tests.ps1                                 # Windows: Run tests
└── usage_events_fallback.jsonl                       # Fallback storage (auto-created)
```

---

## Next Module

**Module 12.2:** Advanced Analytics & Dashboarding with Grafana

Build real-time dashboards for per-tenant usage visibility and automated alerting.

---

## Practathon Challenges

- **Easy (60-90 min):** Implement basic query metering and daily aggregations
- **Medium (2-3 hours):** Add token tracking, cost calculation, and Grafana dashboard
- **Hard (5-6 hours):** Full integration with overage detection, email alerts, and billing export

---

## Additional Resources

- **ClickHouse Documentation:** https://clickhouse.com/docs
- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **Module Video:** [Link to M12.1 Video]
- **Source Script:** `augmented_M12_VideoM12_1_Usage_Mete.md`

---

**Module Duration:** 40 minutes
**Audience:** Level 3 learners (post-Module 11)
**Prerequisites:** Level 2 Module 7 (Observability) + Module 11 (Multi-Tenant Architecture)
