# Module 7.2: Application Performance Monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep code-level profiling for RAG systems using Datadog APM**
>
> From "this span is slow" to "line 247 is the bottleneck" in 30 seconds

## Overview

This module demonstrates how to integrate **Datadog APM (Application Performance Monitoring)** with OpenTelemetry-instrumented RAG systems to identify code-level bottlenecks beyond what distributed tracing reveals.

### What This Does

- **Code-level profiling**: See which functions consume CPU/memory down to specific line numbers
- **Memory leak detection**: Track memory growth patterns and identify leaking objects
- **Database query profiling**: Analyze actual SQL queries with execution plans
- **OpenTelemetry bridge**: Works alongside M7.1 distributed tracing without duplication

### What This DOESN'T Do

- Replace code optimization (APM shows problems, you still fix them)
- Eliminate need for load testing (profiling shows behavior, not capacity)
- Diagnose network issues (use distributed tracing for that)

---

## Quick Start

### Prerequisites

- Python 3.8+
- Completed Level 1 M2.3 (Prometheus/Grafana)
- Completed Level 2 M7.1 (OpenTelemetry Tracing)
- Datadog account (14-day free trial available)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your DD_API_KEY from https://app.datadoghq.com/organization-settings/api-keys
```

### Basic Usage

```bash
# Run smoke tests
pytest tests/test_m7_application_performance_monitoring.py -v

# Run demo via API
python app.py

# Start FastAPI server
python app.py
# Visit http://localhost:8000/docs for API documentation
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Process query with APM profiling
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the compliance requirements?", "user_id": "test_user"}'

# Get memory statistics
curl http://localhost:8000/memory/stats

# Run memory leak detection
curl -X POST http://localhost:8000/memory/leak-check \
  -H "Content-Type: application/json" \
  -d '{"iterations": 10}'
```

---

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Your RAG Application                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │  OpenTelemetry   │ ◄─────► │  Datadog APM     │    │
│  │  (M7.1 Tracing)  │  Bridge │  (Profiling)     │    │
│  └──────────────────┘         └──────────────────┘    │
│           │                            │               │
│           └────────────┬───────────────┘               │
│                        ▼                                │
│              ┌──────────────────┐                      │
│              │  Profiled RAG    │                      │
│              │  Pipeline        │                      │
│              ├──────────────────┤                      │
│              │ @tracer.wrap()   │                      │
│              │ decorators       │                      │
│              └──────────────────┘                      │
│                        │                                │
│           ┌────────────┼────────────┐                  │
│           ▼            ▼            ▼                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│   │ CPU      │  │ Memory   │  │ Query    │           │
│   │ Profiling│  │ Profiling│  │ Profiling│           │
│   └──────────┘  └──────────┘  └──────────┘           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Datadog UI      │
              │  - Flame Graphs  │
              │  - Trace Details │
              │  - Memory Trends │
              └──────────────────┘
```

### Observability Pyramid

```
┌─────────────────────────────────────────┐
│ APM (WHY at code level) ◄── M7.2       │ ◄── You are here
├─────────────────────────────────────────┤
│ Traces (WHERE in pipeline) ◄── M7.1    │
├─────────────────────────────────────────┤
│ Metrics (HOW MUCH) ◄── M2.3            │
├─────────────────────────────────────────┤
│ Logs (WHAT happened) ◄── M2.3          │
└─────────────────────────────────────────┘
```

### Key Components

1. **APMManager** (`l2_m7_application_performance_monitoring.py:53`)
   - Initializes Datadog tracer with OpenTelemetry bridge
   - Manages continuous profiler lifecycle
   - Enforces production safety limits (max 5% CPU overhead)

2. **ProfiledRAGPipeline** (`l2_m7_application_performance_monitoring.py:145`)
   - RAG pipeline with `@tracer.wrap()` decorators
   - Custom span tagging (user_id, query_length)
   - Simulates O(n²) bottleneck for APM detection

3. **MemoryProfiledComponent** (`l2_m7_application_performance_monitoring.py:332`)
   - Memory leak detection with `tracemalloc`
   - Per-function memory allocation tracking
   - Alert on large allocations (>100MB)

---

## Common Failures & Fixes

### Failure 1: APM Overhead Crushing Performance (5-15% slowdown)

**Symptom:**
```
P95 latency increased from 800ms to 1.2s after enabling APM
CPU usage increased from 70% to 95%
```

**Root Cause:**
Too aggressive profiling configuration (10% capture, 100% sampling)

**Fix:**
```python
# config.py - Production-safe defaults
DD_PROFILING_CAPTURE_PCT = 1  # ✅ 1% instead of 10%
DD_TRACE_SAMPLE_RATE = 0.1     # ✅ 10% instead of 100%
DD_PROFILING_MAX_TIME_USAGE_PCT = 5  # ✅ Safety limit
```

**Prevention:**
- Always start with 1% profiling
- Set `DD_PROFILING_MAX_TIME_USAGE_PCT=5` safety limit
- Load test with APM enabled BEFORE production

---

### Failure 2: Profiling Crashes Application (OOM)

**Symptom:**
```bash
MemoryError: Cannot allocate memory
Process killed by OOM killer
```

**Root Cause:**
Using `@profile` decorator from `memory_profiler` library in production with large datasets

**Fix:**
```python
# ❌ NEVER in production
from memory_profiler import profile

@profile  # Line-by-line tracking = huge overhead
def process_batch(docs):
    ...

# ✅ Use Datadog's sampling-based profiling instead
DD_PROFILING_MEMORY_ENABLED = True
# No decorator needed - automatic sampling
```

**Prevention:**
- Never use `@profile` decorator in production
- Use sampling-based profilers (Datadog, py-spy)
- Profile in staging with production-sized datasets

---

### Failure 3: Memory Leak Detection Challenges

**Symptom:**
```
Slow memory growth over hours/days
Memory baseline: 200MB → After 24h: 1.2GB
```

**Root Cause:**
Slow retention-based leaks (circular references, unbounded caches) don't show up in short profiling sessions

**Fix:**
```python
# Run long-term monitoring
results = monitor_memory_leak(iterations=100)

if results['leak_detected']:
    # Investigate with objgraph
    import objgraph
    objgraph.show_growth()  # Shows growing object types
```

**Prevention:**
- Monitor memory over hours, not minutes
- Track `memory.growth_mb` metric in Datadog
- Alert on sustained growth (>10MB/hour)

---

### Failure 4: Query Optimization Complexity

**Symptom:**
```
APM shows query is slow but EXPLAIN ANALYZE is confusing
Sequential Scan detected but adding index didn't help
```

**Root Cause:**
Complex execution plans require understanding query planner decisions

**Fix:**
```python
# Use APM query profiling with EXPLAIN ANALYZE
analysis = profiler.analyze_query_performance(query)
print(analysis['issues'])
# ["Sequential scan detected - consider adding index"]
```

**Prevention:**
- Start with simple queries, optimize incrementally
- Use APM query analysis as starting point, not answer
- Test indexes in staging with production data volumes

---

### Failure 5: APM Cost Explosion ($500+ bill)

**Symptom:**
```
Expected: $51/month
Actual: $523/month
Cause: 100M spans/month at $5 per 1M spans
```

**Root Cause:**
High traffic + default sampling (100%) = millions of spans

**Fix:**
```python
# Reduce sampling at high traffic
if requests_per_hour > 10000:
    DD_TRACE_SAMPLE_RATE = 0.01  # 1% for high traffic
else:
    DD_TRACE_SAMPLE_RATE = 0.1   # 10% for normal traffic
```

**Prevention:**
- Start with low sampling (10%)
- Monitor span count in Datadog billing dashboard
- Set budget alerts at $100, $200 thresholds

---

## Decision Card

### ✅ Benefit
Deep code-level profiling reveals bottlenecks down to specific function calls and line numbers. Reduces debugging time from hours to minutes by showing CPU hotspots, memory leaks, and slow queries with flame graphs.

### ❌ Limitation
Adds 2-5% CPU overhead in production even with conservative sampling (1% profiling, 10% trace sampling). Cost scales rapidly: $51/month minimum, rising to $300+/month at 100K requests/hour due to per-span analysis fees.

### 💰 Cost
- **Time**: 2-4 hours for initial setup, 1-2 days for production tuning
- **Monthly**: $51-100 for small deployments (1-3 hosts), $300-800 for medium scale (10-15 hosts, 10M spans/day)
- **Complexity**: 300+ lines of APM config code

### 🤔 Use When
- Traffic >1K requests/hour with known performance problems (P95 >3s)
- Budget allows $50-200/month for APM
- Team of 3+ engineers who will actively monitor dashboards
- No compliance restrictions on third-party telemetry

### 🚫 Avoid When
- Traffic <1K requests/hour (use py-spy for one-time profiling instead)
- Budget <$100/month total (use open-source Grafana Tempo)
- Processing sensitive data requiring full data sovereignty
- No known performance issues yet (premature optimization)

---

## Alternative Solutions

### 1. Open-Source: Grafana Tempo + Grafana
- **Cost**: $0 (self-hosted) or $50/month (Grafana Cloud)
- **Pros**: Full control, no vendor lock-in
- **Cons**: Manual setup, less powerful profiling

### 2. Cloud Providers: AWS X-Ray, GCP Cloud Profiler
- **Cost**: Pay-per-use (usually cheaper at low scale)
- **Pros**: Native cloud integration
- **Cons**: Vendor lock-in, limited cross-cloud

### 3. Manual Profiling: py-spy
- **Cost**: $0
- **Pros**: Zero overhead when not profiling
- **Cons**: Manual, one-time snapshots only

### Decision Framework
```
Choose APM when you have ALL of:
✓ High traffic (>1K requests/hour)
✓ Known performance problems
✓ Adequate budget ($50-200/month)
✓ Data privacy clearance
```

---

## Troubleshooting

### APM Not Showing Data in Datadog UI

```bash
# Check configuration
python -c "from config import apm_config; print(f'Configured: {apm_config.is_configured}')"

# Check APM initialization
python -c "from l2_m7_application_performance_monitoring import apm_manager; print(f'Initialized: {apm_manager.initialize()}')"

# Verify API key
curl -H "DD-API-KEY: ${DD_API_KEY}" https://api.datadoghq.com/api/v1/validate

# Check spans are being sent
export DD_TRACE_DEBUG=true
python app.py
# Look for "Sending spans to Datadog" in logs
```

### High APM Overhead

```bash
# Measure overhead
python -c "
import psutil
import time
for i in range(10):
    print(f'CPU: {psutil.cpu_percent(interval=1):.1f}%')
"
# Should be <75% under normal load

# Reduce profiling if overhead high
export DD_PROFILING_CAPTURE_PCT=0.5  # Reduce from 1% to 0.5%
export DD_TRACE_SAMPLE_RATE=0.05      # Reduce from 10% to 5%
```

### Memory Leak Not Detected

```bash
# Run longer monitoring
python -c "
from l2_m7_application_performance_monitoring import monitor_memory_leak
results = monitor_memory_leak(iterations=100)  # Increase from 10 to 100
print(results)
"

# Use objgraph for detailed analysis
pip install objgraph
python -c "
import objgraph
objgraph.show_growth(limit=10)  # Shows top 10 growing objects
"
```

---

## Project Structure

```
.
├── README.md                                    # This file
├── requirements.txt                             # Dependencies
├── .env.example                                 # Environment template
├── config.py                                    # Configuration management
├── app.py                                       # FastAPI wrapper
├── src/
│   └── l3_m7_application_performance_monitoring/
│       └── __init__.py                          # Core module
├── tests/
│   └── test_m7_application_performance_monitoring.py  # Smoke tests
├── notebooks/
│   └── L3_M7_Application_Performance_Monitoring.ipynb  # Jupyter notebook
├── configs/                                     # Configuration files
├── scripts/                                     # Utility scripts
├── example_data.json                            # Sample data (JSON)
└── example_data.txt                             # Sample data (text)
```

---

## Testing

```bash
# Run all tests
pytest tests/test_m7_application_performance_monitoring.py -v

# Run specific test
pytest tests/test_m7_application_performance_monitoring.py::TestProfiledRAGPipeline::test_process_query_basic -v

# Run with coverage
pytest tests/ --cov=src.l3_m7_application_performance_monitoring --cov-report=html
```

---

## Production Checklist

- [ ] API keys configured in `.env`
- [ ] Sampling rates production-safe (≤10% traces, ≤1% profiling)
- [ ] Safety limits set (`DD_PROFILING_MAX_TIME_USAGE_PCT=5`)
- [ ] Load tested with APM enabled
- [ ] APM overhead measured (<5% CPU increase)
- [ ] Cost monitoring dashboard created
- [ ] Budget alerts configured ($100, $200 thresholds)
- [ ] Rollback plan documented
- [ ] Team trained on Datadog UI

---

## Next Steps

After mastering APM:
1. **Module 7.3**: Error Tracking & Root Cause Analysis
2. **Module 8.1**: Advanced Cost Optimization
3. **Module 8.2**: Multi-Region Deployment

---

## Resources

- [Datadog APM Documentation](https://docs.datadoghq.com/tracing/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Module 7.1: Distributed Tracing](../M7.1/)
- [Module 2.3: Production Monitoring](../../Level1/M2.3/)

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yesvisare/ccc_l2_aug_practical/issues)
- **Discord**: #practathon-m7-2
- **Docs**: [Module 7.2 Full Guide](./M7_2_Application_Perf.md)

---

**Built with ❤️ for Level 2 learners mastering production observability**
