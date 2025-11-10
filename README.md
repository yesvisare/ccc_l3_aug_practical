# Module 10.3: Multi-Agent Orchestration

**Level 3 - CCC Practical Implementation**

A production-ready three-agent system (Planner, Executor, Validator) for complex multi-step analytical queries using LangGraph.

## Overview

Single agents struggle with complex tasks requiring simultaneous research, strategy, and validation. This module implements role-based agent teams with structured message passing protocols.

**Reality check:** Multi-agent orchestration adds 2-5x latency (30-60s vs 8-12s) and 3x API costs (~$0.045 vs ~$0.015 per query). Only use when quality improvement >20% confirmed via A/B testing.

### Key Features

- ✅ Three specialized agents with explicit role constraints
- ✅ TypedDict state management for type safety
- ✅ Conditional routing with LangGraph StateGraph
- ✅ Adaptive routing (simple queries bypass multi-agent)
- ✅ Production monitoring and cost tracking
- ✅ FastAPI wrapper with graceful degradation

### Architecture

```
┌─────────┐
│  Query  │
└────┬────┘
     │
     ▼
┌─────────┐     ┌──────────┐     ┌───────────┐
│ Planner ├────→│ Executor ├────→│ Validator │
└─────────┘     └──────────┘     └─────┬─────┘
     ▲                                  │
     │        Rejected (feedback)       │
     └──────────────────────────────────┘
                                        │
                   Approved             │
                                        ▼
                                   ┌────────┐
                                   │  END   │
                                   └────────┘
```

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Linux/Mac:**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - Model to use (default: gpt-4o-mini)
- `OPENAI_TEMPERATURE` - Temperature setting (default: 0.7)
- `MAX_ITERATIONS` - Max validation loops (default: 3)
- `OFFLINE` - Skip external API calls (default: false)
- `LOG_LEVEL` - Logging verbosity (default: INFO)

### 3. Run Examples

**Windows (PowerShell):**
```powershell
# Run API Server
.\scripts\run_api.ps1

# Or manually:
$env:PYTHONPATH=$PWD; uvicorn app:app --reload
```

**Linux/Mac:**
```bash
# Run API Server
./scripts/run_api.sh

# Or manually:
export PYTHONPATH=$PWD && uvicorn app:app --reload
```

#### Jupyter Notebook
```bash
jupyter notebook notebooks/L3_M10_Multi-Agent_Orchestration.ipynb
```

### 4. Test Installation

**Windows (PowerShell):**
```powershell
# Run tests
.\scripts\run_tests.ps1

# Or manually:
$env:PYTHONPATH=$PWD; python -m pytest tests/ -q
```

**Linux/Mac:**
```bash
# Run tests
./scripts/run_tests.sh

# Or manually:
export PYTHONPATH=$PWD && python3 -m pytest tests/ -q
```

### 5. Offline Mode

Run without external API calls (for testing/development):

**Windows:**
```powershell
$env:OFFLINE="true"; python app.py
```

**Linux/Mac:**
```bash
OFFLINE=true python app.py
```

When in offline mode, the API returns `{"skipped": true}` responses instead of making LLM calls.

## How It Works

### Three Agent Roles

**1. Planner Agent**
- **Role:** Strategy and task decomposition only
- **Input:** Complex user query
- **Output:** Structured plan with 3-5 sub-tasks
- **Constraint:** Does NOT execute tasks

**2. Executor Agent**
- **Role:** Task completion only
- **Input:** Individual sub-tasks from plan
- **Output:** Completed results for each task
- **Constraint:** Does NOT plan or validate

**3. Validator Agent**
- **Role:** Quality control and completeness checking
- **Input:** Executor results
- **Output:** Approval/rejection with actionable feedback
- **Constraint:** Does NOT execute or plan

### Message Passing Protocol

```python
from l2_m10_multi_agent_orchestration import run_multi_agent_query, should_use_multi_agent

# 1. Check if multi-agent is appropriate
routing = should_use_multi_agent("Analyze top 3 competitors and create strategy report")
print(f"Recommendation: {routing['recommendation']}")  # "multi-agent"

# 2. Execute query (if appropriate)
if routing['recommendation'] == 'multi-agent':
    result = run_multi_agent_query("Analyze top 3 competitors and create strategy report")

    print(f"Status: {result['validation_status']}")  # "approved"
    print(f"Time: {result['metadata']['total_time_seconds']}s")  # 30-60s
    print(f"Cost: ${result['metadata']['estimated_cost_usd']}")  # ~$0.045
```

### Workflow Diagram

```
User Query → [Complexity Check]
                │
                ├─ Simple (90%) → Single-Agent (3-5s, $0.015)
                │
                └─ Complex (10%) → Multi-Agent (30-60s, $0.045)
                                       │
                                       ├→ Planner (5s)
                                       ├→ Executor (15s)
                                       ├→ Validator (8s)
                                       │
                                       └→ [Approved ✓ | Rejected ✗ → Re-plan]
```

## Common Failures & Fixes

### 1. Communication Deadlock

**Symptom:** System hangs indefinitely

**Cause:** Missing conditional edges in routing logic

**Fix:**
```python
# Validate all routing values and explicit edge definition
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "validator": "validator",
        "end": END  # Explicit end path
    }
)
```

### 2. Role Confusion

**Symptom:** Executor starts planning instead of executing

**Cause:** Prompts lack explicit role constraints

**Fix:**
```python
prompt = f"""You are an EXECUTOR agent. Your ONLY role is to complete assigned tasks.

DO:
- Complete the specific task given to you
- Provide concrete, factual information

DON'T:
- Create new sub-tasks or plans
- Skip steps or assume information
- Validate your own work

Task: {task_desc}
"""
```

### 3. Coordination Overhead (12x Latency)

**Symptom:** 47 seconds vs 8 seconds for single-agent

**Cause:** Sequential execution creates bottlenecks

**Fixes:**
- ✅ Cache planner outputs for similar queries
- ✅ Parallelize independent executor tasks
- ✅ Adaptive routing (simple queries bypass multi-agent)

### 4. Validation Loops

**Symptom:** Endless iterations without improvement

**Cause:** Non-specific feedback like "not enough information"

**Fix:**
```python
# Require actionable feedback in validator prompt
prompt = f"""...
Provide SPECIFIC, ACTIONABLE feedback if rejecting:
- List exact gaps or missing information
- Suggest additional tasks needed

Respond in JSON:
{{
  "status": "approved" or "rejected",
  "feedback": "specific feedback",
  "missing_items": ["list", "of", "specific", "gaps"]
}}
"""
```

### 5. Multi-Agent Overkill

**Symptom:** Simple queries routed to multi-agent (3x cost, no benefit)

**Cause:** No complexity routing

**Fix:**
```python
# Use adaptive routing
routing = should_use_multi_agent(query)
if routing['recommendation'] == 'single-agent':
    # Use faster, cheaper single-agent approach
    result = single_agent_query(query)
else:
    # Use multi-agent only when justified
    result = run_multi_agent_query(query)
```

## Decision Card

### ✅ Use When

- Multi-step analytical queries where quality justifies 3x cost
- >30 second latency tolerance
- >$10K/month AI budget
- >20% A/B-tested quality improvement vs single-agent confirmed
- Independent validation required (regulatory, high-stakes)

### ❌ Avoid When

- **Simple queries** - Factual lookup doesn't benefit from coordination
- **Real-time (<5s)** - Minimum 9-15s latency impossible to avoid
- **Low budget (<$500/month)** - 3x costs consume budget quickly
- **Deterministic workflows** - Fixed steps better served by scripts
- **High-compliance** - Healthcare/finance need explicit audit trails

### Costs & Benefits

| Metric | Single-Agent | Multi-Agent | Trade-off |
|--------|-------------|-------------|-----------|
| **Latency** | 3-5s | 30-60s | 2-5x slower |
| **Cost/query** | $0.015 | $0.045 | 3x more expensive |
| **Code complexity** | ~50 lines | 400+ lines | 8x more code |
| **Quality (complex)** | Baseline | +15-30% | Modest improvement |
| **Debugging** | Easy | Hard | Much harder |
| **Monthly cost (1K req/hr)** | $10,800 | $32,600 | $21,800 more |

### ROI Calculation

**Break-even requires:**
- Quality improvement >20% measured via A/B testing
- High-value queries (>$0.15 value per query)
- Low volume (<5,000 complex queries/day)

**Example:** Customer success analytics
- Value per insight: $5
- Quality improvement: 25% (1 in 4 provides actionable insight)
- Additional value: $1.25 per query
- Multi-agent premium: $0.03
- **ROI: 42x** ✅ Justified

**Counter-example:** Product FAQ
- Value per answer: $0.02
- Quality improvement: 5% (negligible for simple answers)
- Additional value: $0.001 per query
- Multi-agent premium: $0.03
- **ROI: -30x** ❌ Not justified

## Troubleshooting

### API Key Not Configured

```bash
# Error: OPENAI_API_KEY not configured

# Fix:
1. Copy .env.example to .env
2. Add your OpenAI API key
3. Restart the application
```

### LangGraph Import Error

```bash
# Error: ModuleNotFoundError: No module named 'langgraph'

# Fix:
pip install langgraph==0.2.45
```

### High Costs

```bash
# Costs >$1,000/day unexpected

# Debug:
1. Check logs for query volume
2. Verify adaptive routing is enabled
3. Ensure simple queries bypass multi-agent
4. Set MAX_COST_PER_QUERY in .env

# Monitor:
grep "estimated_cost_usd" logs/*.log | awk '{sum+=$NF} END {print sum}'
```

### Slow Performance (>2 minutes)

```bash
# Query taking >120 seconds

# Debug:
1. Check MAX_ITERATIONS (should be ≤3)
2. Review validator feedback quality (specific vs generic)
3. Enable caching for planner outputs
4. Consider parallel executor for independent tasks

# Optimize:
- Reduce model temperature (0.3 vs 0.7)
- Use gpt-4o-mini instead of gpt-4
- Implement planner output caching
```

### Validation Rejection Loops

```bash
# Validator rejects indefinitely

# Debug:
1. Check validation_feedback in logs
2. Ensure feedback is actionable and specific
3. Verify MAX_ITERATIONS limit is set

# Fix validator prompt to require:
- Specific gap identification
- Suggested additional tasks
- Clear acceptance criteria
```

## API Reference

### Core Functions

#### `run_multi_agent_query(query: str) -> Dict[str, Any]`

Execute a query through the full multi-agent pipeline.

**Args:**
- `query` (str): User query to process

**Returns:**
```python
{
    'success': bool,
    'query': str,
    'plan': List[Dict],
    'results': List[str],
    'validation_status': str,  # "approved" | "rejected" | "pending"
    'validation_feedback': str,
    'metadata': {
        'total_time_seconds': float,
        'iterations': int,
        'estimated_cost_usd': float,
        'num_steps': int
    },
    'messages': List[str]
}
```

#### `should_use_multi_agent(query: str) -> Dict[str, Any]`

Determine if query should use multi-agent or single-agent approach.

**Args:**
- `query` (str): User query to evaluate

**Returns:**
```python
{
    'recommendation': str,  # "multi-agent" | "single-agent"
    'reason': str,
    'estimated_latency_seconds': str,
    'estimated_cost_usd': float,
    'warning': str  # Optional, for overkill scenarios
}
```

### FastAPI Endpoints

#### `GET /health`

Health check and configuration status.

#### `POST /query`

Execute multi-agent query with adaptive routing.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze emerging AI trends and identify top 3 opportunities",
    "force_multi_agent": false,
    "max_iterations": 3
  }'
```

#### `POST /route`

Check routing recommendation without executing.

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our return policy?"}'
```

## Project Structure

```
.
├── app.py                              # FastAPI wrapper
├── config.py                           # Configuration management
├── example_data.json                   # Sample queries
├── requirements.txt                    # Dependencies
├── .env.example                        # Environment template
├── .gitignore                          # Git ignore rules
├── pyproject.toml                      # pytest configuration
├── README.md                           # This file
│
├── src/
│   └── l3_m10_multi_agent_orchestration/
│       └── __init__.py                 # Core implementation
│
├── notebooks/
│   └── L3_M10_Multi-Agent_Orchestration.ipynb  # Interactive tutorial
│
├── tests/
│   ├── conftest.py                     # Test configuration
│   └── test_m10_multi_agent_orchestration.py   # Test suite
│
├── configs/
│   └── example.json                    # Configuration reference
│
└── scripts/
    ├── run_api.ps1                     # Windows API server
    ├── run_api.sh                      # Linux/Mac API server
    ├── run_tests.ps1                   # Windows tests
    └── run_tests.sh                    # Linux/Mac tests
```

## Production Deployment

### Scaling Thresholds

**100 requests/hour:** Serverless functions sufficient (AWS Lambda, Cloud Functions)

**1,000 requests/hour:** Add caching and parallel execution
```python
# Redis cache for planner outputs
from redis import Redis
cache = Redis()

# Check cache before planning
cached_plan = cache.get(f"plan:{query_hash}")
if cached_plan:
    state['plan'] = json.loads(cached_plan)
```

**10,000+ requests/hour:** Hybrid model
- 90% simple queries → cached/single-agent
- 10% complex queries → multi-agent

### Monitoring Essentials

Required metrics:
- P95 latency per agent (planner <5s, executor <15s, validator <8s)
- Validation rejection rate <20%
- Iterations per query <2 average
- Cost per query <$0.10
- Zero deadlock incidents

```python
# Prometheus metrics (optional)
from prometheus_client import Histogram, Counter

query_duration = Histogram('multiagent_query_duration_seconds', 'Query duration')
query_cost = Histogram('multiagent_query_cost_usd', 'Query cost in USD')
query_counter = Counter('multiagent_queries_total', 'Total queries', ['status'])
```

### Cost Optimization

At 10,000 requests/hour without optimization:
- LLM calls: $324,000/month ❌ Unsustainable

With adaptive routing (90% simple → single-agent):
- LLM calls: ~$50,000/month ✅ Viable

**Optimization strategies:**
1. Complexity-based routing (save 70% on simple queries)
2. Planner output caching (save 33% on similar queries)
3. Model selection (gpt-4o-mini vs gpt-4 saves 60%)
4. Batch similar queries (save 20% on throughput)

## Next Steps

### Immediate

1. ✅ Complete this module
2. Run A/B test vs single-agent baseline
3. Measure quality improvement (target >20%)
4. Calculate ROI using actual query value

### Advanced Topics

- **Module 11:** Advanced RAG patterns for production scale
- **Module 12:** Multi-modal agents (text + images + code)
- **Module 13:** Agent evaluation and quality metrics

## Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Multi-Agent Systems Paper](https://arxiv.org/abs/2308.08155)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## License

MIT License - See LICENSE file for details

## Contributing

Issues and PRs welcome! Please follow the existing code style and include tests.

---

**Remember:** Multi-agent orchestration is powerful but expensive. Start simple, measure quality gaps, and only introduce complexity when A/B testing proves it's worth the 3x cost and 2-5x latency trade-off.
