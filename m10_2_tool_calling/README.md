# Module 10.2: Tool Calling & Function Execution

Production-grade tool ecosystem for agentic RAG systems with sandboxing, timeouts, and comprehensive error handling.

## Overview

This module teaches you how to build safe, production-ready tool calling infrastructure for AI agents. Transform your M10.1 ReAct agent from a one-trick pony (search only) into a Swiss Army knife that can:

- 🔍 Search vector databases
- 🧮 Calculate mathematical expressions safely
- 🗄️ Query databases with parameterized SQL
- 🌐 Call external APIs with rate limiting
- 💬 Send Slack notifications

**Critical Reality Check:** Tool calling adds 50-500ms latency per execution. Only use when agents must **DO** things, not just retrieve information.

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (optional for basic testing)
```

### 3. Run Examples

**Option A: Python Module**
```bash
python l2_m10_tool_calling_function_execution.py
```

**Option B: FastAPI Server**
```bash
python app.py
# Visit http://localhost:8000/docs for interactive API
```

**Option C: Jupyter Notebook**
```bash
jupyter notebook L2_M10_Tool_Calling_Function_Execution.ipynb
```

### 4. Run Tests

```bash
pytest tests_smoke.py -v
```

---

## Architecture

### System Diagram

```
┌─────────────────┐
│   LLM Agent     │  "I need to calculate 10000 * 0.002"
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        Tool Selection Layer              │
│  Parses: {"tool": "calculator",          │
│           "args": {"expression": ...}}   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       SafeToolExecutor                   │
│  1. Validate arguments against schema    │
│  2. Execute with timeout (ThreadPool)    │
│  3. Retry on transient failures          │
│  4. Validate JSON-serializable result    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          Tool Registry                   │
│  - knowledge_search (vector DB)          │
│  - calculator (SafeEval)                 │
│  - database_query (PostgreSQL)           │
│  - api_call (HTTP with rate limiting)    │
│  - slack_notification (webhooks)         │
└──────────────────────────────────────────┘
```

### 5-Step Execution Flow

1. **Tool Definition:** Register tools with Pydantic schemas
2. **Tool Selection:** LLM generates structured JSON
3. **Sandboxed Execution:** ThreadPoolExecutor with timeouts
4. **Result Validation:** Ensure JSON-serializable output
5. **Observation Integration:** Feed result back to ReAct loop

---

## How It Works

### Tool Registry

Central catalog of all available tools with Pydantic validation:

```python
from l2_m10_tool_calling_function_execution import (
    tool_registry,
    register_default_tools
)

# Register 5 production tools
register_default_tools(tool_registry)

# List available tools
tools = tool_registry.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
```

### Safe Execution

Execute tools with automatic safety protections:

```python
from l2_m10_tool_calling_function_execution import SafeToolExecutor

executor = SafeToolExecutor(tool_registry)

# Execute calculator (validates, times out, retries)
result = executor.execute_tool("calculator", {
    "expression": "2 + 2 * 10"
})

print(f"Result: {result.result}")  # {'result': 22}
print(f"Time: {result.execution_time_ms:.2f}ms")
```

### ReAct Agent Integration

Full agent loop with tool calling:

```python
from l2_m10_tool_calling_function_execution import ReActAgent

agent = ReActAgent(executor)
response = agent.run("How do I implement tool calling?")

print(response["answer"])
print(f"Iterations: {response['iterations']}")
```

---

## Common Failures & Fixes

### Failure 1: Code Injection Attack ☠️

**Attack:**
```python
executor.execute_tool("calculator", {"expression": "import os; os.system('rm -rf /')"})
```

**Mitigation:**
- Calculator validates only allowed characters: `0-9 + - * / ( ) .`
- Blocks keywords: `import`, `eval`, `exec`, `__`

**Result:** ✅ BLOCKED - Error returned to agent

### Failure 2: SQL Injection Attack ☠️

**Attack:**
```python
executor.execute_tool("database_query", {
    "query": "SELECT * FROM users; DROP TABLE users; --"
})
```

**Mitigation:**
- Only `SELECT` queries allowed
- Parameterized statements prevent injection

**Result:** ✅ BLOCKED - Only SELECT queries permitted

### Failure 3: Tool Timeout ⏱️

**Scenario:** External API takes 60s to respond, hangs agent

**Mitigation:**
- Per-tool timeout enforcement (default 30s)
- ThreadPoolExecutor kills hung threads
- Returns error instead of blocking

**Result:** ✅ Timeout enforced, agent continues

### Failure 4: Invalid Arguments 🔍

**Scenario:** LLM generates malformed JSON or missing params

**Mitigation:**
- Schema validation before execution
- Required parameter checking
- Type validation

**Result:** ✅ Rejected before execution, no crash

### Failure 5: Non-Serializable Result 🚫

**Scenario:** Tool returns Python object instead of JSON

**Mitigation:**
- Post-execution JSON serialization check
- Reject invalid results
- Return structured error

**Result:** ✅ Tool error propagated to agent gracefully

---

## Decision Card

### ✅ Choose This Approach When:

- Agents must take **actions** beyond retrieval (calculate, query DB, call APIs)
- You **control tool implementations** (can ensure idempotency)
- Latency targets permit **50-500ms overhead** per tool call
- Tools are **retry-safe** (no duplicate side effects)

### ❌ Avoid When:

- **Information-only agents** - If you only need search/retrieval, don't add complexity
- **Sub-100ms latency requirements** - Sandboxing overhead is too high
- **Cascading failure dependencies** - When one tool failure breaks others
- **Non-idempotent tools** - Retry logic can cause duplicates

### Alternative Solutions

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Pre-Approved Outputs** | Zero execution risk | Inflexible | Predictable queries only |
| **Human-in-the-Loop** | Maximum safety | Slow, not scalable | High-stakes decisions |
| **Managed Platforms** (Zapier, n8n) | Managed infra | Vendor lock-in, costs | Quick prototypes |
| **Container Isolation** (Docker) | Strong isolation | High resource overhead | Multi-tenant systems |

---

## Production Considerations

### Cost Breakdown (10K conversations/hour)

```
Monthly Costs:
├── API Calls:      $2,000 - $5,000 (external services)
├── Compute:        $500 - $1,000   (self-hosted)
└── Storage:        $200 - $500     (execution logs)
────────────────────────────────────────────────
Total:              $2,700 - $6,500/month
```

### Monitoring Requirements

Track these metrics in production:

- **Success Rate:** Tool calls succeeded vs failed (target: >99%)
- **Latency Percentiles:** p50, p95, p99 execution times
- **Cost Per Call:** Track API usage by tool
- **Error Categories:** Code injection attempts, timeouts, validation failures

### Deployment Checklist

- [ ] Load test with realistic query patterns (10K+ requests/hour)
- [ ] Implement circuit breakers for failing tools (5 failures → disable 60s)
- [ ] Set up distributed tracing (OpenTelemetry/Jaeger)
- [ ] Create runbooks for common failures (see above)
- [ ] Establish SLA targets for agent latency (e.g., p95 < 2s)
- [ ] Configure alerts for tool failure rates >5%

---

## Troubleshooting

### "RestrictedPython installation failed"

**Solution:** Install C compiler
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install
```

### "Tool execution timed out"

**Cause:** External API or database query too slow

**Solution:**
1. Check tool timeout in registry (default 30s)
2. Increase for specific tool if justified
3. Implement caching for expensive operations
4. Consider async execution for long operations

### "Agent reaches max iterations without answer"

**Cause:** Tool errors prevent agent from progressing

**Solution:**
1. Check tool execution logs for errors
2. Review agent trace: `response["trace"]`
3. Verify tool results are JSON-serializable
4. Add fallback tools for critical paths

### "High latency (>1s per query)"

**Cause:** Tool execution overhead or LLM calls

**Solution:**
1. Profile with execution statistics: `tool_registry.get_stats()`
2. Reduce retry counts for fast-failing tools
3. Implement tool result caching
4. Consider pre-computing expensive operations

---

## File Structure

```
m10_2_tool_calling/
├── l2_m10_tool_calling_function_execution.py  # Core module (all logic)
├── config.py                                   # Environment config
├── app.py                                      # FastAPI entrypoint
├── tests_smoke.py                              # Smoke tests
├── L2_M10_Tool_Calling_Function_Execution.ipynb  # Tutorial notebook
├── requirements.txt                            # Python dependencies
├── .env.example                                # Environment template
├── example_data.json                           # Sample queries/failures
└── README.md                                   # This file
```

---

## Next Steps

### Continue Learning

➡️ **M10.3: Multi-Agent Orchestration** - Coordinate multiple specialized agents
➡️ **M10.4: Conversational RAG** - Add multi-turn memory to tool-calling agents

### Practathon Challenges

**Easy (90 minutes):**
- Add 2 custom tools (e.g., weather API, file reader)
- Implement tool result caching

**Medium (2-3 hours):**
- Build circuit breaker pattern for failing tools
- Add tool execution cost tracking
- Implement tool versioning (v1, v2 of same tool)

**Hard (5-6 hours):**
- Create performance dashboard with Grafana
- Build adaptive timeout system (learns optimal timeouts)
- Implement tool dependency graph (tool A requires tool B)

---

## References

- [M10.1: ReAct Pattern](../m10_1_react_pattern/) - Prerequisite module
- [RestrictedPython Docs](https://restrictedpython.readthedocs.io/) - Sandboxing library
- [Tenacity Docs](https://tenacity.readthedocs.io/) - Retry logic
- [FastAPI Docs](https://fastapi.tiangolo.com/) - API framework

---

## License

CCC Level 3 - Module 10.2

**Warning:** This implementation is for educational purposes. Production deployments require additional security hardening, especially for multi-tenant systems.

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review example_data.json for failure scenarios
3. Run smoke tests to verify setup: `pytest tests_smoke.py -v`
4. Consult module video (40 minutes) for detailed explanations
