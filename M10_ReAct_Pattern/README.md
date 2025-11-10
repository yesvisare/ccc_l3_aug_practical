# Module 10.1: ReAct Pattern Implementation

> **Agentic RAG with Thought → Action → Observation Reasoning Loop**

A complete, production-ready implementation of the ReAct (Reasoning and Acting) pattern for building AI agents that can autonomously select and execute tools to answer complex multi-step queries.

**Based on:** `augmented_M10_VideoM10_1_ReAct_Pat.md`
**Level:** 3 (Requires Level 1 M1.4 and Level 2 completion)
**Duration:** 42 minutes learning + 40-60 hours implementation

---

## 📋 Overview

### What is ReAct?

The **ReAct pattern** gives RAG systems the ability to:
- **Think** (Thought) - Reason about what information is needed
- **Act** (Action) - Execute tools to gather that information
- **Observe** (Observation) - Learn from tool results and decide next steps
- **Repeat** - Continue the cycle until the query is fully answered

### Real-World Analogy

Like a detective solving a case:
1. **Thought:** "I need to check the suspect's alibi"
2. **Action:** Interview witnesses
3. **Observation:** "The alibi checks out, but there's a timeline gap"
4. **Thought:** "I should examine phone records"
5. **Action:** Request phone records
6. **Observation:** "Multiple calls to unknown number"
...and so on until conclusion.

### When to Use This

✅ **Use ReAct agents for:**
- Complex queries requiring 2-5 different tools
- Multi-step reasoning (gather → calculate → synthesize)
- High-value queries where 3-10s latency is acceptable
- Scenarios where margin >$0.10/query supports $0.02 agent cost

❌ **DON'T use for:**
- Simple retrieval (90% of queries) - use static pipeline
- Real-time applications (<1s latency required)
- High-volume low-margin scenarios
- When tools are unreliable (>10% failure rate)

---

## 🚀 Quickstart

### 1. Installation

```bash
cd M10_ReAct_Pattern
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required: OPENAI_API_KEY
# Optional: PINECONE_API_KEY (for Level 1 RAG integration)
```

### 3. Test the Implementation

**Option A: Run the module directly**
```bash
# Linux/Mac
PYTHONPATH=. python -m src.l3_m10_react_pattern_implementation

# Windows PowerShell
$env:PYTHONPATH="."; python -m src.l3_m10_react_pattern_implementation
```

**Option B: Run the FastAPI server**
```bash
# Linux/Mac
python app.py

# Windows PowerShell (using provided script)
powershell -File scripts/run_api.ps1
# Or manually:
# $env:PYTHONPATH="."; uvicorn app:app --reload
```

**Option C: Run the Jupyter notebook**
```bash
jupyter notebook notebooks/L3_M10_ReAct_Pattern_Implementation.ipynb
```

### 4. Test the API

```bash
# Simple query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our refund policy?", "use_agent": true}'

# Complex multi-step query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare our Q3 revenue of $125,000 to the SaaS industry growth rate and calculate the percentage difference",
    "use_agent": true
  }'
```

### 5. Run Tests

```bash
# Linux/Mac
PYTHONPATH=. pytest -q tests/

# Windows PowerShell (using provided script)
powershell -File scripts/run_tests.ps1
# Or manually:
# $env:PYTHONPATH="."; pytest -q tests/
```

---

## 🌍 Environment Variables

The system reads configuration from `.env` file (copy from `.env.example`):

### Required
- `OPENAI_API_KEY` - OpenAI API key for LLM access (required for agent functionality)

### Optional
- `AGENT_MODEL` - OpenAI model name (default: `gpt-4`)
- `AGENT_TEMPERATURE` - Temperature for reasoning (default: `0.0` for deterministic)
- `AGENT_MAX_ITERATIONS` - Maximum reasoning steps (default: `8`)
- `AGENT_TIMEOUT_SECONDS` - Execution timeout (default: `60`)
- `PINECONE_API_KEY` - Pinecone API key for vector search (Level 1 integration)
- `PINECONE_ENVIRONMENT` - Pinecone environment (default: `us-west1-gcp`)
- `PINECONE_INDEX_NAME` - Pinecone index name (default: `level1-rag`)
- `INDUSTRY_API_KEY` - External industry data API key
- `INDUSTRY_API_URL` - Industry data API URL
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `ENABLE_AGENT` - Enable/disable agent (default: `true`)
- `FALLBACK_TO_STATIC` - Fallback to static pipeline on errors (default: `true`)

See `configs/example.json` for detailed documentation of all configuration keys.

---

## 🔒 Offline Mode

The implementation supports **OFFLINE mode** for demonstration and exploration without API keys:

```bash
# Set OFFLINE environment variable
export OFFLINE=true  # Linux/Mac
$env:OFFLINE="true"  # Windows PowerShell

# Run notebook or scripts - external API calls will be skipped
jupyter notebook notebooks/L3_M10_ReAct_Pattern_Implementation.ipynb
```

**What happens in OFFLINE mode:**
- ✅ Notebook structure and explanations fully accessible
- ✅ Tool registry and configuration examples work
- ✅ Tests pass or skip gracefully
- ⚠️ Agent execution skipped (requires OPENAI_API_KEY)
- ⚠️ External API calls return mock data

**Use cases:**
- Exploring the codebase without API costs
- Learning the ReAct pattern structure
- Running smoke tests in CI/CD without keys
- Teaching/demo scenarios

---

## 🏗️ Architecture

### System Diagram

```
┌─────────────────────────────────────────────────┐
│                  User Query                     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Query Classifier                   │
│  (Simple vs Complex)                            │
└────────┬────────────────────────────────┬───────┘
         │                                │
    Simple│                          Complex
         │                                │
         ▼                                ▼
┌──────────────────┐        ┌──────────────────────────┐
│ Static Pipeline  │        │     ReAct Agent          │
│ (Level 1 RAG)    │        │  Thought → Action Loop   │
│ 300ms, $0.002    │        │  3-10s, $0.01-0.03       │
└──────────────────┘        └────────┬─────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌────────────┐   ┌─────────────┐
              │RAG_Search│   │ Calculator │   │Industry_Data│
              └──────────┘   └────────────┘   └─────────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                            ┌─────────────────┐
                            │  Final Answer   │
                            └─────────────────┘
```

### Components

1. **Tool Registry** (`get_tools()`)
   - RAG_Search: Semantic search over company documents
   - Calculator: Safe mathematical expression evaluation
   - Industry_Data: External benchmark API integration

2. **ReAct Prompt** (`get_react_prompt()`)
   - Structured prompt for Thought-Action-Observation format
   - Safety limits (max 8 iterations, explicit stopping criteria)
   - Efficiency guidance (stop when you have enough info)

3. **Agent Executor** (`ReActAgent`, `StatefulReActAgent`)
   - Orchestrates the reasoning loop
   - Handles tool selection and execution
   - Error recovery with fallback to static pipeline

4. **State Management** (`AgentState`)
   - Tracks reasoning steps for debugging
   - Session-based conversation memory
   - Persistent trace logging

5. **FastAPI Wrapper** (`app.py`)
   - REST API endpoints
   - Health checks and metrics
   - Graceful handling of missing API keys

---

## 📖 How It Works

### The ReAct Loop

```python
# 1. User asks complex question
query = "Compare our Q3 revenue to industry benchmarks"

# 2. Agent thinks about what to do
Thought: "I need to find our Q3 revenue first"

# 3. Agent selects and executes tool
Action: RAG_Search
Action Input: "Q3 revenue"

# 4. Tool returns observation
Observation: "Q3 revenue was $125,000"

# 5. Agent reasons about next step
Thought: "Now I need industry benchmark data"

# 6. Agent executes next tool
Action: Industry_Data
Action Input: "SaaS,growth_rate"

# 7. Tool returns observation
Observation: "Industry benchmark: 25-35% YoY"

# 8. Agent has enough info to answer
Thought: "I now have enough information to answer"
Final Answer: "Your Q3 revenue of $125,000 represents strong performance..."
```

### Key Design Decisions (from Script)

**1. Tools return plain text, not structured data**
- ❌ Bad: `{"result": 143750}`
- ✅ Good: `"Calculation: 125000 * 1.15 = 143,750.00"`
- Why: LLM can read plain text reliably; structured data causes parsing failures

**2. Max 8 iterations with explicit stopping criteria**
- Prevents infinite loops
- Balances thoroughness with latency
- Each step ~1-2s = 8-16s max

**3. Temperature = 0.0 for deterministic reasoning**
- Production reliability over creativity
- Consistent tool selection
- Reproducible debugging

**4. Fallback to static pipeline on failure**
- Graceful degradation
- System remains available
- Error traces preserved for debugging

---

## 🐛 Common Failures & Fixes

### Failure #1: Infinite Reasoning Loop

**Symptom:** Agent repeats same action 3+ times
**Cause:** Tool returns unhelpful observation (e.g., "No documents found")
**Fix:** Loop detection + better tool error messages

**Example:**
```python
# Agent gets stuck:
Step 1 - Action: RAG_Search("California cities")
Step 1 - Observation: [No relevant documents found]
Step 2 - Action: RAG_Search("California cities")  # Same action!
Step 2 - Observation: [No relevant documents found]
...
```

**Solution:** Implemented in `StatefulReActAgent` with action signature tracking

### Failure #2: Wrong Tool Selection

**Symptom:** Uses RAG_Search when should use Calculator
**Cause:** Unclear tool descriptions or weak LLM reasoning
**Fix:** Pre-classify query type + use GPT-4 instead of GPT-3.5

**Accuracy:**
- GPT-4: 85-90% correct tool selection
- GPT-3.5-turbo: 60-70% correct tool selection

### Failure #3: State Corruption Across Turns

**Symptom:** Agent forgets previous conversation context
**Cause:** No conversation memory between turns
**Fix:** Session-based history in `StatefulReActAgent`

### Failure #4: Observation Parsing Failures

**Symptom:** OutputParserException from LangChain
**Cause:** Tool returned dict/JSON instead of plain text
**Fix:** Standardize all tool outputs to strings

### Failure #5: Missing Stop Condition

**Symptom:** Agent keeps searching after finding answer
**Cause:** Prompt doesn't emphasize efficiency
**Fix:** Add explicit stopping criteria to prompt

---

## 💰 Cost Analysis (from Script)

### Per-Query Costs

| Approach | Cost | Latency | When to Use |
|----------|------|---------|-------------|
| **Static Pipeline** | $0.002 | 300ms | Simple retrieval (90% traffic) |
| **ReAct Agent** | $0.01-0.03 | 3-10s | Complex multi-step (<10% traffic) |

### Monthly Costs at Scale

| Scale | Compute | LLM Calls | Tool Costs | **Total** |
|-------|---------|-----------|------------|-----------|
| **Small** (100 req/hr) | $50 | $1,400 | $40 | **$1,490** |
| **Medium** (1K req/hr) | $200 | $14,000 | $400 | **$14,600** |
| **Large** (10K+ req/hr) | $800 | $140,000 | $4,000 | **$144,800** |

### Cost Optimization Tips

1. **Cache tool results aggressively** (30-50% savings)
2. **Route simple queries to static pipeline** (40-60% savings)
3. **Use GPT-3.5-turbo for simple agents** (20-30% savings, but lower accuracy)

---

## 📊 Decision Card

**Quick reference for choosing between agents and static pipelines**

### ✅ BENEFIT
Enables multi-step reasoning queries requiring 2-5 tools. Autonomous tool selection without manual orchestration. Handles queries like "Compare metrics to industry, calculate differences, suggest strategies."

### ❌ LIMITATION
Adds 3-10s P95 latency vs 300ms static. Agent reasoning is probabilistic—10-15% tool selection errors even with GPT-4. Requires guard rails for infinite loops and state corruption.

### 💰 COST
- **Implementation:** 40-60 hours
- **Monthly at scale:** $1,500-15,000 (100-1K req/hr)
- **Complexity:** +500 LOC, LangChain dependency, state management

### 🤔 USE WHEN
- <10% queries are complex requiring 2-5 tools
- Volume <1,000 req/hr
- Can tolerate 3-10s latency
- Margin >$0.10/query
- Tools reliable >95%

### 🚫 AVOID WHEN
- 90%+ queries are simple retrieval
- Need <1s latency
- Margin <$0.05/query
- Tools unreliable
- Building first production system (use managed framework instead)

---

## 🧪 Testing

### Run Smoke Tests

```bash
python tests_smoke.py
```

**Tests include:**
- Configuration validation
- Tool registry structure
- Tool output format (must be strings)
- Agent state management
- API endpoint structure
- Graceful degradation without API keys

### Manual Testing Checklist

- [ ] Simple RAG query completes in 1-2 steps
- [ ] Calculator query uses Calculator tool first
- [ ] Industry data query uses Industry_Data tool
- [ ] Multi-step query uses 3+ tools in sequence
- [ ] Agent stops when it has sufficient information
- [ ] Fallback works when agent fails
- [ ] Conversation memory persists across turns
- [ ] API returns proper error messages

---

## 🛠️ Troubleshooting

### "OPENAI_API_KEY not configured"
- Copy `.env.example` to `.env`
- Add your OpenAI API key: `OPENAI_API_KEY=sk-...`

### "Agent execution failed"
- Check that your API key has credits
- Verify OpenAI service is available
- Review `agent_traces/` directory for debugging logs

### "Max iterations reached"
- Query is too complex for 8 steps
- Increase `AGENT_MAX_ITERATIONS` in `.env`
- Or simplify the query

### "Tool selection error" (wrong tool used)
- GPT-3.5 has lower accuracy; switch to GPT-4
- Update tool descriptions to be more explicit
- Consider pre-classifying query types

### "OutputParserException"
- Check that all tools return plain text strings
- Review tool outputs with `tests_smoke.py`
- Fix any tools returning dicts/JSON

### Agent is slow (>15s per query)
- Reduce `AGENT_MAX_ITERATIONS` to 5-6
- Add more explicit stopping criteria to prompt
- Monitor average steps per query

---

## 📚 File Structure

```
M10_ReAct_Pattern/
├── README.md                                    # This file
├── LICENSE                                      # Educational license
├── .gitignore                                   # Python defaults
├── pyproject.toml                               # pytest configuration
├── requirements.txt                             # Python dependencies
├── .env.example                                 # Environment template
├── config.py                                    # Configuration management
├── app.py                                       # FastAPI server (thin HTTP layer)
├── example_data.json                            # Test queries and scenarios
├── src/                                         # Source code
│   └── l3_m10_react_pattern_implementation/     # Main package
│       └── __init__.py                          # Core ReAct implementation
├── notebooks/                                   # Jupyter notebooks
│   └── L3_M10_ReAct_Pattern_Implementation.ipynb  # Interactive tutorial
├── tests/                                       # Test files
│   └── test_m10_react_pattern_implementation.py   # Smoke tests
├── configs/                                     # Configuration examples
│   └── example.json                             # Key documentation
├── scripts/                                     # Helper scripts (Windows-first)
│   ├── run_api.ps1                             # Start FastAPI server
│   └── run_tests.ps1                           # Run pytest
└── agent_traces/                                # Agent execution logs (created at runtime)
```

---

## 🎓 PractaThon Challenges

### 🟢 Easy (60 minutes)
**Goal:** Add a weather API tool to the agent

**Requirements:**
- Implement weather tool with OpenWeatherMap API
- Test tool selection accuracy (>90%)
- Verify response time <4s P95

### 🟡 Medium (90-120 minutes)
**Goal:** Implement multi-turn conversation memory

**Requirements:**
- Build ConversationalReActAgent class
- Test context preservation across 3+ turns
- Add session expiry (1 hour inactivity)

### 🔴 Hard (4-5 hours)
**Goal:** Production monitoring and deployment

**Requirements:**
- Implement all 5 failure prevention mechanisms
- Add Prometheus metrics and Grafana dashboard
- A/B testing: 10% agent, 90% static pipeline
- Load test: 100 concurrent queries, P95 <10s, <10% failures

---

## 🔗 Next Steps

1. **Complete the PractaThon challenge** (start with Easy)
2. **Review the full script** (`augmented_M10_VideoM10_1_ReAct_Pat.md`)
3. **Study the 5 failure modes** in detail (Section 8 of script)
4. **Next module:** M10.2 - Building Custom Agent Tools & Integrations
5. **Join Discord** #practathon channel for questions

---

## 📄 License

Educational material from CCC Level 3 - Module 10: Agentic RAG & Tool Use

---

## 🙏 Acknowledgments

Based on the TVH Framework v2.0 (Transparent, Valuable, Honest) teaching approach.

**Key Principles:**
- ✅ Transparent about limitations (3-10s latency, 10-15% errors)
- ✅ Honest about when NOT to use (90% of queries don't need agents)
- ✅ Valuable alternatives provided (4 alternative architectures)
- ✅ Real failure modes documented with fixes

**Script Compliance:** 9,255 words, 42-minute duration, production-ready code with error handling
