"""
Module 10.3: Multi-Agent Orchestration
Implementation of a three-agent system (Planner, Executor, Validator) using LangGraph.

Based on TVH Framework v2.0 requirements with role-based agent teams using structured
message passing protocols for complex multi-step analytical tasks.

Trade-offs (accepted):
- 2-5x slower performance (30-60s vs 8-12s single-agent)
- 3x API costs (~$0.045 vs ~$0.015 per query)
- 400+ lines of orchestration code
"""
import time
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict as ExtTypedDict
import operator
from config import get_langchain_llm, Config, logger


# ============================================================================
# STATE SCHEMA - TypedDict for type safety
# ============================================================================

class AgentState(TypedDict):
    """
    Centralized state schema for multi-agent orchestration.
    All agents read from and write to this shared state.
    """
    # Input
    query: str

    # Planner outputs
    plan: List[Dict[str, str]]  # List of sub-tasks with descriptions

    # Executor outputs
    results: Annotated[List[str], operator.add]  # Cumulative results from sub-tasks

    # Validator outputs
    validation_status: str  # "approved" | "rejected" | "pending"
    validation_feedback: str  # Specific actionable feedback

    # Orchestration metadata
    iterations: int
    current_step: int
    total_cost: float
    start_time: float
    messages: Annotated[List[str], operator.add]  # Conversation log


# ============================================================================
# AGENT NODES - Specialized roles
# ============================================================================

def planner_agent(state: AgentState) -> AgentState:
    """
    Planner Agent: Decomposes complex queries into structured sub-tasks.

    Role: Strategy and task decomposition only - does NOT execute tasks.

    Args:
        state: Current agent state containing query

    Returns:
        Updated state with plan (list of sub-tasks)

    Raises:
        Exception: If LLM call fails or plan parsing fails
    """
    logger.info("🎯 Planner Agent: Analyzing query and creating plan")

    try:
        llm = get_langchain_llm()
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    # Explicit role constraints to prevent role confusion
    prompt = f"""You are a PLANNER agent. Your ONLY role is to decompose queries into sub-tasks.

DO:
- Break down the query into 3-5 concrete, actionable sub-tasks
- Number each sub-task
- Make sub-tasks specific and measurable
- Consider dependencies between sub-tasks

DON'T:
- Execute any tasks yourself
- Provide answers or solutions
- Skip the planning step

Query: {state['query']}

Return ONLY a JSON array of sub-tasks in this format:
[
  {{"step": 1, "task": "specific task description"}},
  {{"step": 2, "task": "specific task description"}}
]
"""

    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start

    logger.info(f"Planner latency: {latency:.2f}s")

    # Parse plan from response
    try:
        # Extract JSON from response (handle markdown code blocks)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        plan = json.loads(content)

        if not isinstance(plan, list) or len(plan) == 0:
            raise ValueError("Plan must be a non-empty list")

    except Exception as e:
        logger.error(f"Failed to parse plan: {e}. Using fallback plan.")
        # Fallback: create single-step plan
        plan = [{"step": 1, "task": state['query']}]

    # Update state
    state['plan'] = plan
    state['messages'].append(f"PLANNER: Created {len(plan)}-step plan")
    state['total_cost'] += 0.015  # Estimated cost per call

    logger.info(f"✓ Plan created with {len(plan)} steps")

    return state


def executor_agent(state: AgentState) -> AgentState:
    """
    Executor Agent: Completes individual sub-tasks from the plan.

    Role: Task completion only - does NOT plan or validate.

    Args:
        state: Current agent state with plan

    Returns:
        Updated state with results from executed sub-tasks

    Raises:
        Exception: If LLM call fails
    """
    logger.info("⚙️ Executor Agent: Executing planned sub-tasks")

    try:
        llm = get_langchain_llm()
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    plan = state.get('plan', [])
    if not plan:
        logger.warning("No plan available for execution")
        state['messages'].append("EXECUTOR: No plan to execute")
        return state

    # Execute each sub-task
    executed_results = []
    total_latency = 0

    for task in plan:
        task_desc = task.get('task', '')
        step_num = task.get('step', 0)

        # Explicit role constraints
        prompt = f"""You are an EXECUTOR agent. Your ONLY role is to complete assigned tasks.

DO:
- Complete the specific task given to you
- Provide concrete, factual information
- Be thorough but concise (2-3 paragraphs max)

DON'T:
- Create new sub-tasks or plans
- Skip steps or assume information
- Validate your own work

Original Query: {state['query']}

Your assigned task (Step {step_num}): {task_desc}

Complete this task now:
"""

        start = time.time()
        response = llm.invoke(prompt)
        latency = time.time() - start
        total_latency += latency

        result = response.content.strip()
        executed_results.append(f"Step {step_num}: {result}")

        logger.info(f"✓ Completed step {step_num} ({latency:.2f}s)")

    # Update state
    state['results'] = executed_results
    state['current_step'] = len(plan)
    state['messages'].append(f"EXECUTOR: Completed {len(plan)} tasks ({total_latency:.2f}s)")
    state['total_cost'] += 0.015 * len(plan)  # Cost per task

    logger.info(f"✓ Execution complete: {len(executed_results)} tasks ({total_latency:.2f}s)")

    return state


def validator_agent(state: AgentState) -> AgentState:
    """
    Validator Agent: Quality control and completeness checking.

    Role: Validation only - does NOT execute or plan.

    Args:
        state: Current agent state with results

    Returns:
        Updated state with validation_status and validation_feedback

    Raises:
        Exception: If LLM call fails
    """
    logger.info("✅ Validator Agent: Checking quality and completeness")

    try:
        llm = get_langchain_llm()
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

    query = state.get('query', '')
    results = state.get('results', [])

    if not results:
        state['validation_status'] = 'rejected'
        state['validation_feedback'] = 'No results to validate'
        state['messages'].append("VALIDATOR: Rejected - no results")
        return state

    # Explicit role constraints with actionable feedback requirements
    prompt = f"""You are a VALIDATOR agent. Your ONLY role is quality control.

DO:
- Check if results fully answer the original query
- Verify completeness and accuracy
- Provide SPECIFIC, ACTIONABLE feedback if rejecting
- List exact gaps or missing information

DON'T:
- Execute tasks yourself
- Provide generic feedback like "not enough information"
- Approve incomplete work

Original Query: {query}

Results to validate:
{chr(10).join(results)}

Respond in JSON format:
{{
  "status": "approved" or "rejected",
  "feedback": "specific feedback with gaps listed if rejected, or confirmation if approved",
  "missing_items": ["list", "of", "specific", "gaps"] if rejected
}}
"""

    start = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start

    logger.info(f"Validator latency: {latency:.2f}s")

    # Parse validation response
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        validation = json.loads(content)
        status = validation.get('status', 'rejected')
        feedback = validation.get('feedback', 'No feedback provided')

    except Exception as e:
        logger.error(f"Failed to parse validation: {e}. Defaulting to rejected.")
        status = 'rejected'
        feedback = f"Validation parsing failed: {str(e)}"

    # Update state
    state['validation_status'] = status
    state['validation_feedback'] = feedback
    state['messages'].append(f"VALIDATOR: {status.upper()} - {feedback}")
    state['total_cost'] += 0.015  # Cost per validation

    logger.info(f"✓ Validation: {status} ({latency:.2f}s)")

    return state


# ============================================================================
# ROUTING LOGIC - Conditional edges for workflow control
# ============================================================================

def should_continue(state: AgentState) -> Literal["validator", "end"]:
    """
    Routing: After executor, always go to validator.
    Prevents communication deadlock by explicit routing.
    """
    return "validator"


def check_validation(state: AgentState) -> Literal["end", "planner"]:
    """
    Routing: After validator, either end (if approved) or replan (if rejected).
    Prevents validation loops through iteration limits.
    """
    status = state.get('validation_status', 'pending')
    iterations = state.get('iterations', 0)
    max_iterations = Config.MAX_ITERATIONS

    if status == 'approved':
        logger.info("✓ Validation approved - ending workflow")
        return "end"

    if iterations >= max_iterations:
        logger.warning(f"⚠️ Max iterations ({max_iterations}) reached - ending workflow")
        state['messages'].append(f"SYSTEM: Max iterations reached ({max_iterations})")
        return "end"

    logger.info(f"⟳ Validation rejected - replanning (iteration {iterations + 1}/{max_iterations})")
    state['iterations'] = iterations + 1
    return "planner"


# ============================================================================
# WORKFLOW COMPILATION - LangGraph StateGraph
# ============================================================================

def create_multi_agent_graph():
    """
    Create and compile the multi-agent orchestration graph.

    Returns:
        Compiled StateGraph ready for invocation

    Graph structure:
        START → planner → executor → validator → [approved: END | rejected: planner]
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.error("langgraph not installed. Run: pip install langgraph")
        raise

    # Initialize graph with state schema
    workflow = StateGraph(AgentState)

    # Add nodes (agents)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("validator", validator_agent)

    # Add edges (routing)
    workflow.set_entry_point("planner")  # Start with planner
    workflow.add_edge("planner", "executor")  # Planner → Executor
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "validator": "validator",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "validator",
        check_validation,
        {
            "end": END,
            "planner": "planner"  # Loop back if rejected
        }
    )

    # Compile graph
    app = workflow.compile()

    logger.info("✓ Multi-agent graph compiled successfully")

    return app


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================

def run_multi_agent_query(query: str) -> Dict[str, Any]:
    """
    Execute a query using the multi-agent orchestration system.

    Args:
        query: User query string

    Returns:
        Dict containing:
            - success: bool
            - query: original query
            - results: list of execution results
            - validation_status: approval status
            - metadata: timing, cost, iterations
            - messages: conversation log

    Raises:
        Exception: If graph creation or execution fails
    """
    logger.info(f"\n{'='*70}\n🚀 Starting multi-agent query: {query[:50]}...\n{'='*70}")

    start_time = time.time()

    # Initialize state
    initial_state: AgentState = {
        'query': query,
        'plan': [],
        'results': [],
        'validation_status': 'pending',
        'validation_feedback': '',
        'iterations': 0,
        'current_step': 0,
        'total_cost': 0.0,
        'start_time': start_time,
        'messages': []
    }

    try:
        # Create and run graph
        app = create_multi_agent_graph()
        final_state = app.invoke(initial_state)

        # Calculate total time
        total_time = time.time() - start_time

        # Prepare response
        response = {
            'success': True,
            'query': query,
            'plan': final_state.get('plan', []),
            'results': final_state.get('results', []),
            'validation_status': final_state.get('validation_status', 'unknown'),
            'validation_feedback': final_state.get('validation_feedback', ''),
            'metadata': {
                'total_time_seconds': round(total_time, 2),
                'iterations': final_state.get('iterations', 0),
                'estimated_cost_usd': round(final_state.get('total_cost', 0), 4),
                'num_steps': len(final_state.get('plan', [])),
                'num_results': len(final_state.get('results', []))
            },
            'messages': final_state.get('messages', [])
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Query completed successfully")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Estimated cost: ${final_state.get('total_cost', 0):.4f}")
        logger.info(f"   Validation: {final_state.get('validation_status', 'unknown')}")
        logger.info(f"{'='*70}\n")

        return response

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Multi-agent query failed: {e}", exc_info=True)

        return {
            'success': False,
            'query': query,
            'error': str(e),
            'metadata': {
                'total_time_seconds': round(total_time, 2),
                'iterations': 0,
                'estimated_cost_usd': 0.0
            },
            'messages': [f"ERROR: {str(e)}"]
        }


def should_use_multi_agent(query: str) -> Dict[str, Any]:
    """
    Adaptive routing: Determine if query should use multi-agent or single-agent.

    Heuristics based on query complexity:
    - Simple factual queries → single-agent (3x faster, 3x cheaper)
    - Complex analytical queries → multi-agent

    Args:
        query: User query string

    Returns:
        Dict with recommendation and reasoning
    """
    # Simple heuristics (in production, use classifier model)
    complexity_indicators = [
        'analyze', 'strategy', 'compare', 'evaluate', 'comprehensive',
        'research', 'assess', 'investigate', 'multiple', 'both'
    ]

    query_lower = query.lower()
    complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
    word_count = len(query.split())

    # Decision logic
    if complexity_score >= 2 or word_count > 20:
        return {
            'recommendation': 'multi-agent',
            'reason': f'Complex query (score: {complexity_score}, words: {word_count})',
            'estimated_latency_seconds': '30-60',
            'estimated_cost_usd': 0.045
        }
    else:
        return {
            'recommendation': 'single-agent',
            'reason': f'Simple query (score: {complexity_score}, words: {word_count})',
            'estimated_latency_seconds': '3-5',
            'estimated_cost_usd': 0.015,
            'warning': 'Multi-agent would be overkill (3x cost, 2-5x latency)'
        }


# ============================================================================
# CLI USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check configuration
    if not Config.is_configured():
        print("⚠️ OPENAI_API_KEY not configured. Set it in .env file.")
        print("Example: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print("\n" + "="*70)
    print("Multi-Agent Orchestration System - CLI Demo")
    print("="*70 + "\n")

    # Example queries
    example_queries = [
        "Analyze our top 3 competitors and create a strategy report",
        "What is our return policy?",  # Should NOT use multi-agent
    ]

    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {query}")
        print('='*70)

        # Check if multi-agent is appropriate
        routing = should_use_multi_agent(query)
        print(f"\n📊 Routing recommendation: {routing['recommendation']}")
        print(f"   Reason: {routing['reason']}")

        if routing['recommendation'] == 'single-agent':
            print(f"   ⚠️ Warning: {routing.get('warning', '')}")
            print("\n   Skipping multi-agent execution (use single-agent instead)")
            continue

        # Run multi-agent query
        result = run_multi_agent_query(query)

        # Display results (truncated)
        if result['success']:
            print(f"\n✅ Status: {result['validation_status']}")
            print(f"📝 Plan: {len(result['plan'])} steps")
            print(f"📊 Results: {len(result['results'])} items")
            print(f"⏱️  Time: {result['metadata']['total_time_seconds']}s")
            print(f"💰 Cost: ${result['metadata']['estimated_cost_usd']}")

            # Show first result only (keep output short)
            if result['results']:
                print(f"\n📄 First result preview:")
                print(f"   {result['results'][0][:150]}...")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

    print("\n" + "="*70)
    print("Demo complete")
    print("="*70 + "\n")
