"""
Module 10.1: ReAct Pattern Implementation
Implements Thought → Action → Observation reasoning loop for complex multi-step queries.

Based on: augmented_M10_VideoM10_1_ReAct_Pat.md
"""
import os
import ast
import operator
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def rag_search_tool(query: str) -> str:
    """
    Search internal documents using semantic search (wraps Level 1 pipeline).
    Use this when the user asks about company-specific information.

    Args:
        query: Search query string

    Returns:
        Relevant document chunks as formatted plain text string
    """
    try:
        # Mock implementation - in production, call actual Pinecone search
        logger.info(f"RAG Search: {query[:50]}...")

        # Simulated results
        mock_results = [
            {"text": f"Document about {query}: Our refund policy offers a 30-day money-back guarantee for all products."},
            {"text": f"Related info for {query}: Refunds are processed within 5-7 business days."},
            {"text": f"Additional context for {query}: Contact support@company.com to initiate refund."}
        ]

        # Format as plain text for agent consumption
        formatted = []
        for idx, result in enumerate(mock_results[:3]):
            formatted.append(f"[Doc {idx+1}] {result['text'][:200]}")

        return "\n\n".join(formatted)

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return f"Search error: {str(e)}. Unable to find relevant documents."


def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions safely.
    Use this when the user's query requires calculations.

    Args:
        expression: Math expression like "125000 * 1.15 - 98000"

    Returns:
        Calculated result as plain text string with interpretation
    """
    try:
        logger.info(f"Calculator: {expression}")

        # Safe operators only (no exec/eval of arbitrary code)
        operators_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def eval_expr(node):
            """Recursively evaluate AST nodes safely."""
            if isinstance(node, ast.Num):  # number
                return node.n
            elif isinstance(node, ast.BinOp):  # binary operation
                return operators_map[type(node.op)](
                    eval_expr(node.left),
                    eval_expr(node.right)
                )
            elif isinstance(node, ast.UnaryOp):  # unary operation
                return operators_map[type(node.op)](eval_expr(node.operand))
            else:
                raise ValueError(f"Unsupported operation: {type(node).__name__}")

        # Parse and evaluate
        result = eval_expr(ast.parse(expression, mode='eval').body)

        # Return plain text with interpretation
        return f"Calculation: {expression} = {result:,.2f}"

    except SyntaxError:
        return f"Calculation error: Invalid expression format '{expression}'. Please use basic arithmetic operators (+, -, *, /, **)."
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Calculation error: {str(e)}. Please check the expression format."


def industry_data_tool(query: str) -> str:
    """
    Fetch industry benchmark data from external API (or mock data).
    Use this when comparing company metrics to industry standards.

    Args:
        query: Query in format "industry,metric" (e.g., "SaaS,growth_rate")

    Returns:
        Industry benchmark data as formatted plain text string
    """
    try:
        logger.info(f"Industry Data: {query}")

        # Parse query
        parts = query.split(",")
        if len(parts) != 2:
            return "Industry data error: Please use format 'industry,metric' (e.g., 'SaaS,growth_rate')"

        industry = parts[0].strip()
        metric = parts[1].strip()

        # Mock data - in production, call actual API
        mock_data = {
            "SaaS": {
                "growth_rate": "25-35% YoY",
                "churn_rate": "5-7% monthly",
                "ltv_cac_ratio": "3:1 to 5:1",
                "revenue": "$50M-100M median ARR"
            },
            "E-commerce": {
                "growth_rate": "15-25% YoY",
                "conversion_rate": "2-3%",
                "cart_abandonment": "65-70%",
                "revenue": "$10M-30M median annual"
            },
            "Fintech": {
                "growth_rate": "40-60% YoY",
                "customer_acquisition_cost": "$50-150",
                "revenue": "$20M-80M median"
            }
        }

        industry_data = mock_data.get(industry, {})
        metric_value = industry_data.get(metric, "Data not available for this metric")

        if metric_value == "Data not available for this metric":
            available = ", ".join(industry_data.keys()) if industry_data else "none"
            return f"Industry benchmark for {industry} - {metric}: Not available. Available metrics: {available}"

        return f"Industry benchmark for {industry} - {metric}: {metric_value}"

    except Exception as e:
        logger.error(f"Industry data error: {e}")
        return f"API error: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

def get_tools() -> List[Tool]:
    """
    Create and return the tool registry for the ReAct agent.

    Returns:
        List of LangChain Tool objects
    """
    return [
        Tool(
            name="RAG_Search",
            func=rag_search_tool,
            description=(
                "Search internal company documents using semantic search. "
                "Use when user asks about company-specific policies, procedures, "
                "or historical data. Input should be a search query string."
            )
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description=(
                "Perform mathematical calculations. Use ONLY when the query "
                "contains explicit mathematical expressions or asks for arithmetic. "
                "Input should be a mathematical expression like '125000 * 1.15'."
            )
        ),
        Tool(
            name="Industry_Data",
            func=industry_data_tool,
            description=(
                "Fetch industry benchmark data for comparisons. Use when comparing "
                "company metrics to industry standards. Input format: 'industry,metric' "
                "(e.g., 'SaaS,growth_rate')."
            )
        ),
    ]


# ============================================================================
# REACT PROMPT
# ============================================================================

REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to tools. Answer the user's question by reasoning step-by-step.

You MUST follow this exact format:

Thought: [Your reasoning about what to do next]
Action: [The tool to use, must be one of: {tool_names}]
Action Input: [The input to pass to the tool]
Observation: [Result from the tool - this will be provided]

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now have enough information to answer
Final Answer: [Your final answer to the user]

IMPORTANT RULES:
1. Always start with a Thought before taking an Action
2. Use EXACTLY ONE tool per Action (no multiple tools)
3. After each Observation, think about whether you need more information
4. When you have enough information, provide a Final Answer
5. Maximum {max_iterations} reasoning steps - if you can't solve it by then, explain what's missing
6. EFFICIENCY: Stop as soon as you have sufficient information - don't keep searching "just to be thorough"

STOPPING CRITERIA:
- If RAG_Search returns a clear, complete answer → provide Final Answer immediately
- If you've found relevant information from 2+ searches → synthesize and answer
- If you've tried 3+ tool calls without progress → provide best answer with available info

Available Tools:
{tools}

Question: {input}

Agent Trajectory:
{agent_scratchpad}"""


def get_react_prompt(max_iterations: int = 8) -> PromptTemplate:
    """
    Create the ReAct prompt template.

    Args:
        max_iterations: Maximum reasoning steps allowed

    Returns:
        PromptTemplate configured for ReAct pattern
    """
    return PromptTemplate(
        template=REACT_PROMPT_TEMPLATE,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"max_iterations": str(max_iterations)}
    )


# ============================================================================
# AGENT STATE MANAGEMENT
# ============================================================================

class AgentState:
    """
    Manages agent execution state for debugging and recovery.
    Tracks reasoning steps, timing, and errors.
    """

    def __init__(self, query: str, session_id: str):
        self.query = query
        self.session_id = session_id
        self.steps: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()
        self.status = "running"
        self.error: Optional[str] = None

    def add_step(self, thought: str, action: str, action_input: str, observation: str):
        """Record a reasoning step."""
        self.steps.append({
            "step_num": len(self.steps) + 1,
            "thought": thought[:200],  # Truncate for storage
            "action": action,
            "action_input": str(action_input)[:100],
            "observation": str(observation)[:200],
            "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"Step {len(self.steps)}: {action}({action_input[:50]}...)")

    def mark_complete(self, final_answer: str):
        """Mark agent execution as complete."""
        self.status = "complete"
        self.final_answer = final_answer
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Agent completed in {self.duration_seconds:.2f}s with {len(self.steps)} steps")

    def mark_failed(self, error: str):
        """Mark agent execution as failed."""
        self.status = "failed"
        self.error = error
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        logger.error(f"Agent failed after {self.duration_seconds:.2f}s: {error}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging/debugging."""
        return {
            "query": self.query,
            "session_id": self.session_id,
            "status": self.status,
            "steps": self.steps,
            "num_steps": len(self.steps),
            "duration_seconds": getattr(self, 'duration_seconds', None),
            "error": self.error,
            "final_answer": getattr(self, 'final_answer', None)[:500] if hasattr(self, 'final_answer') else None
        }

    def save_to_file(self, filepath: str):
        """Save state to file for debugging."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved agent trace to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")


# ============================================================================
# REACT AGENT
# ============================================================================

class ReActAgent:
    """
    ReAct agent that implements Thought → Action → Observation reasoning loop.
    Wraps LangChain's agent implementation with safety limits and error handling.
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_iterations: int = None,
        timeout_seconds: int = None
    ):
        """
        Initialize ReAct agent.

        Args:
            model_name: OpenAI model to use for reasoning (default: from config)
            temperature: 0 for deterministic, higher for creative (default: from config)
            max_iterations: Max Thought-Action cycles before stopping (default: from config)
            timeout_seconds: Total execution timeout (default: from config)
        """
        # Use config defaults if not provided
        self.model_name = model_name or Config.AGENT_MODEL
        self.temperature = temperature if temperature is not None else Config.AGENT_TEMPERATURE
        self.max_iterations = max_iterations or Config.AGENT_MAX_ITERATIONS
        self.timeout_seconds = timeout_seconds or Config.AGENT_TIMEOUT_SECONDS

        logger.info(f"Initializing ReAct agent: model={self.model_name}, max_iter={self.max_iterations}")

        # Initialize LLM
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=Config.OPENAI_API_KEY
        )

        # Get tools and prompt
        self.tools = get_tools()
        self.prompt = get_react_prompt(max_iterations=self.max_iterations)

        # Create ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor with safety limits
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            max_execution_time=self.timeout_seconds,
            verbose=True,  # Show reasoning process
            handle_parsing_errors=True,  # Gracefully handle bad formats
            early_stopping_method="generate",  # Return best answer so far if timeout
            return_intermediate_steps=True  # Capture reasoning steps
        )

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Execute ReAct reasoning loop for user query.

        Args:
            user_query: User's question

        Returns:
            Dict with 'output' (final answer), 'steps' (reasoning trace), 'error' (if any)
        """
        try:
            logger.info(f"Processing query: {user_query[:100]}...")

            # Execute agent
            result = self.agent_executor.invoke({"input": user_query})

            # Extract reasoning steps
            steps = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    steps.append({
                        "thought": getattr(action, 'log', '')[:200],
                        "action": action.tool,
                        "action_input": str(action.tool_input)[:100],
                        "observation": str(observation)[:200]
                    })

            logger.info(f"Agent completed: {len(steps)} steps")

            return {
                "output": result.get("output", ""),
                "steps": steps,
                "num_steps": len(steps),
                "error": None
            }

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")

            # Fallback to simple response
            if Config.FALLBACK_TO_STATIC:
                fallback_answer = self._fallback_query(user_query)
                return {
                    "output": fallback_answer,
                    "steps": [],
                    "num_steps": 0,
                    "error": f"Agent failed: {str(e)}. Used fallback pipeline."
                }
            else:
                return {
                    "output": "",
                    "steps": [],
                    "num_steps": 0,
                    "error": f"Agent failed: {str(e)}"
                }

    def _fallback_query(self, query: str) -> str:
        """Simple fallback when agent fails."""
        # In production, this would call your Level 1 static pipeline
        return f"I apologize, but I encountered an error processing your complex query. For '{query[:50]}...', please try rephrasing or breaking it into simpler questions."


class StatefulReActAgent(ReActAgent):
    """
    ReAct agent with state persistence and conversation memory.
    Tracks execution state for debugging and supports multi-turn conversations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history: Dict[str, List[Dict]] = {}

    def query(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Execute ReAct with state tracking.

        Args:
            user_query: User's question
            session_id: Optional session ID for multi-turn conversations

        Returns:
            Dict with output, steps, state, error
        """
        session_id = session_id or str(uuid.uuid4())

        # Initialize state
        state = AgentState(user_query, session_id)

        try:
            # Add conversation context if exists
            augmented_query = self._add_conversation_context(user_query, session_id)

            # Execute agent
            result = self.agent_executor.invoke({"input": augmented_query})

            # Capture steps
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    state.add_step(
                        thought=getattr(action, 'log', ''),
                        action=action.tool,
                        action_input=str(action.tool_input),
                        observation=str(observation)
                    )

            # Mark complete
            state.mark_complete(result.get("output", ""))

            # Save state for debugging
            trace_dir = "agent_traces"
            os.makedirs(trace_dir, exist_ok=True)
            state.save_to_file(f"{trace_dir}/{session_id}.json")

            # Update conversation history
            self._update_conversation_history(session_id, user_query, result.get("output", ""))

            return {
                "output": result.get("output", ""),
                "steps": state.steps,
                "num_steps": len(state.steps),
                "state": state.to_dict(),
                "error": None
            }

        except Exception as e:
            # Record failure
            state.mark_failed(str(e))
            state.save_to_file(f"agent_traces/{session_id}_failed.json")

            # Attempt fallback
            if Config.FALLBACK_TO_STATIC:
                fallback_answer = self._fallback_query(user_query)

                return {
                    "output": fallback_answer,
                    "steps": state.steps,
                    "num_steps": len(state.steps),
                    "state": state.to_dict(),
                    "error": f"Agent failed: {str(e)}. Used fallback."
                }
            else:
                return {
                    "output": "",
                    "steps": state.steps,
                    "num_steps": len(state.steps),
                    "state": state.to_dict(),
                    "error": f"Agent failed: {str(e)}"
                }

    def _add_conversation_context(self, query: str, session_id: str) -> str:
        """Add conversation history context to query."""
        history = self.conversation_history.get(session_id, [])

        if not history:
            return query

        # Build context from last 3 turns (to avoid exceeding context window)
        context_turns = history[-3:]
        context_str = "\n".join([
            f"Previous Q: {turn['query'][:100]}\nPrevious A: {turn['answer'][:100]}"
            for turn in context_turns
        ])

        return f"Conversation history:\n{context_str}\n\nCurrent question: {query}"

    def _update_conversation_history(self, session_id: str, query: str, answer: str):
        """Update conversation history."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep only last 5 turns
        if len(self.conversation_history[session_id]) > 5:
            self.conversation_history[session_id] = self.conversation_history[session_id][-5:]


# ============================================================================
# CLI USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys

    # Simple test
    print("=== ReAct Pattern Implementation Test ===\n")

    # Check configuration
    if not Config.OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not set - agent will not work")
        print("   Set it in .env file or environment variable")
        sys.exit(1)

    # Initialize agent
    print(f"Initializing agent (model={Config.AGENT_MODEL})...")
    agent = StatefulReActAgent()

    # Test queries
    test_queries = [
        "What is our refund policy?",  # Simple RAG search
        "What is 125000 * 1.15?",  # Calculator
        "What is the SaaS industry growth rate?",  # Industry data
    ]

    print(f"\nRunning {len(test_queries)} test queries...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[Query {i}] {query}")
        result = agent.query(query)

        print(f"  Steps: {result['num_steps']}")
        print(f"  Answer: {result['output'][:150]}...")
        if result['error']:
            print(f"  Error: {result['error']}")
        print()

    print("=== Test Complete ===")
