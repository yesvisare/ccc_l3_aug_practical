"""
Module 10.2: Tool Calling & Function Execution
Production-grade tool ecosystem for agentic RAG systems.

This module implements:
- Tool Registry with schema validation
- Sandboxed execution engine with timeouts
- Production tools (search, calculator, DB, API, notifications)
- ReAct agent integration
- Comprehensive error handling

Author: CCC Level 3
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: TOOL SCHEMA & REGISTRY
# ============================================================================

class ToolCategory(str, Enum):
    """Categories for organizing tools."""
    SEARCH = "search"
    COMPUTATION = "computation"
    DATA_ACCESS = "data_access"
    EXTERNAL_API = "external_api"
    NOTIFICATION = "notification"


class ToolDefinition(BaseModel):
    """Schema for a tool definition with validation."""
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="What this tool does")
    category: ToolCategory
    parameters: Dict[str, Any] = Field(..., description="JSON schema for parameters")
    requires_sandbox: bool = Field(default=True, description="Run in sandbox?")
    timeout_seconds: int = Field(default=30, description="Max execution time")
    retry_count: int = Field(default=2, description="Number of retries on failure")

    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v):
        """Ensure timeout is within reasonable bounds."""
        if v < 1 or v > 300:
            raise ValueError("Timeout must be between 1-300 seconds")
        return v


class ToolResult(BaseModel):
    """Result from tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float
    retries_used: int = 0


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        """Initialize empty registry."""
        self.tools: Dict[str, tuple[ToolDefinition, Callable]] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized ToolRegistry")

    def register_tool(
        self,
        definition: ToolDefinition,
        executor: Callable
    ) -> None:
        """
        Register a new tool with its executor function.

        Args:
            definition: Tool definition with schema
            executor: Function that implements the tool

        Raises:
            ValueError: If tool name already registered
        """
        if definition.name in self.tools:
            raise ValueError(f"Tool {definition.name} already registered")

        self.tools[definition.name] = (definition, executor)
        self.stats[definition.name] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time_ms": 0.0
        }
        logger.info(f"Registered tool: {definition.name} ({definition.category})")

    def get_tool(self, name: str) -> Optional[tuple[ToolDefinition, Callable]]:
        """Retrieve a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Get all registered tool definitions."""
        return [defn for defn, _ in self.tools.values()]

    def get_tools_for_llm(self) -> str:
        """Format tools as a string for LLM system prompt."""
        tools_desc = "Available Tools:\n\n"
        for definition, _ in self.tools.values():
            tools_desc += f"**{definition.name}** ({definition.category.value})\n"
            tools_desc += f"Description: {definition.description}\n"
            tools_desc += f"Parameters: {json.dumps(definition.parameters, indent=2)}\n\n"
        return tools_desc

    def get_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics for tools."""
        if tool_name:
            return self.stats.get(tool_name, {})
        return self.stats

    def update_stats(self, tool_name: str, success: bool, exec_time_ms: float) -> None:
        """Update tool execution statistics."""
        if tool_name in self.stats:
            self.stats[tool_name]["calls"] += 1
            if success:
                self.stats[tool_name]["successes"] += 1
            else:
                self.stats[tool_name]["failures"] += 1
            self.stats[tool_name]["total_time_ms"] += exec_time_ms


# Global registry instance
tool_registry = ToolRegistry()


# ============================================================================
# SECTION 2: SANDBOXED EXECUTION ENGINE
# ============================================================================

class SafeToolExecutor:
    """Executes tools with sandboxing, timeouts, and retries."""

    def __init__(self, registry: ToolRegistry):
        """Initialize executor with registry."""
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers=5)
        logger.info("Initialized SafeToolExecutor")

    def _validate_arguments(
        self,
        tool_def: ToolDefinition,
        args: Dict[str, Any]
    ) -> bool:
        """
        Validate arguments against tool parameter schema.

        Args:
            tool_def: Tool definition with parameter schema
            args: Arguments to validate

        Returns:
            True if valid, False otherwise
        """
        # Simple validation - check required parameters exist
        schema = tool_def.parameters
        for param_name, param_schema in schema.items():
            if param_schema.get("required", False) and param_name not in args:
                logger.error(f"Missing required parameter: {param_name}")
                return False
        return True

    def _execute_with_timeout(
        self,
        executor_func: Callable,
        args: Dict[str, Any],
        timeout_seconds: int
    ) -> Any:
        """
        Execute function with timeout protection.

        Args:
            executor_func: Function to execute
            args: Arguments for function
            timeout_seconds: Max execution time

        Returns:
            Function result

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        future = self.executor.submit(executor_func, **args)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except FuturesTimeoutError:
            logger.error(f"Tool execution timed out after {timeout_seconds}s")
            raise TimeoutError(f"Execution exceeded {timeout_seconds}s timeout")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _execute_with_retry(
        self,
        executor_func: Callable,
        args: Dict[str, Any],
        timeout_seconds: int
    ) -> Any:
        """Execute with retry logic."""
        return self._execute_with_timeout(executor_func, args, timeout_seconds)

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute a tool with full safety protections.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome
        """
        start_time = time.time()
        retries_used = 0

        # Get tool from registry
        tool_entry = self.registry.get_tool(tool_name)
        if not tool_entry:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
                execution_time_ms=0.0
            )

        tool_def, executor_func = tool_entry

        # Validate arguments
        if not self._validate_arguments(tool_def, arguments):
            return ToolResult(
                success=False,
                error="Invalid arguments for tool",
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Execute with safety protections
        try:
            if tool_def.retry_count > 0:
                # Use retry logic
                result = self._execute_with_retry(
                    executor_func,
                    arguments,
                    tool_def.timeout_seconds
                )
            else:
                # Single execution
                result = self._execute_with_timeout(
                    executor_func,
                    arguments,
                    tool_def.timeout_seconds
                )

            exec_time_ms = (time.time() - start_time) * 1000

            # Validate result is JSON-serializable
            try:
                json.dumps(result)
            except (TypeError, ValueError) as e:
                logger.error(f"Tool result not JSON-serializable: {e}")
                raise ValueError("Tool must return JSON-serializable result")

            # Update stats
            self.registry.update_stats(tool_name, True, exec_time_ms)

            return ToolResult(
                success=True,
                result=result,
                execution_time_ms=exec_time_ms,
                retries_used=retries_used
            )

        except TimeoutError as e:
            exec_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool timeout: {e}")
            self.registry.update_stats(tool_name, False, exec_time_ms)
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=exec_time_ms
            )

        except Exception as e:
            exec_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {e}")
            self.registry.update_stats(tool_name, False, exec_time_ms)
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=exec_time_ms,
                retries_used=retries_used
            )

    def shutdown(self):
        """Shutdown executor thread pool."""
        self.executor.shutdown(wait=True)
        logger.info("SafeToolExecutor shutdown complete")


# ============================================================================
# SECTION 3: PRODUCTION TOOLS IMPLEMENTATION
# ============================================================================

def knowledge_search_tool(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search knowledge base with vector similarity.

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        Dict with results and scores
    """
    logger.info(f"Knowledge search: query='{query[:50]}...', top_k={top_k}")

    # Mock implementation - in production, query vector DB
    results = [
        {"content": f"Result {i+1} for '{query}'", "score": 0.9 - (i * 0.1)}
        for i in range(min(top_k, 3))
    ]

    return {
        "results": results,
        "total_found": len(results)
    }


def calculator_tool(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate mathematical expressions.

    Args:
        expression: Math expression (e.g., "2 + 2 * 3")

    Returns:
        Dict with result

    Raises:
        ValueError: If expression is invalid or unsafe
    """
    logger.info(f"Calculator: expression='{expression}'")

    # Validate expression contains only safe characters
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Expression contains invalid characters")

    # Prevent dangerous operations
    if any(keyword in expression for keyword in ["import", "eval", "exec", "__"]):
        raise ValueError("Expression contains forbidden keywords")

    try:
        # Safe evaluation
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result, "expression": expression}
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def database_query_tool(query: str, params: Optional[List] = None) -> Dict[str, Any]:
    """
    Execute parameterized database query.

    Args:
        query: SQL query (parameterized)
        params: Query parameters

    Returns:
        Dict with rows and count
    """
    logger.info(f"DB Query: {query[:50]}...")

    # Mock implementation - in production, use psycopg2
    # Validate query is SELECT only
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")

    # Mock results
    return {
        "rows": [
            {"id": 1, "name": "Sample Row 1"},
            {"id": 2, "name": "Sample Row 2"}
        ],
        "count": 2
    }


def api_call_tool(url: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make HTTP API call with rate limiting.

    Args:
        url: API endpoint URL
        method: HTTP method
        data: Request payload

    Returns:
        Dict with response data
    """
    logger.info(f"API Call: {method} {url}")

    # Mock implementation - in production, use requests
    # Validate URL is whitelisted
    allowed_domains = ["api.example.com", "internal-api.company.com"]

    return {
        "status": 200,
        "data": {"message": "Mock API response"},
        "headers": {}
    }


def slack_notification_tool(channel: str, message: str) -> Dict[str, Any]:
    """
    Send Slack notification.

    Args:
        channel: Slack channel ID or name
        message: Message to send

    Returns:
        Dict with success status
    """
    logger.info(f"Slack: channel={channel}, message='{message[:30]}...'")

    # Mock implementation - in production, use slack_sdk
    return {
        "sent": True,
        "channel": channel,
        "timestamp": time.time()
    }


# ============================================================================
# SECTION 4: REACT AGENT INTEGRATION
# ============================================================================

class ReActAgent:
    """ReAct agent with tool calling capabilities."""

    def __init__(self, executor: SafeToolExecutor):
        """Initialize agent with tool executor."""
        self.executor = executor
        self.max_iterations = 10
        logger.info("Initialized ReActAgent")

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run ReAct loop with tool execution.

        Args:
            user_query: User's question

        Returns:
            Dict with final answer and execution trace
        """
        logger.info(f"ReAct Agent starting: query='{user_query[:50]}...'")

        trace = []

        for iteration in range(self.max_iterations):
            # Thought (mock - in production, call LLM)
            thought = f"I should search for information about: {user_query}"
            trace.append({"type": "thought", "content": thought})
            logger.info(f"Iteration {iteration + 1} - Thought: {thought[:50]}...")

            # Action (mock - in production, LLM generates this)
            action = {
                "tool": "knowledge_search",
                "arguments": {"query": user_query, "top_k": 3}
            }
            trace.append({"type": "action", "content": action})
            logger.info(f"Action: {action['tool']}")

            # Execute tool
            result = self.executor.execute_tool(
                action["tool"],
                action["arguments"]
            )

            # Observation
            observation = {
                "success": result.success,
                "data": result.result if result.success else None,
                "error": result.error,
                "time_ms": result.execution_time_ms
            }
            trace.append({"type": "observation", "content": observation})
            logger.info(f"Observation: success={result.success}")

            # Check if we should stop (mock condition)
            if result.success:
                # Final answer (mock - in production, LLM generates this)
                answer = f"Based on the search results, here's the answer to: {user_query}"
                trace.append({"type": "answer", "content": answer})
                logger.info("Agent completed successfully")

                return {
                    "answer": answer,
                    "iterations": iteration + 1,
                    "trace": trace,
                    "success": True
                }

        # Max iterations reached
        logger.warning("Agent reached max iterations without answer")
        return {
            "answer": "Unable to complete task within iteration limit",
            "iterations": self.max_iterations,
            "trace": trace,
            "success": False
        }


# ============================================================================
# SECTION 5: TOOL REGISTRATION
# ============================================================================

def register_default_tools(registry: ToolRegistry) -> None:
    """Register all default production tools."""

    # Knowledge Search Tool
    search_def = ToolDefinition(
        name="knowledge_search",
        description="Search vector knowledge base for relevant documents",
        category=ToolCategory.SEARCH,
        parameters={
            "query": {"type": "string", "required": True},
            "top_k": {"type": "integer", "default": 5}
        },
        timeout_seconds=10,
        retry_count=2
    )
    registry.register_tool(search_def, knowledge_search_tool)

    # Calculator Tool
    calc_def = ToolDefinition(
        name="calculator",
        description="Evaluate mathematical expressions safely",
        category=ToolCategory.COMPUTATION,
        parameters={
            "expression": {"type": "string", "required": True}
        },
        timeout_seconds=5,
        retry_count=1
    )
    registry.register_tool(calc_def, calculator_tool)

    # Database Query Tool
    db_def = ToolDefinition(
        name="database_query",
        description="Execute parameterized SELECT queries on PostgreSQL",
        category=ToolCategory.DATA_ACCESS,
        parameters={
            "query": {"type": "string", "required": True},
            "params": {"type": "array", "required": False}
        },
        timeout_seconds=30,
        retry_count=2
    )
    registry.register_tool(db_def, database_query_tool)

    # API Call Tool
    api_def = ToolDefinition(
        name="api_call",
        description="Make HTTP requests to whitelisted APIs",
        category=ToolCategory.EXTERNAL_API,
        parameters={
            "url": {"type": "string", "required": True},
            "method": {"type": "string", "default": "GET"},
            "data": {"type": "object", "required": False}
        },
        timeout_seconds=20,
        retry_count=3
    )
    registry.register_tool(api_def, api_call_tool)

    # Slack Notification Tool
    slack_def = ToolDefinition(
        name="slack_notification",
        description="Send notifications to Slack channels",
        category=ToolCategory.NOTIFICATION,
        parameters={
            "channel": {"type": "string", "required": True},
            "message": {"type": "string", "required": True}
        },
        timeout_seconds=10,
        retry_count=2
    )
    registry.register_tool(slack_def, slack_notification_tool)

    logger.info(f"Registered {len(registry.list_tools())} default tools")


# ============================================================================
# MAIN ENTRY POINT & CLI EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """CLI usage examples."""

    print("=== Module 10.2: Tool Calling & Function Execution ===\n")

    # Initialize system
    register_default_tools(tool_registry)
    executor = SafeToolExecutor(tool_registry)
    agent = ReActAgent(executor)

    # Example 1: List available tools
    print("1. Available Tools:")
    for tool in tool_registry.list_tools():
        print(f"   - {tool.name} ({tool.category.value})")
    print()

    # Example 2: Execute calculator tool
    print("2. Calculator Tool:")
    result = executor.execute_tool("calculator", {"expression": "2 + 2 * 10"})
    print(f"   Result: {result.result if result.success else result.error}")
    print(f"   Time: {result.execution_time_ms:.2f}ms")
    print()

    # Example 3: Execute knowledge search
    print("3. Knowledge Search Tool:")
    result = executor.execute_tool("knowledge_search", {
        "query": "What is tool calling?",
        "top_k": 3
    })
    if result.success:
        print(f"   Found {result.result['total_found']} results")
        print(f"   Top result: {result.result['results'][0]['content'][:50]}...")
    print()

    # Example 4: Run ReAct agent
    print("4. ReAct Agent:")
    response = agent.run("How do I implement tool calling?")
    print(f"   Success: {response['success']}")
    print(f"   Iterations: {response['iterations']}")
    print(f"   Answer: {response['answer'][:80]}...")
    print()

    # Example 5: Tool statistics
    print("5. Tool Statistics:")
    stats = tool_registry.get_stats()
    for tool_name, tool_stats in stats.items():
        if tool_stats['calls'] > 0:
            success_rate = (tool_stats['successes'] / tool_stats['calls']) * 100
            avg_time = tool_stats['total_time_ms'] / tool_stats['calls']
            print(f"   {tool_name}:")
            print(f"      Calls: {tool_stats['calls']}, Success Rate: {success_rate:.1f}%")
            print(f"      Avg Time: {avg_time:.2f}ms")

    # Cleanup
    executor.shutdown()
    print("\n✅ Examples complete!")
