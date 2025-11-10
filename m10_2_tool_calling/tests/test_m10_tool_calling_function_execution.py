"""
Smoke tests for Module 10.2: Tool Calling & Function Execution

Minimal tests to verify:
- Config loads
- Core functions return plausible shapes
- Network paths gracefully skip without keys
"""

import pytest
import json
from pathlib import Path

# Import modules to test
import config
from src.l3_m10_tool_calling_function_execution import (
    ToolRegistry,
    ToolDefinition,
    ToolCategory,
    SafeToolExecutor,
    ReActAgent,
    register_default_tools,
    tool_registry as global_registry
)


# ============================================================================
# CONFIG TESTS
# ============================================================================

def test_config_loads():
    """Test that config module loads without errors."""
    assert hasattr(config, 'DEFAULT_TIMEOUT_SECONDS')
    assert hasattr(config, 'MAX_AGENT_ITERATIONS')
    assert config.DEFAULT_TIMEOUT_SECONDS > 0


def test_config_validation():
    """Test config validation returns dict."""
    result = config.validate_config()
    assert isinstance(result, dict)
    assert 'database' in result
    assert 'openai' in result


def test_is_configured():
    """Test service configuration check."""
    # Should return bool
    result = config.is_configured('database')
    assert isinstance(result, bool)


# ============================================================================
# TOOL REGISTRY TESTS
# ============================================================================

def test_tool_registry_initialization():
    """Test ToolRegistry initializes correctly."""
    registry = ToolRegistry()
    assert len(registry.tools) == 0
    assert len(registry.stats) == 0


def test_tool_definition_validation():
    """Test ToolDefinition validates parameters."""
    # Valid definition
    tool_def = ToolDefinition(
        name="test_tool",
        description="A test tool",
        category=ToolCategory.COMPUTATION,
        parameters={"x": {"type": "number"}},
        timeout_seconds=10
    )
    assert tool_def.name == "test_tool"
    assert tool_def.timeout_seconds == 10

    # Invalid timeout should raise
    with pytest.raises(ValueError):
        ToolDefinition(
            name="bad_tool",
            description="Bad timeout",
            category=ToolCategory.COMPUTATION,
            parameters={},
            timeout_seconds=500  # Exceeds max 300
        )


def test_register_default_tools():
    """Test default tools register successfully."""
    registry = ToolRegistry()
    register_default_tools(registry)

    tools = registry.list_tools()
    assert len(tools) == 5

    tool_names = [t.name for t in tools]
    assert "knowledge_search" in tool_names
    assert "calculator" in tool_names
    assert "database_query" in tool_names


def test_get_tools_for_llm():
    """Test LLM context string generation."""
    registry = ToolRegistry()
    register_default_tools(registry)

    llm_context = registry.get_tools_for_llm()
    assert isinstance(llm_context, str)
    assert "Available Tools" in llm_context
    assert "calculator" in llm_context


# ============================================================================
# TOOL EXECUTION TESTS
# ============================================================================

def test_safe_executor_initialization():
    """Test SafeToolExecutor initializes."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    assert executor.registry == registry
    assert executor.executor is not None

    executor.shutdown()


def test_calculator_tool_execution():
    """Test calculator tool executes successfully."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    result = executor.execute_tool("calculator", {"expression": "2 + 2"})

    assert result.success is True
    assert result.result == {"result": 4, "expression": "2 + 2"}
    assert result.execution_time_ms >= 0

    executor.shutdown()


def test_calculator_blocks_injection():
    """Test calculator blocks code injection attempts."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    # Attempt code injection
    result = executor.execute_tool("calculator", {"expression": "import os"})

    assert result.success is False
    assert "invalid" in result.error.lower() or "forbidden" in result.error.lower()

    executor.shutdown()


def test_knowledge_search_tool():
    """Test knowledge search returns plausible shape."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    result = executor.execute_tool("knowledge_search", {
        "query": "test query",
        "top_k": 3
    })

    assert result.success is True
    assert "results" in result.result
    assert "total_found" in result.result
    assert isinstance(result.result["results"], list)

    executor.shutdown()


def test_database_query_blocks_non_select():
    """Test database tool blocks non-SELECT queries."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    result = executor.execute_tool("database_query", {
        "query": "DROP TABLE users"
    })

    assert result.success is False
    assert "SELECT" in result.error

    executor.shutdown()


def test_invalid_tool_name():
    """Test executor handles invalid tool name gracefully."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    result = executor.execute_tool("nonexistent_tool", {})

    assert result.success is False
    assert "not found" in result.error.lower()

    executor.shutdown()


def test_missing_required_argument():
    """Test executor validates required arguments."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    # knowledge_search requires 'query' parameter
    result = executor.execute_tool("knowledge_search", {})

    assert result.success is False

    executor.shutdown()


# ============================================================================
# REACT AGENT TESTS
# ============================================================================

def test_react_agent_initialization():
    """Test ReActAgent initializes correctly."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)
    agent = ReActAgent(executor)

    assert agent.executor == executor
    assert agent.max_iterations == 10

    executor.shutdown()


def test_react_agent_run():
    """Test ReAct agent completes a run."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)
    agent = ReActAgent(executor)

    response = agent.run("Test query")

    assert isinstance(response, dict)
    assert "answer" in response
    assert "success" in response
    assert "iterations" in response
    assert "trace" in response
    assert isinstance(response["trace"], list)

    executor.shutdown()


# ============================================================================
# STATISTICS TESTS
# ============================================================================

def test_tool_statistics_tracking():
    """Test tool execution statistics are tracked."""
    registry = ToolRegistry()
    register_default_tools(registry)
    executor = SafeToolExecutor(registry)

    # Execute a tool
    executor.execute_tool("calculator", {"expression": "1 + 1"})

    # Check stats
    stats = registry.get_stats("calculator")
    assert stats["calls"] == 1
    assert stats["successes"] == 1
    assert stats["failures"] == 0

    executor.shutdown()


# ============================================================================
# DATA FILE TESTS
# ============================================================================

def test_example_data_exists():
    """Test example_data.json exists and is valid."""
    data_path = Path(__file__).parent.parent / "example_data.json"
    assert data_path.exists()

    with open(data_path) as f:
        data = json.load(f)

    assert "sample_queries" in data
    assert "sample_tool_calls" in data
    assert "failure_scenarios" in data


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    print("=" * 60)
    print("Running Smoke Tests for Module 10.2")
    print("=" * 60 + "\n")

    pytest.main([__file__, "-v", "--tb=short"])
