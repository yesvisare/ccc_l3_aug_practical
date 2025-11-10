"""
Smoke tests for ReAct Pattern Implementation.
Basic tests to verify core functionality without requiring API keys.
"""
import pytest
import json
import os
from unittest.mock import Mock, patch
from config import Config
from src.l3_m10_react_pattern_implementation import (
    calculator_tool,
    industry_data_tool,
    get_tools,
    AgentState
)


class TestConfig:
    """Test configuration loading and validation."""

    def test_config_loads(self):
        """Config should load without errors."""
        assert Config is not None
        assert hasattr(Config, 'AGENT_MODEL')
        assert hasattr(Config, 'AGENT_MAX_ITERATIONS')

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        assert Config.AGENT_MAX_ITERATIONS >= 1
        assert Config.AGENT_TIMEOUT_SECONDS > 0
        assert Config.AGENT_TEMPERATURE >= 0.0

    def test_config_info(self):
        """Config info should return dict without sensitive data."""
        info = Config.get_info()
        assert isinstance(info, dict)
        assert 'agent_model' in info
        assert 'OPENAI_API_KEY' not in info  # Should not expose actual key


class TestTools:
    """Test individual tool implementations."""

    def test_calculator_basic(self):
        """Calculator should handle basic arithmetic."""
        result = calculator_tool("2 + 2")
        assert "4" in result
        assert isinstance(result, str)  # Must return string for agent

    def test_calculator_complex(self):
        """Calculator should handle complex expressions."""
        result = calculator_tool("125000 * 1.15")
        assert "143750" in result or "143,750" in result

    def test_calculator_error_handling(self):
        """Calculator should handle invalid input gracefully."""
        result = calculator_tool("invalid expression")
        assert "error" in result.lower()
        assert isinstance(result, str)

    def test_industry_data_valid(self):
        """Industry data should return benchmarks."""
        result = industry_data_tool("SaaS,growth_rate")
        assert isinstance(result, str)
        assert "SaaS" in result
        assert "%" in result or "growth" in result.lower()

    def test_industry_data_invalid_format(self):
        """Industry data should handle invalid format."""
        result = industry_data_tool("invalid")
        assert "error" in result.lower() or "format" in result.lower()

    def test_industry_data_unknown_metric(self):
        """Industry data should handle unknown metrics."""
        result = industry_data_tool("SaaS,unknown_metric")
        assert isinstance(result, str)
        # Should return graceful message, not crash

    def test_tools_registry(self):
        """Tool registry should return valid tools."""
        tools = get_tools()
        assert len(tools) == 3  # RAG_Search, Calculator, Industry_Data

        tool_names = [t.name for t in tools]
        assert "RAG_Search" in tool_names
        assert "Calculator" in tool_names
        assert "Industry_Data" in tool_names

    def test_tool_outputs_are_strings(self):
        """All tools must return plain text strings, not dicts/JSON."""
        tools = get_tools()

        for tool in tools:
            # Test each tool with sample input
            if tool.name == "Calculator":
                result = tool.func("2 + 2")
            elif tool.name == "Industry_Data":
                result = tool.func("SaaS,growth_rate")
            else:  # RAG_Search
                result = tool.func("test query")

            assert isinstance(result, str), f"{tool.name} must return string, got {type(result)}"
            # Should not be JSON
            try:
                json.loads(result)
                # If it parses as JSON, that's a problem (should be plain text)
                assert False, f"{tool.name} returned JSON instead of plain text"
            except (json.JSONDecodeError, TypeError):
                pass  # Good - not JSON


class TestAgentState:
    """Test agent state management."""

    def test_agent_state_creation(self):
        """Agent state should initialize correctly."""
        state = AgentState("test query", "test-session")
        assert state.query == "test query"
        assert state.session_id == "test-session"
        assert state.status == "running"
        assert len(state.steps) == 0

    def test_agent_state_add_step(self):
        """Agent state should track steps."""
        state = AgentState("test", "session")
        state.add_step("thinking", "Calculator", "2+2", "4")

        assert len(state.steps) == 1
        assert state.steps[0]['action'] == "Calculator"

    def test_agent_state_mark_complete(self):
        """Agent state should mark completion."""
        state = AgentState("test", "session")
        state.mark_complete("final answer")

        assert state.status == "complete"
        assert hasattr(state, 'duration_seconds')
        assert state.final_answer == "final answer"

    def test_agent_state_mark_failed(self):
        """Agent state should mark failures."""
        state = AgentState("test", "session")
        state.mark_failed("error message")

        assert state.status == "failed"
        assert state.error == "error message"
        assert hasattr(state, 'duration_seconds')

    def test_agent_state_to_dict(self):
        """Agent state should serialize to dict."""
        state = AgentState("test", "session")
        state.add_step("thought", "action", "input", "obs")
        state.mark_complete("answer")

        data = state.to_dict()
        assert isinstance(data, dict)
        assert data['status'] == 'complete'
        assert data['num_steps'] == 1


class TestAgentWithoutKeys:
    """Test agent behavior without API keys (graceful degradation)."""

    def test_agent_requires_api_key(self):
        """Agent should fail gracefully without API key."""
        # Only test if key is actually missing
        if not Config.OPENAI_API_KEY:
            from src.l3_m10_react_pattern_implementation import ReActAgent

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                agent = ReActAgent()


class TestExampleData:
    """Test example data file is valid."""

    def test_example_data_exists(self):
        """Example data file should exist."""
        assert os.path.exists("example_data.json")

    def test_example_data_valid_json(self):
        """Example data should be valid JSON."""
        with open("example_data.json", 'r') as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "test_queries" in data

    def test_example_data_has_query_types(self):
        """Example data should have different query types."""
        with open("example_data.json", 'r') as f:
            data = json.load(f)

        assert "simple_rag" in data["test_queries"]
        assert "calculator" in data["test_queries"]
        assert "multi_step" in data["test_queries"]


@pytest.mark.asyncio
class TestAPI:
    """Test FastAPI endpoints (without starting server)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_query_endpoint_structure(self, client):
        """Query endpoint should accept correct format."""
        # This will likely fail without API key, but should have proper structure
        response = client.post(
            "/query",
            json={"query": "test", "use_agent": False}
        )
        # Should return 200 or 503 (service unavailable), not 422 (validation error)
        assert response.status_code in [200, 503, 500]


if __name__ == "__main__":
    """Run smoke tests."""
    print("Running smoke tests for ReAct Pattern Implementation...")
    print("=" * 60)

    # Run with pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-W", "ignore::DeprecationWarning"
    ])

    if exit_code == 0:
        print("\n✅ All smoke tests passed!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")

    exit(exit_code)
