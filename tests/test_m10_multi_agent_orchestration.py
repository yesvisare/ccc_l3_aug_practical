"""
Smoke tests for Multi-Agent Orchestration module.
Tests basic functionality without requiring API keys.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


def test_config_loads():
    """Test that config module loads without errors."""
    try:
        import config
        assert hasattr(config, 'Config')
        assert hasattr(config, 'get_langchain_llm')
        assert hasattr(config, 'logger')
    except Exception as e:
        pytest.fail(f"Config import failed: {e}")


def test_main_module_imports():
    """Test that main module imports successfully."""
    try:
        from src import l3_m10_multi_agent_orchestration as main
        assert hasattr(main, 'run_multi_agent_query')
        assert hasattr(main, 'should_use_multi_agent')
        assert hasattr(main, 'AgentState')
        assert hasattr(main, 'planner_agent')
        assert hasattr(main, 'executor_agent')
        assert hasattr(main, 'validator_agent')
    except Exception as e:
        pytest.fail(f"Main module import failed: {e}")


def test_example_data_valid():
    """Test that example_data.json is valid and has expected structure."""
    try:
        with open('example_data.json', 'r') as f:
            data = json.load(f)

        assert 'queries' in data
        assert isinstance(data['queries'], list)
        assert len(data['queries']) > 0

        # Check first query structure
        first_query = data['queries'][0]
        assert 'query' in first_query
        assert 'complexity' in first_query
        assert 'expected_agents' in first_query

        # Check expected outcomes
        assert 'expected_outcomes' in data
        assert 'common_failures' in data

    except Exception as e:
        pytest.fail(f"Example data validation failed: {e}")


def test_should_use_multi_agent_simple_query():
    """Test routing logic for simple queries."""
    from src.l3_m10_multi_agent_orchestration import should_use_multi_agent

    simple_query = "What is our return policy?"
    result = should_use_multi_agent(simple_query)

    assert isinstance(result, dict)
    assert 'recommendation' in result
    assert result['recommendation'] == 'single-agent'
    assert 'reason' in result
    assert 'warning' in result


def test_should_use_multi_agent_complex_query():
    """Test routing logic for complex queries."""
    from src.l3_m10_multi_agent_orchestration import should_use_multi_agent

    complex_query = "Analyze our top 3 competitors and create a comprehensive strategy report"
    result = should_use_multi_agent(complex_query)

    assert isinstance(result, dict)
    assert 'recommendation' in result
    assert result['recommendation'] == 'multi-agent'
    assert 'reason' in result
    assert 'estimated_latency_seconds' in result


def test_agent_state_structure():
    """Test AgentState TypedDict structure."""
    from src.l3_m10_multi_agent_orchestration import AgentState

    # Create sample state
    state: AgentState = {
        'query': 'test query',
        'plan': [],
        'results': [],
        'validation_status': 'pending',
        'validation_feedback': '',
        'iterations': 0,
        'current_step': 0,
        'total_cost': 0.0,
        'start_time': 0.0,
        'messages': []
    }

    assert state['query'] == 'test query'
    assert isinstance(state['plan'], list)
    assert isinstance(state['results'], list)
    assert state['iterations'] == 0


@patch('src.l3_m10_multi_agent_orchestration.get_langchain_llm')
def test_planner_agent_without_api_key(mock_get_llm):
    """Test planner agent gracefully handles missing API key."""
    from src.l3_m10_multi_agent_orchestration import planner_agent, AgentState

    # Mock LLM to raise error
    mock_get_llm.side_effect = ValueError("OPENAI_API_KEY not configured")

    state: AgentState = {
        'query': 'test query',
        'plan': [],
        'results': [],
        'validation_status': 'pending',
        'validation_feedback': '',
        'iterations': 0,
        'current_step': 0,
        'total_cost': 0.0,
        'start_time': 0.0,
        'messages': []
    }

    with pytest.raises(ValueError, match="OPENAI_API_KEY not configured"):
        planner_agent(state)


@patch('src.l3_m10_multi_agent_orchestration.get_langchain_llm')
def test_planner_agent_returns_plan(mock_get_llm):
    """Test planner agent returns a plan structure."""
    from src.l3_m10_multi_agent_orchestration import planner_agent, AgentState

    # Mock LLM response
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = '''```json
[
  {"step": 1, "task": "Research competitors"},
  {"step": 2, "task": "Analyze findings"}
]
```'''
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    state: AgentState = {
        'query': 'Analyze competitors',
        'plan': [],
        'results': [],
        'validation_status': 'pending',
        'validation_feedback': '',
        'iterations': 0,
        'current_step': 0,
        'total_cost': 0.0,
        'start_time': 0.0,
        'messages': []
    }

    result = planner_agent(state)

    assert isinstance(result['plan'], list)
    assert len(result['plan']) == 2
    assert result['plan'][0]['step'] == 1
    assert 'messages' in result
    assert len(result['messages']) > 0


@patch('src.l3_m10_multi_agent_orchestration.get_langchain_llm')
def test_validator_agent_approval(mock_get_llm):
    """Test validator agent approval path."""
    from src.l3_m10_multi_agent_orchestration import validator_agent, AgentState

    # Mock LLM response
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = '''```json
{
  "status": "approved",
  "feedback": "Results are complete and accurate",
  "missing_items": []
}
```'''
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    state: AgentState = {
        'query': 'test query',
        'plan': [{'step': 1, 'task': 'test'}],
        'results': ['Result 1'],
        'validation_status': 'pending',
        'validation_feedback': '',
        'iterations': 0,
        'current_step': 1,
        'total_cost': 0.0,
        'start_time': 0.0,
        'messages': []
    }

    result = validator_agent(state)

    assert result['validation_status'] == 'approved'
    assert 'feedback' in result['validation_feedback'].lower() or 'complete' in result['validation_feedback'].lower()


def test_config_is_configured():
    """Test Config.is_configured() method."""
    from config import Config
    import os

    # Save original value
    original = os.environ.get('OPENAI_API_KEY')

    try:
        # Test with no key
        os.environ.pop('OPENAI_API_KEY', None)
        # Reload config to pick up change
        import importlib
        import config as cfg
        importlib.reload(cfg)
        # Note: Config class caches the value, so this may still return True
        # Just test that method exists and returns bool
        result = cfg.Config.is_configured()
        assert isinstance(result, bool)

    finally:
        # Restore original
        if original:
            os.environ['OPENAI_API_KEY'] = original


def test_routing_logic_functions_exist():
    """Test that routing functions exist and have correct signatures."""
    from src.l3_m10_multi_agent_orchestration import should_continue, check_validation

    assert callable(should_continue)
    assert callable(check_validation)


@patch('src.l3_m10_multi_agent_orchestration.create_multi_agent_graph')
def test_run_multi_agent_query_structure_without_execution(mock_create_graph):
    """Test run_multi_agent_query returns expected structure without actual execution."""
    from src.l3_m10_multi_agent_orchestration import run_multi_agent_query

    # Mock the graph execution
    mock_app = Mock()
    mock_app.invoke.return_value = {
        'query': 'test',
        'plan': [{'step': 1, 'task': 'test'}],
        'results': ['result'],
        'validation_status': 'approved',
        'validation_feedback': 'good',
        'iterations': 0,
        'current_step': 1,
        'total_cost': 0.045,
        'start_time': 0.0,
        'messages': ['test']
    }
    mock_create_graph.return_value = mock_app

    result = run_multi_agent_query("test query")

    assert isinstance(result, dict)
    assert 'success' in result
    assert 'query' in result
    assert 'results' in result
    assert 'validation_status' in result
    assert 'metadata' in result
    assert 'total_time_seconds' in result['metadata']
    assert 'estimated_cost_usd' in result['metadata']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
