"""
Smoke tests for Module 9.1: Query Decomposition & Planning

Minimal tests to verify:
- Config loads
- Core functions return plausible shapes
- Network paths gracefully skip without keys
"""

import asyncio
import json
from pathlib import Path

import pytest

from l2_m9_query_decomposition_planning import (
    QueryDecomposer,
    DependencyGraph,
    ParallelExecutionEngine,
    AnswerSynthesizer,
    QueryDecompositionPipeline,
    SubQuery,
    DecompositionError,
    DependencyError,
    SynthesisError
)
from config import Config


def test_config_loads():
    """Test that configuration loads without errors."""
    assert Config is not None
    assert hasattr(Config, 'MAX_SUB_QUERIES')
    assert Config.MAX_SUB_QUERIES == 6
    assert hasattr(Config, 'MAX_CONCURRENT_RETRIEVALS')
    print("✓ Config loads successfully")


def test_example_data_exists():
    """Test that example data file exists and is valid JSON."""
    example_path = Path("example_data.json")
    assert example_path.exists(), "example_data.json not found"

    with open(example_path) as f:
        data = json.load(f)

    assert "sample_queries" in data
    assert len(data["sample_queries"]) > 0
    print("✓ Example data loads and is valid")


def test_dependency_graph_parallel():
    """Test dependency graph with parallel queries (no dependencies)."""
    queries = [
        SubQuery(id="q1", query="Test query 1", dependencies=[]),
        SubQuery(id="q2", query="Test query 2", dependencies=[]),
        SubQuery(id="q3", query="Test query 3", dependencies=[])
    ]

    graph = DependencyGraph(queries)
    levels = graph.get_execution_levels()

    # All parallel queries should be in one level
    assert len(levels) == 1
    assert len(levels[0]) == 3
    print("✓ Parallel dependency graph works")


def test_dependency_graph_sequential():
    """Test dependency graph with sequential dependencies."""
    queries = [
        SubQuery(id="q1", query="First", dependencies=[]),
        SubQuery(id="q2", query="Second", dependencies=["q1"]),
        SubQuery(id="q3", query="Third", dependencies=["q1", "q2"])
    ]

    graph = DependencyGraph(queries)
    levels = graph.get_execution_levels()

    # Should have 3 levels
    assert len(levels) == 3
    assert "q1" in levels[0]
    assert "q2" in levels[1]
    assert "q3" in levels[2]
    print("✓ Sequential dependency graph works")


def test_circular_dependency_detected():
    """Test that circular dependencies raise DependencyError."""
    queries = [
        SubQuery(id="q1", query="First", dependencies=["q2"]),
        SubQuery(id="q2", query="Second", dependencies=["q1"])
    ]

    with pytest.raises(DependencyError):
        DependencyGraph(queries)

    print("✓ Circular dependencies detected correctly")


@pytest.mark.asyncio
async def test_parallel_executor_mock():
    """Test parallel executor with mock retrieval."""
    async def mock_retrieval(query: str) -> str:
        await asyncio.sleep(0.01)
        return f"Result for: {query}"

    executor = ParallelExecutionEngine(mock_retrieval, max_concurrent=2)

    queries = [
        SubQuery(id="q1", query="Test 1", dependencies=[]),
        SubQuery(id="q2", query="Test 2", dependencies=[])
    ]

    results = await executor.execute_level(queries, {})

    assert len(results) == 2
    assert "q1" in results
    assert "q2" in results
    assert "Result for:" in results["q1"]
    print("✓ Parallel executor works with mock retrieval")


@pytest.mark.asyncio
async def test_decomposer_without_api_key():
    """Test that decomposer gracefully handles missing API key."""
    if Config.OPENAI_API_KEY:
        print("⚠️ Skipping (API key present)")
        return

    # This should work even without API key (initialization)
    decomposer = QueryDecomposer(api_key="fake-key")
    assert decomposer is not None
    print("✓ Decomposer initializes without errors")


@pytest.mark.asyncio
async def test_pipeline_without_api_key():
    """Test pipeline gracefully skips when no API key."""
    if Config.OPENAI_API_KEY:
        print("⚠️ Skipping test (API key present)")
        return

    async def mock_retrieval(query: str) -> str:
        return f"Mock: {query}"

    # Should initialize without errors
    pipeline = QueryDecompositionPipeline(
        api_key="fake-key",
        retrieval_function=mock_retrieval
    )
    assert pipeline is not None
    print("✓ Pipeline initializes without API key")


def test_subquery_dataclass():
    """Test SubQuery dataclass creation."""
    sq = SubQuery(id="q1", query="Test query", dependencies=["q0"])

    assert sq.id == "q1"
    assert sq.query == "Test query"
    assert sq.dependencies == ["q0"]
    assert sq.result is None
    print("✓ SubQuery dataclass works")


def test_config_validation():
    """Test config validation."""
    is_valid = Config.validate()

    if Config.OPENAI_API_KEY:
        assert is_valid is True
        print("✓ Config validation passes with API key")
    else:
        assert is_valid is False
        print("✓ Config validation correctly reports missing API key")


def test_config_get_dict():
    """Test config dictionary export."""
    config_dict = Config.get_config_dict()

    assert "max_sub_queries" in config_dict
    assert "max_concurrent_retrievals" in config_dict
    assert "enable_fallback" in config_dict
    assert config_dict["max_sub_queries"] == 6
    print("✓ Config exports to dictionary correctly")


# Run tests
if __name__ == "__main__":
    print("Running smoke tests for Module 9.1...\n")

    # Non-async tests
    test_config_loads()
    test_example_data_exists()
    test_dependency_graph_parallel()
    test_dependency_graph_sequential()
    test_circular_dependency_detected()
    test_subquery_dataclass()
    test_config_validation()
    test_config_get_dict()

    # Async tests
    print("\nRunning async tests...")
    asyncio.run(test_parallel_executor_mock())
    asyncio.run(test_decomposer_without_api_key())
    asyncio.run(test_pipeline_without_api_key())

    print("\n" + "="*60)
    print("✓ All smoke tests passed!")
    print("="*60)
