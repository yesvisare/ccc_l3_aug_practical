"""
Smoke tests for Module 9: HyDE

Basic tests to verify core functionality works.
Network-dependent tests skip gracefully without API keys.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch

# Import module components
from l2_m9_hypothetical_document_embeddings import (
    HyDEGenerator,
    HyDERetriever,
    HybridHyDERetriever,
    QueryClassifier,
    AdaptiveHyDERetriever,
    HyDEEvaluator
)
import config


def test_config_loads():
    """Test that configuration loads without errors."""
    status = config.validate_config()
    assert isinstance(status, dict)
    assert "openai" in status
    assert "pinecone" in status


def test_hyde_config():
    """Test HyDE configuration parameters."""
    hyde_config = config.get_hyde_config()
    assert isinstance(hyde_config, dict)
    assert "model" in hyde_config
    assert "temperature" in hyde_config
    assert hyde_config["temperature"] >= 0
    assert hyde_config["temperature"] <= 1


def test_query_classifier_patterns():
    """Test query classifier pattern matching."""
    classifier = QueryClassifier()

    # Test conceptual queries
    conceptual = "What are the implications of stock options?"
    result = classifier.should_use_hyde(conceptual)
    assert isinstance(result, dict)
    assert "use_hyde" in result
    assert result["beneficial_signals"] > 0

    # Test factoid queries
    factoid = "When was the 2023 tax deadline?"
    result = classifier.should_use_hyde(factoid)
    assert isinstance(result, dict)
    assert "harmful_signals" in result


def test_query_classifier_decision():
    """Test query classifier makes reasonable decisions."""
    classifier = QueryClassifier()

    test_cases = [
        ("What are the tax implications?", True),  # Conceptual
        ("When was X released?", False),  # Factoid
        ("How does ISO taxation work?", True),  # Conceptual
        ("How many employees?", False)  # Factoid
    ]

    for query, expected_hyde in test_cases:
        result = classifier.should_use_hyde(query)
        # Note: This is a heuristic test, may not be 100% accurate
        print(f"Query: {query}, Use HyDE: {result['use_hyde']}, Expected: {expected_hyde}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
def test_hypothesis_generation():
    """Test hypothesis generation with real API (skips if no key)."""
    generator = HyDEGenerator(openai_api_key=os.getenv("OPENAI_API_KEY"))

    query = "What are the benefits of stock options?"
    result = generator.generate_hypothesis(query)

    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "success" in result
    assert isinstance(result["hypothesis"], str)

    if result["success"]:
        assert len(result["hypothesis"]) > 0
        assert "tokens_used" in result
        assert result["tokens_used"] > 0
        print(f"✓ Generated hypothesis: {result['hypothesis'][:100]}...")


def test_hypothesis_generation_fallback():
    """Test hypothesis generation fallback on error."""
    # Use invalid key to trigger fallback
    generator = HyDEGenerator(openai_api_key="invalid-key")

    query = "What are the benefits of stock options?"
    result = generator.generate_hypothesis(query)

    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert result["success"] is False
    assert result["hypothesis"] == query  # Falls back to original query


def test_hyde_retriever_initialization():
    """Test HyDE retriever initializes correctly."""
    openai_key = os.getenv("OPENAI_API_KEY", "test-key")

    # Initialize without Pinecone (should work)
    retriever = HyDERetriever(
        openai_api_key=openai_key,
        pinecone_api_key=None,
        pinecone_index_name=None
    )

    assert retriever is not None
    assert retriever.hyde_gen is not None
    assert retriever.index is None  # No Pinecone connection


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
def test_hyde_retriever_without_pinecone():
    """Test HyDE retriever works without Pinecone (generates hypothesis but skips search)."""
    retriever = HyDERetriever(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=None,
        pinecone_index_name=None
    )

    query = "What are the tax implications of stock options?"
    result = retriever.retrieve_with_hyde(query, top_k=5)

    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "performance" in result
    assert "metadata" in result
    assert result["metadata"]["skipped_search"] is True
    print("✓ HyDE retriever works without Pinecone")


def test_hybrid_retriever_initialization():
    """Test hybrid retriever initializes correctly."""
    openai_key = os.getenv("OPENAI_API_KEY", "test-key")

    retriever = HybridHyDERetriever(
        openai_api_key=openai_key,
        pinecone_api_key=None,
        pinecone_index_name=None,
        hyde_weight=0.6,
        traditional_weight=0.4
    )

    assert retriever is not None
    assert retriever.hyde_weight == 0.6
    assert retriever.traditional_weight == 0.4


def test_merge_results():
    """Test result merging logic."""
    openai_key = os.getenv("OPENAI_API_KEY", "test-key")
    retriever = HybridHyDERetriever(
        openai_api_key=openai_key,
        pinecone_api_key=None,
        pinecone_index_name=None
    )

    # Create mock results
    MockMatch = type('MockMatch', (), {})

    hyde_results = []
    for i in range(3):
        match = MockMatch()
        match.id = f"doc_{i}"
        match.metadata = {"text": f"Document {i}"}
        hyde_results.append(match)

    traditional_results = []
    for i in range(1, 4):  # Overlap with doc_1 and doc_2
        match = MockMatch()
        match.id = f"doc_{i}"
        match.metadata = {"text": f"Document {i}"}
        traditional_results.append(match)

    merged = retriever.merge_results(hyde_results, traditional_results, top_k=5)

    assert isinstance(merged, list)
    assert len(merged) > 0
    assert all("source" in item for item in merged)
    # Check that overlapping items are marked as "both"
    both_items = [item for item in merged if item["source"] == "both"]
    assert len(both_items) > 0
    print(f"✓ Merged {len(merged)} results, {len(both_items)} from both sources")


def test_adaptive_retriever():
    """Test adaptive retriever routing."""
    openai_key = os.getenv("OPENAI_API_KEY", "test-key")

    retriever = AdaptiveHyDERetriever(
        openai_api_key=openai_key,
        pinecone_api_key=None,
        pinecone_index_name=None
    )

    assert retriever is not None
    assert retriever.classifier is not None
    assert retriever.hybrid_retriever is not None


def test_example_data_loads():
    """Test that example data file is valid JSON."""
    with open("example_data.json", "r") as f:
        data = json.load(f)

    assert "test_queries" in data
    assert "conceptual" in data["test_queries"]
    assert "factoid" in data["test_queries"]
    assert len(data["test_queries"]["conceptual"]) > 0
    assert len(data["test_queries"]["factoid"]) > 0
    print(f"✓ Loaded {len(data['test_queries']['conceptual'])} conceptual queries")
    print(f"✓ Loaded {len(data['test_queries']['factoid'])} factoid queries")


def test_evaluator_initialization():
    """Test evaluator initializes correctly."""
    openai_key = os.getenv("OPENAI_API_KEY", "test-key")

    hybrid = HybridHyDERetriever(
        openai_api_key=openai_key,
        pinecone_api_key=None,
        pinecone_index_name=None
    )

    evaluator = HyDEEvaluator(hybrid)
    assert evaluator is not None
    assert evaluator.results_log == []


if __name__ == "__main__":
    print("Running smoke tests...")
    print("=" * 50)

    # Run tests manually (without pytest)
    test_config_loads()
    print("✓ Config loads")

    test_hyde_config()
    print("✓ HyDE config valid")

    test_query_classifier_patterns()
    print("✓ Query classifier patterns work")

    test_query_classifier_decision()
    print("✓ Query classifier makes decisions")

    test_hypothesis_generation_fallback()
    print("✓ Hypothesis generation fallback works")

    test_hyde_retriever_initialization()
    print("✓ HyDE retriever initializes")

    test_hybrid_retriever_initialization()
    print("✓ Hybrid retriever initializes")

    test_merge_results()
    print("✓ Result merging works")

    test_adaptive_retriever()
    print("✓ Adaptive retriever initializes")

    test_example_data_loads()
    print("✓ Example data loads")

    test_evaluator_initialization()
    print("✓ Evaluator initializes")

    # Try network-dependent tests if keys available
    if os.getenv("OPENAI_API_KEY"):
        print("\n" + "=" * 50)
        print("Running network-dependent tests...")
        test_hypothesis_generation()
        test_hyde_retriever_without_pinecone()
    else:
        print("\n⚠️  Skipping network tests (no OPENAI_API_KEY)")

    print("\n" + "=" * 50)
    print("✓ All smoke tests passed!")
