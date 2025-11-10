"""
Smoke tests for Module 9.4: Advanced Reranking Strategies

Minimal tests to verify core functionality without requiring external services.
"""

import pytest
import json
from typing import List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.l3_m9_advanced_reranking.l3_m9_advanced_reranking_strategies import (
    Document,
    EnsembleReranker,
    MMRReranker,
    TemporalReranker,
    PersonalizationReranker,
    AdvancedReranker,
    RerankResult
)
from src.l3_m9_advanced_reranking.config import RerankerConfig, get_config


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            text="PyTorch 2.0 released with performance improvements",
            metadata={
                "timestamp": "2024-03-15T10:00:00Z",
                "source": "technical_blog",
                "length": 50,
                "doc_type": "tutorial",
                "technical_depth": 0.8
            },
            score=0.8
        ),
        Document(
            id="doc2",
            text="TensorFlow for production ML systems",
            metadata={
                "timestamp": "2023-06-10T14:30:00Z",
                "source": "documentation",
                "length": 40,
                "doc_type": "reference",
                "technical_depth": 0.7
            },
            score=0.7
        ),
        Document(
            id="doc3",
            text="JAX for high-performance computing",
            metadata={
                "timestamp": "2024-01-20T09:15:00Z",
                "source": "research_paper",
                "length": 45,
                "doc_type": "research",
                "technical_depth": 0.9
            },
            score=0.75
        )
    ]


@pytest.fixture
def user_profile() -> dict:
    """Create sample user profile."""
    return {
        "user_id": "user_123",
        "interaction_count": 150,
        "preferences": {
            "preferred_sources": ["technical_blog", "tutorial"],
            "preferred_depth": 0.75,
            "preferred_length_range": [40, 60]
        }
    }


def test_config_loads():
    """Test that configuration loads successfully."""
    config = get_config()
    assert config is not None
    assert len(config.RERANKER_MODELS) > 0
    assert len(config.ENSEMBLE_WEIGHTS) == len(config.RERANKER_MODELS)
    assert 0.0 <= config.MMR_LAMBDA <= 1.0


def test_config_validation():
    """Test configuration validation."""
    config = RerankerConfig()
    assert config.validate() is True


def test_ensemble_reranker_init():
    """Test ensemble reranker initialization."""
    reranker = EnsembleReranker(
        model_names=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
        weights=[1.0],
        aggregation="weighted"
    )
    assert reranker is not None
    assert len(reranker.weights) == 1
    assert reranker.aggregation == "weighted"


def test_ensemble_rerank_returns_result(sample_documents):
    """Test that ensemble reranking returns valid result."""
    reranker = EnsembleReranker(
        model_names=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
        aggregation="weighted"
    )

    result = reranker.rerank(
        query="machine learning frameworks",
        documents=sample_documents,
        top_k=2
    )

    assert isinstance(result, RerankResult)
    assert len(result.documents) == 2
    assert result.latency_ms >= 0
    assert result.strategy_used == "ensemble"


def test_mmr_reranker_init():
    """Test MMR reranker initialization."""
    reranker = MMRReranker(lambda_param=0.7)
    assert reranker is not None
    assert reranker.lambda_param == 0.7


def test_mmr_rerank_returns_result(sample_documents):
    """Test that MMR reranking returns valid result."""
    reranker = MMRReranker(lambda_param=0.7)

    result = reranker.rerank(documents=sample_documents, top_k=2)

    assert isinstance(result, RerankResult)
    assert len(result.documents) == 2
    assert result.latency_ms >= 0
    assert result.strategy_used == "mmr"


def test_temporal_reranker_init():
    """Test temporal reranker initialization."""
    reranker = TemporalReranker(decay_days=30, boost_factor=1.5)
    assert reranker is not None
    assert reranker.decay_days == 30
    assert reranker.boost_factor == 1.5


def test_temporal_query_detection():
    """Test temporal query detection."""
    reranker = TemporalReranker()

    assert reranker.is_temporal_query("latest machine learning frameworks")
    assert reranker.is_temporal_query("current trends in AI")
    assert not reranker.is_temporal_query("introduction to Python")


def test_temporal_rerank_returns_result(sample_documents):
    """Test that temporal reranking returns valid result."""
    reranker = TemporalReranker(decay_days=30)

    result = reranker.rerank(
        query="latest frameworks",
        documents=sample_documents
    )

    assert isinstance(result, RerankResult)
    assert len(result.documents) == len(sample_documents)
    assert result.latency_ms >= 0
    assert result.strategy_used == "temporal"


def test_personalization_reranker_init():
    """Test personalization reranker initialization."""
    reranker = PersonalizationReranker(min_interactions=100)
    assert reranker is not None
    assert reranker.min_interactions == 100


def test_personalization_rerank_returns_result(sample_documents, user_profile):
    """Test that personalization reranking returns valid result."""
    reranker = PersonalizationReranker(min_interactions=100)

    result = reranker.rerank(
        documents=sample_documents,
        user_profile=user_profile
    )

    assert isinstance(result, RerankResult)
    assert len(result.documents) == len(sample_documents)
    assert result.latency_ms >= 0
    assert result.strategy_used == "personalization"


def test_personalization_insufficient_interactions(sample_documents):
    """Test personalization with insufficient interactions."""
    reranker = PersonalizationReranker(min_interactions=100)

    user_profile = {
        "user_id": "new_user",
        "interaction_count": 10,
        "preferences": {}
    }

    result = reranker.rerank(
        documents=sample_documents,
        user_profile=user_profile
    )

    assert result.debug_info["personalized"] is False
    assert result.debug_info["reason"] == "insufficient_interactions"


def test_advanced_reranker_init():
    """Test advanced reranker initialization."""
    reranker = AdvancedReranker(
        enable_ensemble=False,  # Disable to avoid model loading
        enable_mmr=True,
        enable_temporal=True,
        enable_personalization=True
    )

    assert reranker is not None
    assert reranker.mmr is not None
    assert reranker.temporal is not None
    assert reranker.personalization is not None


def test_advanced_rerank_pipeline(sample_documents, user_profile):
    """Test full advanced reranking pipeline."""
    reranker = AdvancedReranker(
        enable_ensemble=False,  # Disable to avoid model loading
        enable_mmr=True,
        enable_temporal=True,
        enable_personalization=True,
        config={"mmr_lambda": 0.7, "decay_days": 30, "boost_factor": 1.5}
    )

    result = reranker.rerank(
        query="latest machine learning frameworks",
        documents=sample_documents,
        user_profile=user_profile,
        top_k=2
    )

    assert isinstance(result, RerankResult)
    assert len(result.documents) <= 2
    assert result.latency_ms >= 0
    assert result.strategy_used == "combined"
    assert "pipeline_steps" in result.debug_info


def test_empty_documents_handling():
    """Test handling of empty document list."""
    reranker = MMRReranker()

    result = reranker.rerank(documents=[], top_k=5)

    assert isinstance(result, RerankResult)
    assert len(result.documents) == 0
    assert result.latency_ms >= 0


def test_document_dataclass():
    """Test Document dataclass creation."""
    doc = Document(
        id="test_doc",
        text="Test content",
        metadata={"key": "value"},
        score=0.5
    )

    assert doc.id == "test_doc"
    assert doc.text == "Test content"
    assert doc.metadata["key"] == "value"
    assert doc.score == 0.5


def test_example_data_loads():
    """Test that example data file loads successfully."""
    with open("example_data.json", "r") as f:
        data = json.load(f)

    assert "query" in data
    assert "documents" in data
    assert "user_profile" in data
    assert len(data["documents"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
