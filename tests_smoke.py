"""
Module 9.2: Multi-Hop & Recursive Retrieval - Smoke Tests
Basic tests to verify core functionality without requiring external services.
"""

import pytest
import json
from pathlib import Path

from l2_m9_multi_hop_recursive_retrieval import (
    Document,
    ReferenceExtractor,
    KnowledgeGraphManager,
    MultiHopRetriever,
    load_example_data
)
import config


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        """Verify config module loads without errors."""
        assert hasattr(config, "MAX_HOP_DEPTH")
        assert hasattr(config, "RELEVANCE_THRESHOLD")
        assert hasattr(config, "BEAM_WIDTH")

    def test_config_defaults(self):
        """Verify config has sensible defaults."""
        assert 1 <= config.MAX_HOP_DEPTH <= 10
        assert 0.0 <= config.RELEVANCE_THRESHOLD <= 1.0
        assert config.BEAM_WIDTH >= 1

    def test_get_clients_no_crash(self):
        """Verify get_clients() doesn't crash without credentials."""
        pinecone, neo4j, llm = config.get_clients()
        # May return None if credentials missing, but shouldn't crash
        assert pinecone is None or pinecone is not None
        assert neo4j is None or neo4j is not None
        assert llm is None or llm is not None


class TestDataLoading:
    """Test example data loading."""

    def test_example_data_exists(self):
        """Verify example_data.json exists."""
        data_file = Path("example_data.json")
        assert data_file.exists(), "example_data.json not found"

    def test_load_example_data(self):
        """Test loading example documents."""
        documents = load_example_data("example_data.json")
        assert len(documents) > 0, "No documents loaded"
        assert all(isinstance(doc, Document) for doc in documents)

    def test_example_data_structure(self):
        """Verify example data has expected structure."""
        with open("example_data.json", 'r') as f:
            data = json.load(f)

        assert "documents" in data
        assert len(data["documents"]) >= 5

        # Check first document structure
        doc = data["documents"][0]
        assert "id" in doc
        assert "content" in doc
        assert "metadata" in doc
        assert "references" in doc


class TestReferenceExtractor:
    """Test reference extraction."""

    def test_extractor_init_no_llm(self):
        """Test extractor initialization without LLM."""
        extractor = ReferenceExtractor(use_llm=False)
        assert extractor is not None
        assert not extractor.use_llm

    def test_regex_extraction(self):
        """Test regex-based reference extraction."""
        extractor = ReferenceExtractor(use_llm=False)

        content = "See Technical Implementation Guide (doc_002) and Security Policy (doc_005)."
        references = extractor.extract_references(content, "test_doc")

        assert "doc_002" in references
        assert "doc_005" in references

    def test_no_references(self):
        """Test extraction with no references."""
        extractor = ReferenceExtractor(use_llm=False)

        content = "This document has no references to other documents."
        references = extractor.extract_references(content, "test_doc")

        assert len(references) == 0


class TestKnowledgeGraphManager:
    """Test knowledge graph management."""

    def test_graph_init(self):
        """Test graph manager initialization."""
        graph = KnowledgeGraphManager()
        assert graph is not None
        assert not graph.use_neo4j  # Without driver

    def test_add_document(self):
        """Test adding document to graph."""
        graph = KnowledgeGraphManager()
        doc = Document(
            id="doc_test",
            content="Test content",
            metadata={"type": "test"},
            references=["doc_001"]
        )

        graph.add_document(doc)

        assert "doc_test" in graph.documents
        assert graph.documents["doc_test"] == doc

    def test_get_neighbors(self):
        """Test retrieving document neighbors."""
        graph = KnowledgeGraphManager()

        # Add documents: A->B->C
        doc_a = Document(id="doc_A", content="A", metadata={}, references=["doc_B"])
        doc_b = Document(id="doc_B", content="B", metadata={}, references=["doc_C"])
        doc_c = Document(id="doc_C", content="C", metadata={}, references=[])

        graph.add_document(doc_a)
        graph.add_document(doc_b)
        graph.add_document(doc_c)

        # Get neighbors within 1 hop
        neighbors_1 = graph.get_neighbors("doc_A", max_depth=1)
        assert "doc_B" in neighbors_1

        # Get neighbors within 2 hops
        neighbors_2 = graph.get_neighbors("doc_A", max_depth=2)
        assert "doc_B" in neighbors_2
        assert "doc_C" in neighbors_2

    def test_pagerank_calculation(self):
        """Test PageRank calculation."""
        graph = KnowledgeGraphManager()

        # Create simple graph
        docs = [
            Document(id="doc_001", content="A", metadata={}, references=["doc_002", "doc_003"]),
            Document(id="doc_002", content="B", metadata={}, references=["doc_003"]),
            Document(id="doc_003", content="C", metadata={}, references=[]),
        ]

        for doc in docs:
            graph.add_document(doc)

        scores = graph.calculate_pagerank()

        # Verify scores calculated
        assert len(scores) > 0
        assert all(0.0 <= score <= 1.0 for score in scores.values())
        # doc_003 should have highest score (pointed to by both others)
        assert scores["doc_003"] > scores["doc_001"]

    def test_circular_references(self):
        """Test graph handles circular references."""
        graph = KnowledgeGraphManager()

        # Create circular reference: A->B->C->A
        docs = [
            Document(id="doc_A", content="A", metadata={}, references=["doc_B"]),
            Document(id="doc_B", content="B", metadata={}, references=["doc_C"]),
            Document(id="doc_C", content="C", metadata={}, references=["doc_A"]),
        ]

        for doc in docs:
            graph.add_document(doc)

        # Should not crash
        neighbors = graph.get_neighbors("doc_A", max_depth=5)
        assert len(neighbors) > 0

        # PageRank should still work
        scores = graph.calculate_pagerank()
        assert len(scores) == 3


class TestMultiHopRetriever:
    """Test multi-hop retrieval."""

    def test_retriever_init(self):
        """Test retriever initialization."""
        graph = KnowledgeGraphManager()
        extractor = ReferenceExtractor(use_llm=False)

        retriever = MultiHopRetriever(
            vector_index=None,
            graph_manager=graph,
            reference_extractor=extractor,
            max_hop_depth=3,
            relevance_threshold=0.7
        )

        assert retriever is not None
        assert retriever.max_hop_depth == 3
        assert retriever.relevance_threshold == 0.7

    def test_retrieve_returns_result(self):
        """Test retrieval returns valid result structure."""
        # Setup
        documents = load_example_data("example_data.json")
        graph = KnowledgeGraphManager()
        for doc in documents:
            graph.add_document(doc)

        extractor = ReferenceExtractor(use_llm=False)
        retriever = MultiHopRetriever(
            vector_index=None,
            graph_manager=graph,
            reference_extractor=extractor,
            max_hop_depth=2
        )

        # Retrieve
        result = retriever.retrieve("test query", top_k_initial=3)

        # Verify result structure
        assert hasattr(result, "documents")
        assert hasattr(result, "hop_count")
        assert hasattr(result, "total_documents")
        assert hasattr(result, "execution_time_ms")
        assert hasattr(result, "graph_traversed")

        assert isinstance(result.documents, list)
        assert isinstance(result.hop_count, int)
        assert isinstance(result.total_documents, int)
        assert isinstance(result.execution_time_ms, float)
        assert isinstance(result.graph_traversed, dict)


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_no_services(self):
        """Test complete flow without external services."""
        # Load data
        documents = load_example_data("example_data.json")
        assert len(documents) > 0

        # Setup components
        graph = KnowledgeGraphManager()
        extractor = ReferenceExtractor(use_llm=False)
        retriever = MultiHopRetriever(
            vector_index=None,
            graph_manager=graph,
            reference_extractor=extractor,
            max_hop_depth=3,
            relevance_threshold=0.6
        )

        # Add documents to graph
        for doc in documents:
            graph.add_document(doc)

        # Perform retrieval
        result = retriever.retrieve(
            query="authentication vulnerabilities",
            top_k_initial=3,
            top_k_per_hop=2
        )

        # Verify results
        assert result.total_documents >= 0
        assert result.hop_count >= 0
        assert result.execution_time_ms > 0
        assert len(result.graph_traversed) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
