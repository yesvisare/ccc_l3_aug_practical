"""
Smoke tests for Conversational RAG with Memory module.

Tests basic functionality without requiring live API keys or services.
"""

import json
import pytest
from unittest.mock import Mock, patch
from src.l3_m10_conversational_rag_memory import (
    Turn,
    ConversationMemoryManager,
    ReferenceResolver,
    SessionManager,
    ConversationalRAG,
)
from config import Config, validate_config


def test_config_loads():
    """Test that configuration loads without errors."""
    assert Config.SHORT_TERM_BUFFER_SIZE >= 1
    assert Config.MAX_CONTEXT_TOKENS > 0
    assert Config.REDIS_SESSION_TTL > 0


def test_validate_config():
    """Test configuration validation."""
    is_valid, warnings = validate_config()
    assert isinstance(is_valid, bool)
    assert isinstance(warnings, list)


def test_turn_serialization():
    """Test Turn serialization/deserialization."""
    turn = Turn(
        role="user", content="Hello world", timestamp=1234567890.0, entities=["world"]
    )

    # Serialize
    data = turn.to_dict()
    assert data["role"] == "user"
    assert data["content"] == "Hello world"

    # Deserialize
    restored = Turn.from_dict(data)
    assert restored.role == turn.role
    assert restored.content == turn.content
    assert restored.entities == turn.entities


def test_memory_manager_basic():
    """Test basic memory manager operations without LLM."""
    memory = ConversationMemoryManager(short_term_size=3, llm_client=None)

    # Add turns
    memory.add_turn("user", "What is Python?")
    memory.add_turn("assistant", "Python is a programming language.")

    # Get context
    context = memory.get_context()
    assert "What is Python?" in context
    assert "Python is a programming language" in context

    # Check buffer size
    assert len(memory.short_term_buffer) == 2


def test_memory_manager_buffer_overflow():
    """Test that memory manager handles buffer overflow."""
    memory = ConversationMemoryManager(short_term_size=2, llm_client=None)

    # Add more turns than buffer size
    for i in range(5):
        memory.add_turn("user", f"Message {i}")

    # Should keep only last 2 turns
    assert len(memory.short_term_buffer) <= 2


def test_memory_manager_serialization():
    """Test memory manager serialization."""
    memory = ConversationMemoryManager(short_term_size=5, llm_client=None)

    memory.add_turn("user", "Hello")
    memory.add_turn("assistant", "Hi there")

    # Serialize
    data = memory.to_dict()
    assert len(data["short_term_buffer"]) == 2

    # Deserialize
    restored = ConversationMemoryManager.from_dict(data)
    assert len(restored.short_term_buffer) == 2
    assert restored.short_term_buffer[0].content == "Hello"


def test_reference_resolver_no_spacy():
    """Test reference resolver gracefully handles missing spaCy."""
    with patch("src.l3_m10_conversational_rag_memory.logger"):
        resolver = ReferenceResolver()

        # Should still work without spaCy
        query = "What is it?"
        entities = ["Python"]
        resolved, modified = resolver.resolve_references(query, entities)

        # Will return unchanged if spaCy not available
        assert isinstance(resolved, str)
        assert isinstance(modified, bool)


def test_reference_resolver_extract_entities():
    """Test entity extraction (with or without spaCy)."""
    resolver = ReferenceResolver()

    text = "Python and JavaScript are programming languages."
    entities = resolver.extract_entities(text)

    # Should return a list (empty if spaCy not available)
    assert isinstance(entities, list)


def test_session_manager_no_redis():
    """Test session manager gracefully handles missing Redis."""
    manager = SessionManager(redis_client=None)

    memory = ConversationMemoryManager(short_term_size=5, llm_client=None)
    memory.add_turn("user", "Test")

    # Should handle gracefully
    result = manager.save_session("test-session", memory)
    assert result is False  # Expected to fail without Redis

    loaded = manager.load_session("test-session")
    assert loaded is None  # Expected to return None


def test_session_manager_with_mock_redis():
    """Test session manager with mocked Redis."""
    mock_redis = Mock()
    mock_redis.setex = Mock(return_value=True)
    mock_redis.get = Mock(return_value=None)

    manager = SessionManager(redis_client=mock_redis, ttl=3600)

    memory = ConversationMemoryManager(short_term_size=5, llm_client=None)
    memory.add_turn("user", "Test")

    # Save session
    result = manager.save_session("test-session", memory)
    assert result is True
    mock_redis.setex.assert_called_once()


def test_conversational_rag_initialization():
    """Test ConversationalRAG initialization."""
    mock_llm = Mock()

    rag = ConversationalRAG(
        llm_client=mock_llm, redis_client=None, short_term_size=5, model="gpt-4o-mini"
    )

    assert rag.llm_client == mock_llm
    assert rag.memory is not None
    assert rag.resolver is not None
    assert rag.session_manager is not None


def test_conversational_rag_memory_stats():
    """Test memory statistics retrieval."""
    mock_llm = Mock()
    rag = ConversationalRAG(llm_client=mock_llm)

    stats = rag.get_memory_stats()

    assert "short_term_turns" in stats
    assert "has_long_term_summary" in stats
    assert "estimated_tokens" in stats
    assert isinstance(stats["short_term_turns"], int)


def test_conversational_rag_reset():
    """Test memory reset."""
    mock_llm = Mock()
    rag = ConversationalRAG(llm_client=mock_llm)

    # Add some turns
    rag.memory.add_turn("user", "Test")
    assert len(rag.memory.short_term_buffer) > 0

    # Reset
    rag.reset_memory()
    assert len(rag.memory.short_term_buffer) == 0


def test_example_data_loads():
    """Test that example data file is valid JSON."""
    with open("configs/example_data.json", "r") as f:
        data = json.load(f)

    assert "scenarios" in data
    assert "failure_cases" in data
    assert len(data["scenarios"]) > 0


def test_token_estimation():
    """Test token estimation works."""
    memory = ConversationMemoryManager(short_term_size=5, llm_client=None)

    memory.add_turn("user", "Hello")
    tokens = memory._estimate_tokens()

    assert isinstance(tokens, int)
    assert tokens > 0


def test_get_recent_entities():
    """Test retrieval of recent entities."""
    memory = ConversationMemoryManager(short_term_size=5, llm_client=None)

    memory.add_turn("user", "Tell me about Python", entities=["Python"])
    memory.add_turn("assistant", "Python is great", entities=["Python"])

    recent = memory.get_recent_entities(n=2)
    assert isinstance(recent, list)
    assert "Python" in recent


# Performance/scale tests
def test_memory_handles_many_turns():
    """Test memory manager handles many turns without crashing."""
    memory = ConversationMemoryManager(short_term_size=5, llm_client=None)

    for i in range(100):
        memory.add_turn("user", f"Message {i}")

    # Should not crash and should maintain buffer size
    assert len(memory.short_term_buffer) <= 5


def test_replace_first_occurrence():
    """Test case-insensitive replacement helper."""
    from src.l3_m10_conversational_rag_memory import ReferenceResolver

    result = ReferenceResolver._replace_first_occurrence(
        "What is it and how does it work?", "it", "Python"
    )

    assert result == "What is Python and how does it work?"


if __name__ == "__main__":
    print("Running smoke tests...")
    pytest.main([__file__, "-v"])
