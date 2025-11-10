"""
Smoke tests for M11.4 Vector Index Sharding.

Tests basic functionality without requiring live API keys.
Validates:
- Config loading
- Shard routing determinism
- Function signatures and return types
- Graceful handling of missing services
"""

import pytest
from unittest.mock import Mock, MagicMock
from config import Config
from src.l3_m11_vector_index_sharding import ShardManager, ShardedRAG, monitor_shard_health


class TestConfig:
    """Test configuration loading."""

    def test_config_has_required_fields(self):
        """Config exposes all required settings."""
        assert hasattr(Config, "NUM_SHARDS")
        assert hasattr(Config, "SHARD_PREFIX")
        assert hasattr(Config, "VECTOR_DIMENSION")
        assert hasattr(Config, "MAX_VECTORS_PER_SHARD")
        assert hasattr(Config, "MAX_NAMESPACES_PER_SHARD")

    def test_config_defaults(self):
        """Config defaults are sensible."""
        assert Config.NUM_SHARDS >= 2
        assert Config.VECTOR_DIMENSION == 1536
        assert Config.MAX_VECTORS_PER_SHARD == 300000
        assert Config.MAX_NAMESPACES_PER_SHARD == 20


class TestShardManager:
    """Test shard routing logic."""

    def test_consistent_routing(self):
        """Same tenant always routes to same shard."""
        manager = ShardManager(redis_client=None, num_shards=4)

        tenant_id = "tenant-001"
        shard1 = manager.get_shard_for_tenant(tenant_id)
        shard2 = manager.get_shard_for_tenant(tenant_id)

        assert shard1 == shard2
        assert 0 <= shard1 < 4

    def test_different_tenants_can_differ(self):
        """Different tenants may route to different shards."""
        manager = ShardManager(redis_client=None, num_shards=4)

        # Not guaranteed different, but very likely with 4 shards
        tenants = [f"tenant-{i:03d}" for i in range(10)]
        shards = [manager.get_shard_for_tenant(t) for t in tenants]

        # Should have some distribution (not all same shard)
        assert len(set(shards)) > 1

    def test_index_name_format(self):
        """Index names follow expected format."""
        manager = ShardManager(redis_client=None, num_shards=4, shard_prefix="test-shard")

        index_name = manager.get_shard_index_name("tenant-001")

        assert index_name.startswith("test-shard-")
        assert index_name[-1].isdigit()

    def test_explicit_assignment_requires_redis(self):
        """Explicit assignment fails without Redis."""
        manager = ShardManager(redis_client=None, num_shards=4)

        with pytest.raises(RuntimeError, match="Redis required"):
            manager.assign_tenant_to_shard("tenant-001", 2)

    def test_explicit_assignment_validates_shard_id(self):
        """Explicit assignment validates shard ID range."""
        mock_redis = Mock()
        manager = ShardManager(redis_client=mock_redis, num_shards=4)

        with pytest.raises(ValueError, match="Invalid shard_id"):
            manager.assign_tenant_to_shard("tenant-001", 10)

    def test_shard_stats_without_redis(self):
        """Shard stats gracefully handle missing Redis."""
        manager = ShardManager(redis_client=None, num_shards=4)

        stats = manager.get_shard_stats()

        assert len(stats) == 4
        for i in range(4):
            assert stats[i]["tenant_count"] == 0


class TestShardedRAG:
    """Test sharded RAG operations."""

    def test_initialization(self):
        """ShardedRAG initializes with required components."""
        shard_manager = ShardManager(redis_client=None, num_shards=4)
        rag = ShardedRAG(
            pinecone_client=None,
            openai_client=None,
            shard_manager=shard_manager
        )

        assert rag.shard_manager == shard_manager
        assert rag.vector_dimension == 1536

    def test_embedding_without_openai(self):
        """Embedding gracefully fails without OpenAI."""
        shard_manager = ShardManager(redis_client=None, num_shards=4)
        rag = ShardedRAG(None, None, shard_manager)

        embedding = rag.create_embedding("test text")

        assert embedding is None

    def test_upsert_without_services(self):
        """Upsert gracefully skips without services."""
        shard_manager = ShardManager(redis_client=None, num_shards=4)
        rag = ShardedRAG(None, None, shard_manager)

        result = rag.upsert_documents("tenant-001", [
            {"id": "doc1", "text": "test", "metadata": {}}
        ])

        assert result["skipped"] is True
        assert "reason" in result

    def test_query_without_services(self):
        """Query gracefully skips without services."""
        shard_manager = ShardManager(redis_client=None, num_shards=4)
        rag = ShardedRAG(None, None, shard_manager)

        result = rag.query_single_tenant("tenant-001", "test query")

        assert result["skipped"] is True
        assert result["results"] == []

    def test_cross_shard_query_without_services(self):
        """Cross-shard query gracefully skips without services."""
        shard_manager = ShardManager(redis_client=None, num_shards=4)
        rag = ShardedRAG(None, None, shard_manager)

        result = rag.query_cross_shard("test query")

        assert result["skipped"] is True


class TestMonitoring:
    """Test shard health monitoring."""

    def test_monitor_without_pinecone(self):
        """Monitoring gracefully handles missing Pinecone."""
        manager = ShardManager(redis_client=None, num_shards=4)

        health = monitor_shard_health(manager, pinecone_client=None)

        assert health["needs_rebalancing"] is False
        assert "alerts" in health


class TestIntegration:
    """Integration tests with mock services."""

    def test_full_flow_mock(self):
        """Test complete flow with mocked services."""
        # Mock Redis
        mock_redis = MagicMock()
        mock_redis.hget.return_value = None
        mock_redis.hset.return_value = True
        mock_redis.hgetall.return_value = {
            "tenant-001": "0",
            "tenant-002": "1"
        }

        # Setup
        manager = ShardManager(redis_client=mock_redis, num_shards=4)

        # Test routing
        shard_id = manager.get_shard_for_tenant("tenant-001")
        assert 0 <= shard_id < 4

        # Test stats
        stats = manager.get_shard_stats()
        assert len(stats) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
