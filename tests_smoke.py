"""
Smoke tests for Module 11.3: Resource Management & Throttling
Basic tests to verify core functionality.
"""

import pytest
import redis
import asyncio
import time
from unittest.mock import Mock, patch

# Import our modules
from l2_m11_resource_management_throttling import (
    QuotaManager,
    QuotaType,
    TenantQuotas,
    FairTenantQueue,
    QueuedRequest,
    ResourceWeightedQuota,
    AtomicQuotaManager
)


class TestConfig:
    """Test configuration loading"""

    def test_config_loads(self):
        """Config module imports without errors"""
        try:
            from config import Config
            assert Config.REDIS_HOST is not None
            assert Config.REDIS_PORT > 0
        except Exception as e:
            pytest.skip(f"Config not available: {e}")


class TestQuotaManager:
    """Test QuotaManager functionality"""

    @pytest.fixture
    def redis_client(self):
        """Get Redis client or skip if unavailable"""
        try:
            r = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=2
            )
            r.ping()
            # Clean up test keys
            for key in r.scan_iter("quota:test_*"):
                r.delete(key)
            for key in r.scan_iter("tenant:test_*"):
                r.delete(key)
            return r
        except (redis.ConnectionError, redis.TimeoutError):
            pytest.skip("Redis not available")

    def test_quota_manager_initializes(self, redis_client):
        """QuotaManager initializes with valid tiers"""
        qm = QuotaManager(redis_client)
        assert "free" in qm.quota_tiers
        assert "pro" in qm.quota_tiers
        assert "enterprise" in qm.quota_tiers

    def test_set_tenant_tier(self, redis_client):
        """Can set and retrieve tenant tier"""
        qm = QuotaManager(redis_client)
        qm.set_tenant_tier("test_tenant_1", "pro")

        tier = qm.get_tenant_tier("test_tenant_1")
        assert tier.tier == "pro"
        assert tier.queries_per_hour == 1000

    def test_invalid_tier_raises_error(self, redis_client):
        """Setting invalid tier raises ValueError"""
        qm = QuotaManager(redis_client)
        with pytest.raises(ValueError):
            qm.set_tenant_tier("test_tenant_2", "invalid_tier")

    def test_quota_check_under_limit(self, redis_client):
        """Quota check returns True when under limit"""
        qm = QuotaManager(redis_client)
        qm.set_tenant_tier("test_tenant_3", "free")

        under_quota, current, limit = qm.check_quota(
            "test_tenant_3",
            QuotaType.QUERIES_HOURLY,
            increment=1
        )

        assert under_quota is True
        assert current == 0
        assert limit == 100

    def test_quota_increment(self, redis_client):
        """Usage increments correctly"""
        qm = QuotaManager(redis_client)
        qm.set_tenant_tier("test_tenant_4", "free")

        # Increment usage
        qm.increment_usage("test_tenant_4", QuotaType.QUERIES_HOURLY, 5)

        # Check usage
        usage = qm.get_usage("test_tenant_4", QuotaType.QUERIES_HOURLY)
        assert usage == 5

    def test_record_query(self, redis_client):
        """record_query increments all counters"""
        qm = QuotaManager(redis_client)
        qm.set_tenant_tier("test_tenant_5", "free")

        results = qm.record_query("test_tenant_5", tokens_used=1000)

        assert "queries_hourly" in results
        assert "queries_daily" in results
        assert "queries_monthly" in results
        assert "tokens_monthly" in results

    def test_quota_status(self, redis_client):
        """get_quota_status returns proper structure"""
        qm = QuotaManager(redis_client)
        qm.set_tenant_tier("test_tenant_6", "pro")

        status = qm.get_quota_status("test_tenant_6")

        assert status["tenant_id"] == "test_tenant_6"
        assert status["tier"] == "pro"
        assert "quotas" in status
        assert "queries_hourly" in status["quotas"]


class TestFairTenantQueue:
    """Test FairTenantQueue functionality"""

    @pytest.fixture
    def redis_client(self):
        """Get Redis client or skip if unavailable"""
        try:
            r = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=2
            )
            r.ping()
            # Clean up test queue keys
            for key in r.scan_iter("queue:*"):
                r.delete(key)
            return r
        except (redis.ConnectionError, redis.TimeoutError):
            pytest.skip("Redis not available")

    @pytest.mark.asyncio
    async def test_enqueue_request(self, redis_client):
        """Can enqueue a request"""
        queue = FairTenantQueue(redis_client, max_queue_size=10)

        request = QueuedRequest(
            request_id="test_req_1",
            tenant_id="test_tenant_a",
            query="Test query",
            queued_at=time.time()
        )

        success = await queue.enqueue(request)
        assert success is True

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self, redis_client):
        """Queue rejects when full"""
        queue = FairTenantQueue(redis_client, max_queue_size=2)

        # Fill queue
        for i in range(2):
            req = QueuedRequest(
                request_id=f"req_{i}",
                tenant_id="test_tenant_b",
                query=f"Query {i}",
                queued_at=time.time()
            )
            await queue.enqueue(req)

        # Try to add one more
        overflow_req = QueuedRequest(
            request_id="overflow",
            tenant_id="test_tenant_b",
            query="Overflow",
            queued_at=time.time()
        )
        success = await queue.enqueue(overflow_req)
        assert success is False

    @pytest.mark.asyncio
    async def test_fair_dequeue(self, redis_client):
        """Dequeue uses fair scheduling"""
        queue = FairTenantQueue(redis_client, max_queue_size=10)

        # Enqueue from multiple tenants
        tenants = ["tenant_x", "tenant_y", "tenant_z"]
        for tenant in tenants:
            for i in range(2):
                req = QueuedRequest(
                    request_id=f"{tenant}_req_{i}",
                    tenant_id=tenant,
                    query=f"Query {i}",
                    queued_at=time.time()
                )
                await queue.enqueue(req)

        # Dequeue should alternate between tenants
        dequeued = []
        for _ in range(3):
            req = await queue.dequeue_fair()
            if req:
                dequeued.append(req.tenant_id)

        # Should have one from each tenant (round-robin)
        assert len(set(dequeued)) == 3

    @pytest.mark.asyncio
    async def test_queue_stats(self, redis_client):
        """Queue stats returns correct counts"""
        queue = FairTenantQueue(redis_client, max_queue_size=10)

        # Enqueue some requests
        for i in range(3):
            req = QueuedRequest(
                request_id=f"stat_req_{i}",
                tenant_id="stat_tenant",
                query=f"Query {i}",
                queued_at=time.time()
            )
            await queue.enqueue(req)

        stats = queue.get_queue_stats()
        assert stats["total_queued_requests"] == 3
        assert stats["active_tenants"] == 1


class TestResourceWeightedQuota:
    """Test resource-weighted quota calculations"""

    @pytest.fixture
    def redis_client(self):
        """Mock Redis for weight calculator"""
        return Mock()

    def test_gpt35_weight(self, redis_client):
        """GPT-3.5 has baseline weight"""
        weighted = ResourceWeightedQuota(redis_client)

        request_data = {
            "model": "gpt-3.5-turbo",
            "context": "small context",
            "use_tools": False,
            "use_embeddings": False
        }

        weight = weighted.calculate_query_weight(request_data)
        assert weight >= 1.0
        assert weight < 5.0  # Should be low

    def test_gpt4_weight_higher(self, redis_client):
        """GPT-4 has higher weight than GPT-3.5"""
        weighted = ResourceWeightedQuota(redis_client)

        gpt35_request = {
            "model": "gpt-3.5-turbo",
            "context": "context",
            "use_tools": False,
            "use_embeddings": False
        }

        gpt4_request = {
            "model": "gpt-4",
            "context": "context",
            "use_tools": False,
            "use_embeddings": False
        }

        weight_35 = weighted.calculate_query_weight(gpt35_request)
        weight_4 = weighted.calculate_query_weight(gpt4_request)

        assert weight_4 > weight_35 * 10  # GPT-4 should be 20x

    def test_tools_increase_weight(self, redis_client):
        """Using tools increases weight"""
        weighted = ResourceWeightedQuota(redis_client)

        base_request = {
            "model": "gpt-3.5-turbo",
            "context": "context",
            "use_tools": False,
            "use_embeddings": False
        }

        tools_request = {
            "model": "gpt-3.5-turbo",
            "context": "context",
            "use_tools": True,
            "use_embeddings": False
        }

        weight_base = weighted.calculate_query_weight(base_request)
        weight_tools = weighted.calculate_query_weight(tools_request)

        assert weight_tools > weight_base


class TestAtomicQuotaManager:
    """Test atomic quota operations"""

    @pytest.fixture
    def redis_client(self):
        """Get Redis client or skip if unavailable"""
        try:
            r = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=2
            )
            r.ping()
            # Clean up test keys
            for key in r.scan_iter("quota:atomic_*"):
                r.delete(key)
            for key in r.scan_iter("tenant:atomic_*"):
                r.delete(key)
            return r
        except (redis.ConnectionError, redis.TimeoutError):
            pytest.skip("Redis not available")

    def test_atomic_check_and_increment_success(self, redis_client):
        """Atomic operation succeeds when under quota"""
        qm = AtomicQuotaManager(redis_client)
        qm.set_tenant_tier("atomic_tenant_1", "free")

        success, current, limit = qm.atomic_check_and_increment(
            "atomic_tenant_1",
            QuotaType.QUERIES_HOURLY,
            increment=1
        )

        assert success is True
        assert current == 1
        assert limit == 100

    def test_atomic_check_and_increment_over_quota(self, redis_client):
        """Atomic operation fails when over quota"""
        qm = AtomicQuotaManager(redis_client)
        qm.set_tenant_tier("atomic_tenant_2", "free")

        # Fill quota
        for _ in range(100):
            qm.atomic_check_and_increment(
                "atomic_tenant_2",
                QuotaType.QUERIES_HOURLY,
                increment=1
            )

        # Next request should fail
        success, current, limit = qm.atomic_check_and_increment(
            "atomic_tenant_2",
            QuotaType.QUERIES_HOURLY,
            increment=1
        )

        assert success is False
        assert current == 100
        assert limit == 100


def test_example_data_loads():
    """Example data file is valid JSON"""
    import json
    with open("example_data.json", "r") as f:
        data = json.load(f)
        assert "tenants" in data
        assert "sample_queries" in data
        assert len(data["tenants"]) >= 3


def test_module_imports():
    """All main module components import without errors"""
    from l2_m11_resource_management_throttling import (
        QuotaManager,
        FairTenantQueue,
        ResourceWeightedQuota,
        AtomicQuotaManager,
        QueueWorker,
        QuotaType,
        TenantQuotas,
        QueuedRequest
    )
    assert QuotaManager is not None
    assert FairTenantQueue is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
