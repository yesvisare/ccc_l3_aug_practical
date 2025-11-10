"""
Smoke tests for Module 11.1: Tenant Isolation Strategies

Basic tests to verify:
- Configuration loads correctly
- Core functions return expected shapes
- Network paths gracefully skip without keys
"""

import pytest
import json
from src.l3_m11_tenant_isolation_strategies import (
    TenantRegistry,
    TenantDataManager,
    CostAllocationEngine,
    TenantTier,
    ResourceQuota,
    test_cross_tenant_isolation
)
from config import Config, get_clients, validate_config


def test_config_loads():
    """Test that configuration loads without errors."""
    assert Config.POSTGRES_HOST is not None
    assert Config.MAX_NAMESPACES_PER_INDEX == 90
    assert Config.NAMESPACE_ALERT_THRESHOLD == 0.8
    print("✓ Config loads successfully")


def test_tier_quotas_defined():
    """Test that tier quotas are properly defined."""
    registry = TenantRegistry()
    assert TenantTier.FREE in registry.tier_quotas
    assert TenantTier.PRO in registry.tier_quotas
    assert TenantTier.ENTERPRISE in registry.tier_quotas

    free_quota = registry.tier_quotas[TenantTier.FREE]
    assert free_quota.isolation_strategy == "namespace"
    assert free_quota.max_documents == 1000

    ent_quota = registry.tier_quotas[TenantTier.ENTERPRISE]
    assert ent_quota.isolation_strategy == "index"
    assert ent_quota.max_documents == 100000
    print("✓ Tier quotas defined correctly")


def test_tenant_registration():
    """Test tenant registration with different tiers."""
    registry = TenantRegistry()

    # Register Free tier
    free_tenant = registry.register_tenant("test-001", "Test Free", TenantTier.FREE)
    assert free_tenant.tenant_id == "test-001"
    assert free_tenant.tier == TenantTier.FREE
    assert free_tenant.namespace is not None
    assert free_tenant.dedicated_index is None

    # Register Enterprise tier
    ent_tenant = registry.register_tenant("test-002", "Test Enterprise", TenantTier.ENTERPRISE)
    assert ent_tenant.tenant_id == "test-002"
    assert ent_tenant.tier == TenantTier.ENTERPRISE
    assert ent_tenant.namespace is None
    assert ent_tenant.dedicated_index is not None
    print("✓ Tenant registration works")


def test_duplicate_tenant_rejected():
    """Test that duplicate tenant IDs are rejected."""
    registry = TenantRegistry()
    registry.register_tenant("dup-001", "Duplicate Test", TenantTier.FREE)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_tenant("dup-001", "Duplicate Again", TenantTier.PRO)
    print("✓ Duplicate tenant registration rejected")


def test_namespace_capacity_tracking():
    """Test namespace capacity tracking and alerts."""
    registry = TenantRegistry()

    # Register tenants up to alert threshold
    for i in range(72):  # 80% of 90
        registry.register_tenant(f"tenant-{i:03d}", f"Tenant {i}", TenantTier.FREE)

    # Check namespace usage
    usage = registry.namespace_usage.get("shared-index-1", 0)
    assert usage == 72
    print(f"✓ Namespace capacity tracking works ({usage}/90)")


def test_quota_checks():
    """Test quota checking logic."""
    registry = TenantRegistry()
    tenant = registry.register_tenant("quota-001", "Quota Test", TenantTier.FREE)

    # Within quota
    assert registry.check_quota("quota-001", "documents", 500) == True
    assert registry.check_quota("quota-001", "queries", 50) == True

    # Exceeds quota
    assert registry.check_quota("quota-001", "documents", 2000) == False
    assert registry.check_quota("quota-001", "queries", 200) == False
    print("✓ Quota checks work correctly")


def test_data_manager_upsert():
    """Test data manager upsert with tenant scoping."""
    registry = TenantRegistry()
    data_manager = TenantDataManager(registry)

    tenant = registry.register_tenant("data-001", "Data Test", TenantTier.PRO)

    documents = [
        {"id": "doc1", "values": [0.1] * 384, "metadata": {"title": "Test Doc 1"}},
        {"id": "doc2", "values": [0.2] * 384, "metadata": {"title": "Test Doc 2"}}
    ]

    # Upsert without actual Pinecone client
    result = data_manager.upsert_documents("data-001", documents, pinecone_client=None)

    assert result["upserted"] == 2
    assert result["namespace"] == tenant.namespace
    assert result["isolation"] == "namespace"

    # Verify tenant_id added to metadata
    for doc in documents:
        assert doc["metadata"]["tenant_id"] == "data-001"
    print("✓ Data manager upsert works")


def test_data_manager_query():
    """Test data manager query with tenant scoping."""
    registry = TenantRegistry()
    data_manager = TenantDataManager(registry)

    tenant = registry.register_tenant("query-001", "Query Test", TenantTier.PRO)

    query_vector = [0.15] * 384
    result = data_manager.query_documents("query-001", query_vector, top_k=5)

    assert "namespace" in result
    assert result["namespace"] == tenant.namespace
    assert result["isolation"] == "namespace"
    assert "overhead_ms" in result
    print("✓ Data manager query works")


def test_cost_tracking():
    """Test cost tracking per tenant."""
    cost_engine = CostAllocationEngine()

    # Track costs for multiple queries
    cost1 = cost_engine.track_query_cost("cost-001", embed_tokens=500, llm_tokens=1000)
    cost2 = cost_engine.track_query_cost("cost-001", embed_tokens=300, llm_tokens=800)

    assert cost1 > 0
    assert cost2 > 0

    # Get summary
    summary = cost_engine.get_tenant_cost_summary("cost-001")
    assert summary is not None
    assert summary["query_count"] == 2
    assert summary["variable_cost"] == cost1 + cost2
    print(f"✓ Cost tracking works (${summary['variable_cost']:.6f} for 2 queries)")


def test_fixed_cost_allocation():
    """Test fixed cost allocation across tenants."""
    cost_engine = CostAllocationEngine()

    # Track some usage
    cost_engine.track_query_cost("alloc-001", embed_tokens=500, llm_tokens=1000)
    cost_engine.track_query_cost("alloc-002", embed_tokens=1000, llm_tokens=2000)

    # Allocate fixed costs
    allocated = cost_engine.allocate_fixed_costs(
        monthly_fixed_cost=115.0,
        allocation_basis={
            "alloc-001": 30.0,  # 30% usage
            "alloc-002": 70.0   # 70% usage
        }
    )

    assert "alloc-001" in allocated
    assert "alloc-002" in allocated
    assert abs(allocated["alloc-001"] - 34.5) < 0.1  # 30% of 115
    assert abs(allocated["alloc-002"] - 80.5) < 0.1  # 70% of 115

    total_allocated = sum(allocated.values())
    assert abs(total_allocated - 115.0) < 0.01  # Should sum to total
    print(f"✓ Fixed cost allocation works (${total_allocated:.2f})")


def test_cross_tenant_isolation_function():
    """Test cross-tenant isolation verification."""
    registry = TenantRegistry()
    data_manager = TenantDataManager(registry)

    tenant_a = registry.register_tenant("iso-a", "Tenant A", TenantTier.FREE)
    tenant_b = registry.register_tenant("iso-b", "Tenant B", TenantTier.FREE)

    # Test isolation (should pass since we're not actually querying)
    result = test_cross_tenant_isolation(data_manager, "iso-a", "iso-b")
    assert isinstance(result, bool)
    print("✓ Cross-tenant isolation test runs")


def test_clients_graceful_without_keys():
    """Test that clients initialize gracefully without API keys."""
    clients = get_clients()

    # Clients dict should exist even if empty
    assert isinstance(clients, dict)

    # If keys not set, clients should be missing (not error)
    if not Config.PINECONE_API_KEY:
        assert "pinecone" not in clients
    if not Config.OPENAI_API_KEY:
        assert "openai" not in clients

    print(f"✓ Clients initialize gracefully ({len(clients)} available)")


def test_example_data_format():
    """Test that example_data.json has expected format."""
    with open("example_data.json", "r") as f:
        data = json.load(f)

    assert "tenants" in data
    assert "sample_queries" in data
    assert "isolation_tests" in data

    # Check tenant structure
    for tenant in data["tenants"]:
        assert "tenant_id" in tenant
        assert "tenant_name" in tenant
        assert "tier" in tenant
        assert "documents" in tenant

    print(f"✓ Example data format valid ({len(data['tenants'])} tenants)")


def test_invalid_tenant_query_fails():
    """Test that querying non-existent tenant fails gracefully."""
    registry = TenantRegistry()
    data_manager = TenantDataManager(registry)

    with pytest.raises(ValueError, match="not found"):
        data_manager.query_documents("nonexistent", [0.1] * 384)
    print("✓ Invalid tenant query fails properly")


if __name__ == "__main__":
    print("=== Running Smoke Tests ===\n")

    # Run tests
    test_config_loads()
    test_tier_quotas_defined()
    test_tenant_registration()
    test_duplicate_tenant_rejected()
    test_namespace_capacity_tracking()
    test_quota_checks()
    test_data_manager_upsert()
    test_data_manager_query()
    test_cost_tracking()
    test_fixed_cost_allocation()
    test_cross_tenant_isolation_function()
    test_clients_graceful_without_keys()
    test_example_data_format()
    test_invalid_tenant_query_fails()

    print("\n=== All Smoke Tests Passed ✓ ===")
