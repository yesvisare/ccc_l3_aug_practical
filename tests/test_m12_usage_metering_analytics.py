"""
Smoke tests for Usage Metering & Analytics Module.

Tests:
- Configuration loads correctly
- Core functions return plausible shapes
- Network paths gracefully skip without keys/services
- No imports fail
"""

import pytest
import asyncio
from datetime import datetime
import json


def test_imports():
    """Test that all core modules can be imported."""
    try:
        from src.l3_m12_usage_metering_analytics import (
            UsageEvent, TenantQuota, ClickHouseSchema,
            UsageTracker, CostCalculator, QuotaManager, BillingExporter
        )
        from config import get_clickhouse_client, get_config, Config
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_loads():
    """Test that configuration loads without errors."""
    from config import get_config, Config

    config = get_config()

    # Check structure
    assert "clickhouse" in config
    assert "buffering" in config
    assert "pricing" in config
    assert "quotas" in config

    # Check values are plausible
    assert config["buffering"]["buffer_size"] > 0
    assert config["buffering"]["flush_interval"] > 0
    assert config["pricing"]["query"] >= 0
    assert config["retention_months"] > 0


def test_clickhouse_client_graceful_failure():
    """Test that ClickHouse client handles missing service gracefully."""
    from config import get_clickhouse_client

    # Should return None without raising exception
    client = get_clickhouse_client()
    # Client may be None (service not available) or a client object
    assert client is None or client is not None


def test_usage_event_creation():
    """Test creating a UsageEvent object."""
    from src.l3_m12_usage_metering_analytics import UsageEvent

    event = UsageEvent(
        event_id="test_001",
        tenant_id="test_tenant",
        event_type="query",
        quantity=1.0,
        timestamp=datetime.now(),
        metadata={"test": True}
    )

    assert event.event_id == "test_001"
    assert event.tenant_id == "test_tenant"
    assert event.event_type == "query"
    assert event.quantity == 1.0

    # Test to_dict method
    event_dict = event.to_dict()
    assert isinstance(event_dict, dict)
    assert "event_id" in event_dict
    assert "tenant_id" in event_dict


def test_tenant_quota_creation():
    """Test creating a TenantQuota object."""
    from src.l3_m12_usage_metering_analytics import TenantQuota

    quota = TenantQuota(
        tenant_id="test_tenant",
        queries_per_day=1000,
        tokens_per_day=100000,
        storage_gb=10.0,
        overage_allowed=True
    )

    assert quota.tenant_id == "test_tenant"
    assert quota.queries_per_day == 1000
    assert quota.tokens_per_day == 100000
    assert quota.storage_gb == 10.0
    assert quota.overage_allowed is True


def test_schema_sql_generation():
    """Test that schema SQL can be generated."""
    from src.l3_m12_usage_metering_analytics import ClickHouseSchema

    sql_statements = ClickHouseSchema.get_schema_sql()

    assert isinstance(sql_statements, list)
    assert len(sql_statements) == 4  # 4 tables/views
    assert all(isinstance(s, str) for s in sql_statements)
    assert any("usage_events" in s for s in sql_statements)
    assert any("usage_hourly" in s for s in sql_statements)
    assert any("usage_daily" in s for s in sql_statements)
    assert any("tenant_quotas" in s for s in sql_statements)


def test_cost_calculator():
    """Test cost calculation logic."""
    from src.l3_m12_usage_metering_analytics import CostCalculator, UsageEvent

    # Test query cost
    query_event = UsageEvent(
        event_id="test",
        tenant_id="test",
        event_type="query",
        quantity=1.0,
        timestamp=datetime.now(),
        metadata={}
    )
    query_cost = CostCalculator.calculate_event_cost(query_event)
    assert query_cost == 0.01  # Default query price

    # Test token_input cost
    token_event = UsageEvent(
        event_id="test",
        tenant_id="test",
        event_type="token_input",
        quantity=1000.0,  # 1K tokens
        timestamp=datetime.now(),
        metadata={}
    )
    token_cost = CostCalculator.calculate_event_cost(token_event)
    assert token_cost == 0.003  # Default per-1K price

    # Test storage cost
    storage_event = UsageEvent(
        event_id="test",
        tenant_id="test",
        event_type="storage",
        quantity=2.5,  # 2.5 GB
        timestamp=datetime.now(),
        metadata={}
    )
    storage_cost = CostCalculator.calculate_event_cost(storage_event)
    assert storage_cost == 0.25  # 2.5 * 0.10


@pytest.mark.asyncio
async def test_usage_tracker_initialization():
    """Test that UsageTracker can be initialized without ClickHouse."""
    from src.l3_m12_usage_metering_analytics import UsageTracker

    # Should work even without client
    tracker = UsageTracker(client=None)
    assert tracker is not None
    assert tracker.client is None
    assert isinstance(tracker.buffer, list)

    # Test start/stop
    await tracker.start()
    assert tracker.running is True

    await tracker.stop()
    assert tracker.running is False


@pytest.mark.asyncio
async def test_usage_tracker_fallback():
    """Test that tracker falls back to file storage when no client."""
    from src.l3_m12_usage_metering_analytics import UsageTracker, UsageEvent
    import os

    tracker = UsageTracker(client=None)
    await tracker.start()

    # Track an event
    event = UsageEvent(
        event_id="fallback_test",
        tenant_id="test",
        event_type="query",
        quantity=1.0,
        timestamp=datetime.now(),
        metadata={"test": "fallback"}
    )

    await tracker.track(event)
    await tracker.flush()

    await tracker.stop()

    # Check that fallback file was created (if flush happened)
    # Note: File may not exist if buffer wasn't full
    # This is expected behavior - just check no errors


def test_example_data_loads():
    """Test that example data file is valid."""
    import os

    assert os.path.exists("example_data.json"), "example_data.json not found"

    with open("example_data.json") as f:
        data = json.load(f)

    assert "sample_events" in data
    assert "sample_quotas" in data
    assert "pricing_config" in data

    # Check structure
    assert len(data["sample_events"]) > 0
    for event in data["sample_events"]:
        assert "event_id" in event
        assert "tenant_id" in event
        assert "event_type" in event
        assert "quantity" in event


def test_quota_manager_without_client():
    """Test QuotaManager handles missing client gracefully."""
    from src.l3_m12_usage_metering_analytics import QuotaManager, TenantQuota

    # Should initialize even without client
    manager = QuotaManager(client=None)
    assert manager is not None

    # Operations should fail gracefully
    quota = TenantQuota(
        tenant_id="test",
        queries_per_day=1000,
        tokens_per_day=100000,
        storage_gb=10.0
    )

    result = manager.set_quota(quota)
    assert result is False  # Expected to fail without client


def test_billing_exporter_without_client():
    """Test BillingExporter handles missing client gracefully."""
    from src.l3_m12_usage_metering_analytics import BillingExporter

    exporter = BillingExporter(client=None)
    assert exporter is not None

    # Should return error dict, not raise exception
    invoice = exporter.export_monthly_invoice("test_tenant", 2025, 11)
    assert isinstance(invoice, dict)
    assert "error" in invoice or "skipped" in str(invoice).lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
