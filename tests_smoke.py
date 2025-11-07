"""
Smoke tests for Module 13: Enterprise RAG SaaS

Minimal tests to verify:
1. Configuration loads correctly
2. Core functions return expected shapes
3. Network paths gracefully skip without API keys
4. No import errors
"""

import pytest
import asyncio
import json
from pathlib import Path

# Import modules to test
from l3_m13_complete_saas_build import (
    ComplianceCopilotSaaS,
    ConfigManager,
    UsageTracker,
    TenantContext,
    TenantConfig,
    ResourceLimits,
    ModelTier,
    RetrievalMode,
    UsageRecord
)
from config import Config
import app


# ============================================================================
# Configuration Tests
# ============================================================================

def test_config_loads():
    """Test that configuration loads without errors."""
    assert Config.DEFAULT_MODEL is not None
    assert Config.DEFAULT_RETRIEVAL_MODE is not None
    assert Config.MAX_QUERIES_PER_HOUR > 0


def test_config_validation():
    """Test config validation returns dict."""
    status = Config.validate()
    assert isinstance(status, dict)
    assert 'openai' in status
    assert 'pinecone' in status


def test_environment_settings():
    """Test environment settings are accessible."""
    assert Config.ENVIRONMENT in ['development', 'production', 'test']
    assert isinstance(Config.DEBUG, bool)


# ============================================================================
# Data Model Tests
# ============================================================================

def test_tenant_config_creation():
    """Test TenantConfig instantiation."""
    config = TenantConfig(
        tenant_id="test_tenant",
        model_tier=ModelTier.GPT35,
        retrieval_mode=RetrievalMode.BASIC
    )

    assert config.tenant_id == "test_tenant"
    assert config.model_tier == ModelTier.GPT35
    assert config.pinecone_namespace == "tenant_test_tenant"


def test_resource_limits_defaults():
    """Test ResourceLimits has sane defaults."""
    limits = ResourceLimits()

    assert limits.max_queries_per_hour > 0
    assert limits.max_tokens_per_query > 0
    assert limits.timeout_seconds > 0


def test_usage_record_creation():
    """Test UsageRecord instantiation."""
    from datetime import datetime

    record = UsageRecord(
        tenant_id="test",
        operation="query",
        timestamp=datetime.now(),
        tokens_used=100
    )

    assert record.tenant_id == "test"
    assert record.operation == "query"
    assert record.tokens_used == 100


# ============================================================================
# Context Management Tests
# ============================================================================

def test_tenant_context_set_get():
    """Test tenant context propagation."""
    TenantContext.clear_tenant()

    TenantContext.set_tenant("test_tenant")
    assert TenantContext.get_tenant() == "test_tenant"

    TenantContext.clear_tenant()
    assert TenantContext.get_tenant() is None


# ============================================================================
# Core Functionality Tests
# ============================================================================

@pytest.mark.asyncio
async def test_config_manager_initialization():
    """Test ConfigManager initializes."""
    mgr = ConfigManager()
    assert mgr is not None


@pytest.mark.asyncio
async def test_config_manager_load_tenant():
    """Test tenant config loading."""
    mgr = ConfigManager()
    config = await mgr.load_tenant_config("test_tenant")

    assert config.tenant_id == "test_tenant"
    assert isinstance(config.model_tier, ModelTier)
    assert isinstance(config.retrieval_mode, RetrievalMode)


@pytest.mark.asyncio
async def test_usage_tracker():
    """Test UsageTracker records operations."""
    from datetime import datetime

    tracker = UsageTracker()
    record = UsageRecord(
        tenant_id="test",
        operation="query",
        timestamp=datetime.now(),
        tokens_used=100
    )

    await tracker.track(record)

    usage = tracker.get_tenant_usage("test", hours=1)
    assert len(usage) >= 1
    assert usage[0].tenant_id == "test"


@pytest.mark.asyncio
async def test_copilot_initialization():
    """Test ComplianceCopilotSaaS initializes without errors."""
    copilot = ComplianceCopilotSaaS()

    assert copilot.config_manager is not None
    assert copilot.usage_tracker is not None
    assert copilot.vector_store is not None


@pytest.mark.asyncio
async def test_copilot_query_graceful_degradation():
    """
    Test query execution (simulated - no real API calls).
    Should complete without network errors.
    """
    copilot = ComplianceCopilotSaaS()

    # This uses simulated LLM/vector store, so it should work without API keys
    try:
        response = await copilot.query(
            tenant_id="test_tenant",
            query_text="Test query"
        )

        assert 'answer' in response
        assert 'metadata' in response
        assert response['metadata']['tenant_id'] == "test_tenant"

    except Exception as e:
        # If it fails, it should be a known error (e.g., rate limit, not network)
        assert "Rate limit" in str(e) or "quota" in str(e)


@pytest.mark.asyncio
async def test_copilot_ingestion():
    """Test document ingestion."""
    copilot = ComplianceCopilotSaaS()

    result = await copilot.ingest_documents(
        tenant_id="test_tenant",
        documents=[
            {"text": "Test document 1"},
            {"text": "Test document 2"}
        ]
    )

    assert result['tenant_id'] == "test_tenant"
    assert result['documents_ingested'] == 2
    assert result['status'] == "success"


@pytest.mark.asyncio
async def test_copilot_metrics():
    """Test metrics retrieval."""
    copilot = ComplianceCopilotSaaS()

    # Ingest and query to generate some metrics
    await copilot.ingest_documents(
        tenant_id="metrics_test",
        documents=[{"text": "Test doc"}]
    )

    await copilot.query(
        tenant_id="metrics_test",
        query_text="Test query"
    )

    metrics = copilot.get_tenant_metrics("metrics_test", hours=1)

    assert 'total_queries' in metrics
    assert 'costs' in metrics
    assert metrics['total_queries'] >= 1


# ============================================================================
# FastAPI App Tests
# ============================================================================

def test_app_initialization():
    """Test FastAPI app initializes."""
    from app import app as fastapi_app

    assert fastapi_app is not None
    assert fastapi_app.title is not None


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test /health endpoint structure."""
    from app import health_check

    response = await health_check()

    assert 'status' in response
    assert 'environment' in response
    assert 'services' in response
    assert isinstance(response['services'], dict)


# ============================================================================
# Data File Tests
# ============================================================================

def test_example_data_exists():
    """Test example_data.json exists and is valid."""
    data_file = Path("example_data.json")
    assert data_file.exists(), "example_data.json not found"

    with open(data_file) as f:
        data = json.load(f)

    assert 'tenants' in data
    assert 'sample_documents' in data
    assert 'sample_queries' in data
    assert len(data['tenants']) >= 3


def test_env_example_exists():
    """Test .env.example exists."""
    env_file = Path(".env.example")
    assert env_file.exists(), ".env.example not found"

    content = env_file.read_text()
    assert 'OPENAI_API_KEY' in content
    assert 'PINECONE_API_KEY' in content


# ============================================================================
# Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow():
    """
    End-to-end smoke test:
    1. Initialize copilot
    2. Ingest documents
    3. Execute query
    4. Check metrics
    """
    copilot = ComplianceCopilotSaaS()

    # 1. Ingest
    ingest_result = await copilot.ingest_documents(
        tenant_id="workflow_test",
        documents=[
            {"text": "Policy document 1"},
            {"text": "Policy document 2"}
        ]
    )
    assert ingest_result['status'] == "success"

    # 2. Query
    query_result = await copilot.query(
        tenant_id="workflow_test",
        query_text="What are the policies?"
    )
    assert 'answer' in query_result
    assert query_result['metadata']['tenant_id'] == "workflow_test"

    # 3. Metrics
    metrics = copilot.get_tenant_metrics("workflow_test", hours=1)
    assert metrics['total_queries'] >= 1
    assert metrics['costs']['total_tokens'] > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("Running smoke tests...")
    print("=" * 60)

    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
