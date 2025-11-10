"""
FastAPI application for Module 11.1: Tenant Isolation Strategies

Provides REST API endpoints for tenant management, data operations, and cost tracking.
All business logic delegated to src.l3_m11_tenant_isolation_strategies.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

# Import module components
from src.l3_m11_tenant_isolation_strategies import (
    TenantRegistry,
    TenantDataManager,
    CostAllocationEngine,
    TenantTier,
    test_cross_tenant_isolation
)
from config import Config, get_clients, validate_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Tenant Isolation API",
    description="Multi-tenant SaaS isolation strategies for RAG systems",
    version="1.0.0"
)

# Initialize components
registry = TenantRegistry()
data_manager = TenantDataManager(registry)
cost_engine = CostAllocationEngine()
clients = get_clients()


# Request/Response models
class TenantCreate(BaseModel):
    tenant_id: str = Field(..., description="Unique tenant identifier")
    tenant_name: str = Field(..., description="Human-readable tenant name")
    tier: str = Field(..., description="Subscription tier: free, pro, or enterprise")


class DocumentUpsert(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier")
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to upsert")


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier (mandatory)")
    query_text: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of results to return")
    embed_tokens: int = Field(default=100, description="Estimated embedding tokens")
    llm_tokens: int = Field(default=500, description="Estimated LLM tokens")


class CostAllocation(BaseModel):
    monthly_fixed_cost: float = Field(..., description="Total monthly fixed costs")
    allocation_basis: Dict[str, float] = Field(..., description="Usage percentage per tenant")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "module": "m11_tenant_isolation",
        "config_valid": validate_config(),
        "clients": {
            "pinecone": "pinecone" in clients,
            "openai": "openai" in clients,
            "postgres": "postgres" in clients
        }
    }


# Tenant management
@app.post("/tenants", status_code=status.HTTP_201_CREATED)
async def create_tenant(tenant: TenantCreate):
    """
    Register a new tenant with tier-based configuration.

    Args:
        tenant: Tenant creation request

    Returns:
        Tenant configuration
    """
    try:
        # Parse tier
        tier_enum = TenantTier(tenant.tier.lower())

        # Register tenant
        config = registry.register_tenant(
            tenant_id=tenant.tenant_id,
            tenant_name=tenant.tenant_name,
            tier=tier_enum
        )

        return {
            "tenant_id": config.tenant_id,
            "tenant_name": config.tenant_name,
            "tier": config.tier.value,
            "namespace": config.namespace,
            "dedicated_index": config.dedicated_index,
            "quota": {
                "max_documents": config.quota.max_documents,
                "max_daily_queries": config.quota.max_daily_queries,
                "max_storage_mb": config.quota.max_storage_mb,
                "isolation_strategy": config.quota.isolation_strategy
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_507_INSUFFICIENT_STORAGE, detail=str(e))


@app.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str):
    """
    Retrieve tenant configuration.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Tenant configuration or 404
    """
    config = registry.get_tenant(tenant_id)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    return {
        "tenant_id": config.tenant_id,
        "tenant_name": config.tenant_name,
        "tier": config.tier.value,
        "namespace": config.namespace,
        "dedicated_index": config.dedicated_index
    }


# Data operations
@app.post("/upsert")
async def upsert_documents(request: DocumentUpsert):
    """
    Upsert documents with tenant isolation.

    Args:
        request: Document upsert request

    Returns:
        Upsert result or error
    """
    # Check if clients available
    if not clients.get("pinecone"):
        return {
            "skipped": True,
            "reason": "Pinecone client not available (missing API key)",
            "tenant_id": request.tenant_id,
            "document_count": len(request.documents)
        }

    try:
        result = data_manager.upsert_documents(
            tenant_id=request.tenant_id,
            documents=request.documents,
            pinecone_client=clients.get("pinecone")
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query documents with mandatory tenant scoping.

    Args:
        request: Query request with tenant_id

    Returns:
        Query results with cost tracking
    """
    # Check if clients available
    if not clients.get("pinecone") or not clients.get("openai"):
        return {
            "skipped": True,
            "reason": "Required clients not available (missing API keys)",
            "tenant_id": request.tenant_id,
            "query": request.query_text
        }

    try:
        # Generate dummy embedding (in production, use OpenAI)
        query_vector = [0.1] * 384  # Placeholder

        # Execute query
        result = data_manager.query_documents(
            tenant_id=request.tenant_id,
            query_vector=query_vector,
            top_k=request.top_k,
            pinecone_client=clients.get("pinecone")
        )

        # Track cost
        cost = cost_engine.track_query_cost(
            tenant_id=request.tenant_id,
            embed_tokens=request.embed_tokens,
            llm_tokens=request.llm_tokens
        )

        result["query_cost"] = cost
        return result

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# Cost management
@app.post("/costs/allocate")
async def allocate_costs(request: CostAllocation):
    """
    Allocate fixed infrastructure costs across tenants.

    Args:
        request: Cost allocation request

    Returns:
        Per-tenant cost allocation
    """
    try:
        allocated = cost_engine.allocate_fixed_costs(
            monthly_fixed_cost=request.monthly_fixed_cost,
            allocation_basis=request.allocation_basis
        )
        return {
            "total_fixed_cost": request.monthly_fixed_cost,
            "allocations": allocated
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/costs/{tenant_id}")
async def get_tenant_costs(tenant_id: str):
    """
    Get cost summary for tenant.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Cost summary or 404
    """
    summary = cost_engine.get_tenant_cost_summary(tenant_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No cost data for tenant"
        )
    return summary


# Security testing
@app.post("/security/test-isolation")
async def test_isolation(tenant_a_id: str, tenant_b_id: str):
    """
    Test cross-tenant isolation (security test).

    Args:
        tenant_a_id: First tenant
        tenant_b_id: Second tenant

    Returns:
        Isolation test results
    """
    passed = test_cross_tenant_isolation(data_manager, tenant_a_id, tenant_b_id)
    return {
        "test": "cross_tenant_isolation",
        "tenant_a": tenant_a_id,
        "tenant_b": tenant_b_id,
        "passed": passed,
        "description": "Verifies tenant B cannot access tenant A's data"
    }


# Metrics endpoint (optional)
@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        Basic metrics in text format
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return generate_latest()
    except ImportError:
        return {
            "info": "Prometheus client not installed",
            "total_tenants": len(registry.tenants),
            "namespace_usage": registry.namespace_usage
        }


# Local development server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Tenant Isolation API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
