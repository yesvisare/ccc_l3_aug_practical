"""
FastAPI application for Module 11.2: Tenant-Specific Customization
Provides REST API for tenant configuration management and RAG queries.

Endpoints:
- GET /health - Health check
- GET /config/{tenant_id} - Get tenant configuration
- POST /config/{tenant_id} - Update tenant configuration
- DELETE /config/{tenant_id} - Delete tenant configuration
- GET /tenants - List all tenants
- POST /query - Execute RAG query with tenant config
- GET /metrics - Prometheus metrics (optional)
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config, get_clients
from l2_m11_tenant_specific_customization import (
    TenantConfig,
    TenantConfigRepository,
    simulate_rag_query,
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Module 11.2: Tenant-Specific Customization",
    description="Multi-tenant configuration management for RAG pipelines",
    version="1.0.0",
)

# Initialize repository
db_engine, redis_client = get_clients()
repository = TenantConfigRepository(db_engine=db_engine, redis_client=redis_client)

# Track if services are available
services_available = {
    "database": db_engine is not None,
    "redis": redis_client is not None,
}


# ==============================================================================
# Request/Response Models
# ==============================================================================


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    tenant_id: str
    query: str


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""

    config: Dict[str, Any]
    merge: bool = True


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    services: Dict[str, bool]
    environment: str


# ==============================================================================
# Middleware
# ==============================================================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
    )
    return response


# ==============================================================================
# Endpoints
# ==============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service availability status.
    """
    return HealthResponse(
        status="ok",
        services=services_available,
        environment=Config.ENVIRONMENT,
    )


@app.get("/config/{tenant_id}")
async def get_config(tenant_id: str) -> Dict[str, Any]:
    """
    Get tenant configuration.

    Args:
        tenant_id: Unique tenant identifier

    Returns:
        Tenant configuration dictionary

    If keys/services missing, returns default configuration.
    """
    try:
        config = repository.get_config(tenant_id)
        return config.model_dump()
    except Exception as e:
        logger.error(f"Error getting config for {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/{tenant_id}")
async def update_config(tenant_id: str, request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update tenant configuration.

    Args:
        tenant_id: Unique tenant identifier
        request: Configuration updates and merge flag

    Returns:
        Updated configuration

    Raises:
        400 if validation fails
        500 on internal errors
    """
    try:
        updated_config = repository.update_config(
            tenant_id, request.config, merge=request.merge
        )
        return updated_config.model_dump()
    except ValueError as e:
        logger.warning(f"Validation error for {tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating config for {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/config/{tenant_id}")
async def delete_config(tenant_id: str) -> Dict[str, Any]:
    """
    Delete tenant configuration.

    Args:
        tenant_id: Unique tenant identifier

    Returns:
        Deletion status

    Returns 404 if tenant not found.
    """
    try:
        existed = repository.delete_config(tenant_id)
        if not existed:
            raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")
        return {"status": "deleted", "tenant_id": tenant_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting config for {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tenants")
async def list_tenants() -> Dict[str, Any]:
    """
    List all configured tenants.

    Returns:
        List of tenant IDs
    """
    try:
        tenants = repository.list_tenants()
        return {"tenants": tenants, "count": len(tenants)}
    except Exception as e:
        logger.error(f"Error listing tenants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def execute_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Execute RAG query with tenant-specific configuration.

    Args:
        request: Query request with tenant_id and query text

    Returns:
        Simulated RAG response with applied configuration

    Note: If API keys are missing, returns simulated response with warning.
    """
    try:
        # Check if we have API keys
        has_keys = Config.OPENAI_API_KEY or Config.ANTHROPIC_API_KEY

        if not has_keys:
            logger.warning("⚠️ Skipping API calls (no keys/service)")
            # Return mock response
            config = repository.get_config(request.tenant_id)
            return {
                "skipped": True,
                "reason": "no API keys configured",
                "tenant_id": request.tenant_id,
                "query": request.query,
                "config_applied": {
                    "model": config.model,
                    "temperature": config.temperature,
                    "top_k": config.top_k,
                },
                "answer": "[Mock response - configure API keys for real queries]",
            }

        # Execute simulated query
        result = simulate_rag_query(request.tenant_id, request.query, repository)
        return result

    except Exception as e:
        logger.error(f"Error executing query for {request.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """
    Prometheus metrics endpoint (optional).

    Returns basic metrics if prometheus-client is available.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        metrics_data = generate_latest()
        return JSONResponse(content=metrics_data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return JSONResponse(
            content={"error": "prometheus-client not installed"},
            status_code=501,
        )


# ==============================================================================
# Startup/Shutdown Events
# ==============================================================================


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 80)
    logger.info("Module 11.2: Tenant-Specific Customization - Starting")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Database: {'available' if services_available['database'] else 'unavailable (using memory)'}")
    logger.info(f"Redis: {'available' if services_available['redis'] else 'unavailable (no caching)'}")
    logger.info(f"API Keys: {'configured' if Config.OPENAI_API_KEY or Config.ANTHROPIC_API_KEY else 'not configured'}")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")


# ==============================================================================
# Local Development Runner
# ==============================================================================


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("Starting FastAPI server for Module 11.2")
    print("Docs available at: http://localhost:8000/docs")
    print("=" * 80)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=Config.LOG_LEVEL.lower(),
    )
