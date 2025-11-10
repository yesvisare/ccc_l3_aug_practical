"""
FastAPI entrypoint for Module 13: Enterprise RAG SaaS

Provides HTTP API for multi-tenant RAG operations:
- GET /health - Health check
- POST /query - Execute RAG query
- POST /ingest - Ingest documents for tenant
- GET /metrics/{tenant_id} - Retrieve tenant metrics

No business logic in this file - delegates to l3_m13_complete_saas_build.py
"""

import logging
import os
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import core functionality
from src.l3_m13_complete_saas_build import (
    ComplianceCopilotSaaS,
    ConfigManager,
    UsageTracker,
    ModelTier,
    RetrievalMode
)
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG query."""
    tenant_id: str = Field(..., description="Tenant identifier")
    query: str = Field(..., description="User query text")
    model_tier: Optional[str] = Field(None, description="Override model tier (gpt-3.5-turbo, gpt-4)")
    retrieval_mode: Optional[str] = Field(None, description="Override retrieval mode (basic, hybrid, agentic)")
    max_tokens: Optional[int] = Field(None, description="Max tokens for response")

    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "acme_corp",
                "query": "What are our data encryption requirements?",
                "model_tier": "gpt-4"
            }
        }


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    tenant_id: str = Field(..., description="Tenant identifier")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to ingest")

    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "acme_corp",
                "documents": [
                    {
                        "text": "ACME Data Protection Policy...",
                        "metadata": {"source": "internal_policy", "category": "data_protection"}
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    services: Dict[str, bool]


# ============================================================================
# Application Lifecycle
# ============================================================================

# Global copilot instance
copilot: Optional[ComplianceCopilotSaaS] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize/cleanup on startup/shutdown."""
    global copilot

    # Startup
    logger.info("Initializing ComplianceCopilotSaaS...")
    try:
        copilot = ComplianceCopilotSaaS(
            config_manager=ConfigManager(),
            usage_tracker=UsageTracker()
        )
        logger.info("✓ Copilot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize copilot: {e}")
        copilot = None

    yield

    # Shutdown
    logger.info("Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Module 13: Enterprise RAG SaaS",
    description="Multi-tenant Compliance Copilot API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns system status and service availability.
    """
    service_status = Config.validate()

    return {
        "status": "ok" if copilot is not None else "degraded",
        "environment": Config.ENVIRONMENT,
        "services": service_status
    }


@app.post("/query")
async def query(request: QueryRequest) -> Dict[str, Any]:
    """
    Execute RAG query for a tenant.

    If API keys are missing, returns 200 with skipped=true (graceful degradation).
    """
    if copilot is None:
        return {
            "skipped": True,
            "reason": "Service not initialized (check API keys/configuration)"
        }

    try:
        # Build override config
        override_config = {}
        if request.model_tier:
            try:
                override_config["model_tier"] = ModelTier(request.model_tier)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid model_tier: {request.model_tier}"
                )

        if request.retrieval_mode:
            try:
                override_config["retrieval_mode"] = RetrievalMode(request.retrieval_mode)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid retrieval_mode: {request.retrieval_mode}"
                )

        # Execute query
        response = await copilot.query(
            tenant_id=request.tenant_id,
            query_text=request.query,
            override_config=override_config if override_config else None
        )

        return response

    except Exception as e:
        logger.error(f"Query failed for tenant {request.tenant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/ingest")
async def ingest(request: IngestRequest) -> Dict[str, Any]:
    """
    Ingest documents for a tenant.

    If API keys are missing, returns 200 with skipped=true.
    """
    if copilot is None:
        return {
            "skipped": True,
            "reason": "Service not initialized (check API keys/configuration)"
        }

    try:
        result = await copilot.ingest_documents(
            tenant_id=request.tenant_id,
            documents=request.documents
        )

        return result

    except Exception as e:
        logger.error(f"Ingestion failed for tenant {request.tenant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/metrics/{tenant_id}")
async def get_metrics(tenant_id: str, hours: int = 24) -> Dict[str, Any]:
    """
    Retrieve usage metrics and costs for a tenant.

    Args:
        tenant_id: Tenant identifier
        hours: Time window (default: 24)
    """
    if copilot is None:
        return {
            "skipped": True,
            "reason": "Service not initialized"
        }

    try:
        metrics = copilot.get_tenant_metrics(tenant_id, hours=hours)
        return metrics

    except Exception as e:
        logger.error(f"Metrics retrieval failed for tenant {tenant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Module 13: Enterprise RAG SaaS",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "ingest": "POST /ingest",
            "metrics": "GET /metrics/{tenant_id}"
        },
        "docs": "/docs"
    }


# ============================================================================
# Prometheus Metrics (Optional)
# ============================================================================

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response

    # Define metrics
    query_counter = Counter(
        'rag_queries_total',
        'Total RAG queries',
        ['tenant_id', 'status']
    )

    query_latency = Histogram(
        'rag_query_latency_seconds',
        'RAG query latency',
        ['tenant_id']
    )

    @app.get("/metrics")
    async def prometheus_metrics():
        """Expose Prometheus metrics."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    logger.info("✓ Prometheus metrics enabled at /metrics")

except ImportError:
    logger.warning("prometheus-client not installed, /metrics endpoint disabled")


# ============================================================================
# Local Development Runner
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get settings from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Debug mode: {Config.DEBUG}")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL.lower()
    )
