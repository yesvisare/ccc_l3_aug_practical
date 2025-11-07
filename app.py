"""
FastAPI application for Module 9.1: Query Decomposition & Planning

Provides REST API endpoints for query decomposition and processing.
No business logic here - all logic is in l2_m9_query_decomposition_planning.py
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from l2_m9_query_decomposition_planning import (
    QueryDecompositionPipeline,
    DecompositionError,
    DependencyError,
    SynthesisError
)
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Query Decomposition & Planning API",
    description="Advanced retrieval through query decomposition for complex multi-part queries",
    version="1.0.0"
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(..., description="The query to process")
    complexity_threshold: Optional[float] = Field(
        0.7,
        description="Threshold for using decomposition (0-1)",
        ge=0.0,
        le=1.0
    )
    min_latency_budget_ms: Optional[int] = Field(
        700,
        description="Minimum latency budget for decomposition (ms)",
        ge=0
    )
    enable_fallback: Optional[bool] = Field(
        True,
        description="Enable fallback to simple retrieval on failure"
    )


class QueryResponse(BaseModel):
    """Response model for query processing."""
    answer: str
    method: str
    latency_ms: float
    sub_queries: Optional[int] = None
    execution_levels: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    config_valid: bool
    api_key_present: bool


# Mock retrieval function (replace with actual retrieval in production)
async def mock_retrieval_function(query: str) -> str:
    """
    Mock retrieval function for demonstration.

    In production, replace this with your actual vector search / retrieval logic.
    """
    await asyncio.sleep(0.1)  # Simulate retrieval latency
    return f"Mock retrieval result for query: {query[:100]}..."


# Initialize pipeline (global, lazy initialization)
_pipeline: Optional[QueryDecompositionPipeline] = None


def get_pipeline() -> Optional[QueryDecompositionPipeline]:
    """Get or initialize the query decomposition pipeline."""
    global _pipeline

    if _pipeline is None:
        if not Config.OPENAI_API_KEY:
            logger.warning("No OPENAI_API_KEY configured")
            return None

        _pipeline = QueryDecompositionPipeline(
            api_key=Config.OPENAI_API_KEY,
            retrieval_function=mock_retrieval_function,
            model=Config.DECOMPOSITION_MODEL,
            max_concurrent=Config.MAX_CONCURRENT_RETRIEVALS,
            enable_fallback=Config.ENABLE_FALLBACK
        )
        logger.info("Query decomposition pipeline initialized")

    return _pipeline


# Optional: Prometheus metrics support
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    # Metrics
    query_counter = Counter(
        'query_decomposition_requests_total',
        'Total query requests',
        ['method', 'status']
    )
    query_latency = Histogram(
        'query_decomposition_latency_seconds',
        'Query processing latency',
        ['method']
    )

    METRICS_ENABLED = Config.ENABLE_METRICS
except ImportError:
    METRICS_ENABLED = False
    logger.info("Prometheus metrics not available (prometheus-client not installed)")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and configuration validity.
    """
    return HealthResponse(
        status="ok",
        config_valid=Config.validate(),
        api_key_present=bool(Config.OPENAI_API_KEY)
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process a query with decomposition if beneficial.

    If API key is missing, returns 200 with skipped=true.
    """
    # Check if pipeline is available
    pipeline = get_pipeline()

    if pipeline is None:
        logger.warning("Pipeline not available (no API key)")
        if METRICS_ENABLED:
            query_counter.labels(method="skipped", status="no_key").inc()

        return QueryResponse(
            answer="Service not configured - missing API key",
            method="skipped",
            latency_ms=0,
            error="No OPENAI_API_KEY configured"
        )

    # Process query
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        result = await pipeline.process_query(
            query=request.query,
            complexity_threshold=request.complexity_threshold,
            min_latency_budget_ms=request.min_latency_budget_ms
        )

        if METRICS_ENABLED:
            query_counter.labels(method=result['method'], status="success").inc()
            query_latency.labels(method=result['method']).observe(result['latency_ms'] / 1000)

        return QueryResponse(
            answer=result['answer'],
            method=result['method'],
            latency_ms=result['latency_ms'],
            sub_queries=result.get('sub_queries'),
            execution_levels=result.get('execution_levels'),
            error=result.get('error')
        )

    except (DecompositionError, DependencyError, SynthesisError) as e:
        logger.error(f"Query processing error: {e}")
        if METRICS_ENABLED:
            query_counter.labels(method="error", status="processing_error").inc()

        # If fallback is enabled, this shouldn't happen (pipeline handles it)
        # But catch it just in case
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if METRICS_ENABLED:
            query_counter.labels(method="error", status="unexpected").inc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if METRICS_ENABLED:
    @app.get("/metrics")
    async def metrics():
        """
        Prometheus metrics endpoint.

        Only available if prometheus-client is installed and ENABLE_METRICS=true.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Query Decomposition & Planning API",
        "version": "1.0.0",
        "module": "9.1",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "metrics": "/metrics (if enabled)",
            "docs": "/docs"
        },
        "status": "ok" if Config.validate() else "not_configured"
    }


# Uvicorn runner for local development
if __name__ == "__main__":
    import sys

    # Display startup info
    print("=" * 60)
    print("Query Decomposition & Planning API")
    print("=" * 60)
    print(f"Config Valid: {Config.validate()}")
    print(f"API Key Present: {bool(Config.OPENAI_API_KEY)}")
    print(f"Metrics Enabled: {METRICS_ENABLED}")
    print("=" * 60)

    if not Config.OPENAI_API_KEY:
        print("⚠️  WARNING: No OPENAI_API_KEY configured")
        print("    API will return 'skipped' responses")
        print("    Set OPENAI_API_KEY in .env to enable full functionality")
        print("=" * 60)

    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
