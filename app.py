"""
FastAPI wrapper for Multi-Agent Orchestration module.
Provides REST API endpoints for multi-agent query processing.

Endpoints:
- GET /health - Health check
- POST /query - Execute multi-agent query
- GET /route - Check routing recommendation
- GET /metrics - Prometheus metrics (if available)
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time
import logging

# Import core functionality - NO business logic in this file
from l2_m10_multi_agent_orchestration import (
    run_multi_agent_query,
    should_use_multi_agent
)
from config import Config, logger

# Optional: Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True

    # Metrics
    query_counter = Counter('multiagent_queries_total', 'Total multi-agent queries', ['status'])
    query_duration = Histogram('multiagent_query_duration_seconds', 'Query duration')
    query_cost = Histogram('multiagent_query_cost_usd', 'Query cost in USD')
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus-client not available, metrics disabled")


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Orchestration API",
    description="Module 10.3: Three-agent system (Planner, Executor, Validator) for complex queries",
    version="1.0.0"
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str = Field(..., description="User query to process", min_length=1)
    force_multi_agent: bool = Field(
        default=False,
        description="Force multi-agent even if routing suggests single-agent"
    )
    max_iterations: Optional[int] = Field(
        default=None,
        description="Override default max iterations",
        ge=1,
        le=5
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    success: bool
    query: str
    results: list
    validation_status: str
    metadata: Dict[str, Any]
    messages: list
    routing_recommendation: Optional[Dict[str, Any]] = None


class RouteRequest(BaseModel):
    """Request model for /route endpoint."""
    query: str = Field(..., description="Query to evaluate for routing")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    configured: bool
    config: Dict[str, Any]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service status and configuration info
    """
    return HealthResponse(
        status="ok",
        configured=Config.is_configured(),
        config=Config.get_config_dict()
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Execute a multi-agent query.

    If API keys are missing, returns success=True with skipped=True.
    This allows the service to run without external dependencies.

    Args:
        request: Query request with query text and options

    Returns:
        Query results with metadata
    """
    logger.info(f"Received query: {request.query[:50]}...")

    # Check if configured
    if not Config.is_configured():
        logger.warning("API keys not configured, skipping execution")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "skipped": True,
                "reason": "OPENAI_API_KEY not configured",
                "query": request.query,
                "results": [],
                "validation_status": "skipped",
                "metadata": {
                    "total_time_seconds": 0,
                    "iterations": 0,
                    "estimated_cost_usd": 0.0
                },
                "messages": ["Skipped: No API keys configured"]
            }
        )

    # Get routing recommendation
    routing = should_use_multi_agent(request.query)
    logger.info(f"Routing: {routing['recommendation']} - {routing['reason']}")

    # Check if multi-agent is appropriate (unless forced)
    if routing['recommendation'] == 'single-agent' and not request.force_multi_agent:
        logger.info("Query better suited for single-agent, returning recommendation")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "query": request.query,
                "results": [],
                "validation_status": "not_executed",
                "metadata": {
                    "total_time_seconds": 0,
                    "iterations": 0,
                    "estimated_cost_usd": 0.0
                },
                "messages": [
                    f"Routing: {routing['recommendation']}",
                    f"Reason: {routing['reason']}",
                    routing.get('warning', '')
                ],
                "routing_recommendation": routing
            }
        )

    # Override max iterations if provided
    if request.max_iterations:
        original_max = Config.MAX_ITERATIONS
        Config.MAX_ITERATIONS = request.max_iterations
        logger.info(f"Overriding max_iterations: {original_max} → {request.max_iterations}")

    # Execute multi-agent query
    start_time = time.time()

    try:
        result = run_multi_agent_query(request.query)

        # Track metrics if available
        if PROMETHEUS_AVAILABLE:
            query_duration.observe(time.time() - start_time)
            query_cost.observe(result['metadata'].get('estimated_cost_usd', 0))
            query_counter.labels(
                status='success' if result['success'] else 'failure'
            ).inc()

        # Add routing info
        result['routing_recommendation'] = routing

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)

        if PROMETHEUS_AVAILABLE:
            query_counter.labels(status='error').inc()

        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )

    finally:
        # Restore original max iterations
        if request.max_iterations:
            Config.MAX_ITERATIONS = original_max


@app.post("/route")
async def check_route(request: RouteRequest):
    """
    Check routing recommendation for a query without executing it.

    Args:
        request: Query to evaluate

    Returns:
        Routing recommendation with reasoning
    """
    routing = should_use_multi_agent(request.query)
    return routing


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint (optional).

    Returns:
        Metrics in Prometheus format
    """
    if not PROMETHEUS_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Metrics not available. Install: pip install prometheus-client"}
        )

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": "Internal server error"
        }
    )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("="*70)
    logger.info("Multi-Agent Orchestration API starting...")
    logger.info(f"Model: {Config.OPENAI_MODEL}")
    logger.info(f"Max iterations: {Config.MAX_ITERATIONS}")
    logger.info(f"Configured: {Config.is_configured()}")
    logger.info(f"Metrics: {'enabled' if PROMETHEUS_AVAILABLE else 'disabled'}")
    logger.info("="*70)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("Multi-Agent Orchestration API shutting down...")


# ============================================================================
# UVICORN RUNNER (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("Starting Multi-Agent Orchestration API")
    print("="*70)
    print(f"Configured: {Config.is_configured()}")
    print(f"Docs: http://localhost:8000/docs")
    print("="*70 + "\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    )
