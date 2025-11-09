"""
Module 7.2: Application Performance Monitoring - FastAPI Wrapper
Provides HTTP endpoints for APM-instrumented RAG pipeline
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import uvicorn

# Import module functions
from l2_m7_application_performance_monitoring import (
    ProfiledRAGPipeline,
    MemoryProfiledComponent,
    monitor_memory_leak,
    apm_manager
)
from config import apm_config, app_config

logger = logging.getLogger(__name__)

# ==========================================
# FastAPI App Initialization
# ==========================================

app = FastAPI(
    title=app_config.APP_NAME,
    version=app_config.APP_VERSION,
    description="Application Performance Monitoring for RAG systems with Datadog APM"
)

# Initialize global instances
pipeline = None
memory_profiler = None


@app.on_event("startup")
async def startup_event():
    """Initialize APM and components on startup"""
    global pipeline, memory_profiler

    logger.info("Starting Module 7.2 API...")

    # Initialize APM
    if apm_config.is_configured:
        success = apm_manager.initialize()
        if success:
            logger.info("✅ APM initialized successfully")
        else:
            logger.warning("⚠️  APM initialization failed")
    else:
        logger.warning("⚠️  APM not configured - running without APM")

    # Initialize pipeline
    pipeline = ProfiledRAGPipeline()
    memory_profiler = MemoryProfiledComponent()

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")

    # Cleanup memory profiler
    if memory_profiler:
        memory_profiler.cleanup()

    # Shutdown APM
    apm_manager.shutdown()

    logger.info("Shutdown complete")


# ==========================================
# Request/Response Models
# ==========================================

class QueryRequest(BaseModel):
    """RAG query request"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    user_id: Optional[str] = Field("anonymous", description="User identifier for tracking")


class QueryResponse(BaseModel):
    """RAG query response"""
    query: str
    response: str
    context_length: int
    num_results: int
    apm_enabled: bool


class MemoryCheckRequest(BaseModel):
    """Memory leak check request"""
    iterations: int = Field(10, ge=1, le=100, description="Number of iterations to test")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    apm_enabled: bool
    apm_configured: bool
    service: str
    environment: str


# ==========================================
# API Endpoints
# ==========================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns API status and APM configuration
    """
    return HealthResponse(
        status="ok",
        apm_enabled=apm_manager.is_initialized,
        apm_configured=apm_config.is_configured,
        service=apm_config.DD_SERVICE,
        environment=apm_config.DD_ENV
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process RAG query with APM profiling

    This endpoint demonstrates:
    - End-to-end RAG pipeline profiling
    - Custom span tagging
    - Exception tracking in APM

    Args:
        request: Query request with user query and ID

    Returns:
        Query response with generated answer
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Validate query length
        if len(request.query) > app_config.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Query too long (max {app_config.MAX_QUERY_LENGTH} chars)"
            )

        # Process query with APM profiling
        result = pipeline.process_query(
            query=request.query,
            user_id=request.user_id
        )

        # Return response
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            context_length=result["context_length"],
            num_results=result["num_results"],
            apm_enabled=apm_manager.is_initialized
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/memory/stats")
async def get_memory_stats():
    """
    Get current memory usage statistics

    Returns:
        Memory metrics including current usage, peak, and growth
    """
    if not memory_profiler:
        return {"skipped": True, "reason": "Memory profiler not initialized"}

    try:
        stats = memory_profiler.get_memory_stats()
        return {
            "status": "ok",
            "memory_stats": stats
        }
    except Exception as e:
        logger.error(f"Memory stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/leak-check")
async def check_memory_leak(request: MemoryCheckRequest):
    """
    Run memory leak detection test

    This endpoint runs multiple iterations of document caching
    to detect potential memory leaks

    Args:
        request: Test configuration with iteration count

    Returns:
        Leak detection results
    """
    try:
        results = monitor_memory_leak(iterations=request.iterations)
        return {
            "status": "ok",
            "leak_check": results
        }
    except Exception as e:
        logger.error(f"Memory leak check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/apm/config")
async def get_apm_config():
    """
    Get current APM configuration (sanitized)

    Returns:
        APM configuration without sensitive data
    """
    return {
        "apm_enabled": apm_manager.is_initialized,
        "configured": apm_config.is_configured,
        "service": apm_config.DD_SERVICE,
        "environment": apm_config.DD_ENV,
        "version": apm_config.DD_VERSION,
        "profiling_enabled": apm_config.DD_PROFILING_ENABLED,
        "sample_rate": apm_config.DD_TRACE_SAMPLE_RATE,
        "profiling_capture_pct": apm_config.DD_PROFILING_CAPTURE_PCT,
        "max_cpu_overhead_pct": apm_config.DD_PROFILING_MAX_TIME_USAGE_PCT
    }


@app.get("/apm/metrics")
async def get_apm_metrics():
    """
    Get APM metrics and status

    Returns:
        Current APM metrics if prometheus-client is available
    """
    if not apm_manager.is_initialized:
        return {
            "skipped": True,
            "reason": "APM not initialized"
        }

    # In a real implementation, you would expose prometheus metrics here
    # For now, return basic status
    return {
        "status": "ok",
        "apm_initialized": apm_manager.is_initialized,
        "message": "APM metrics available in Datadog UI"
    }


# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with APM tracking"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app_config.LOG_LEVEL == "DEBUG" else "An error occurred"
        }
    )


# ==========================================
# Development Server
# ==========================================

if __name__ == "__main__":
    """
    Run development server

    Usage:
        python app.py

    The API will be available at:
        - http://localhost:8000
        - Docs: http://localhost:8000/docs
        - OpenAPI: http://localhost:8000/openapi.json
    """
    import sys

    print("=" * 60)
    print("Module 7.2: Application Performance Monitoring API")
    print("=" * 60)
    print()
    print("Starting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print()
    print("Endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /query               - Process RAG query")
    print("  GET  /memory/stats        - Memory statistics")
    print("  POST /memory/leak-check   - Memory leak detection")
    print("  GET  /apm/config          - APM configuration")
    print("  GET  /apm/metrics         - APM metrics")
    print()
    print("APM Status:")
    print(f"  Configured: {apm_config.is_configured}")
    print(f"  Service: {apm_config.DD_SERVICE}")
    print(f"  Environment: {apm_config.DD_ENV}")
    print()
    print("=" * 60)
    print()

    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=app_config.LOG_LEVEL.lower()
    )
