"""
FastAPI application for ReAct Pattern Implementation.
Provides REST API endpoints for agentic query processing.

Module entrypoint with health checks and query endpoints.
"""
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import Config
from l2_m10_react_pattern_implementation import StatefulReActAgent

# Setup logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ReAct Pattern Implementation API",
    description="Module 10.1: Agentic RAG with ReAct pattern for multi-step reasoning",
    version="1.0.0"
)

# Initialize agent (singleton)
agent: Optional[StatefulReActAgent] = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global agent
    try:
        if Config.OPENAI_API_KEY and Config.ENABLE_AGENT:
            logger.info("Initializing ReAct agent...")
            agent = StatefulReActAgent(
                model_name=Config.AGENT_MODEL,
                temperature=Config.AGENT_TEMPERATURE,
                max_iterations=Config.AGENT_MAX_ITERATIONS,
                timeout_seconds=Config.AGENT_TIMEOUT_SECONDS
            )
            logger.info("Agent initialized successfully")
        else:
            logger.warning("Agent not initialized - missing API key or disabled in config")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        agent = None


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query to process", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversations")
    use_agent: bool = Field(True, description="Whether to use agent or fallback to static pipeline")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Compare our Q3 revenue of $125,000 to the SaaS industry growth rate",
                "session_id": "user-session-123",
                "use_agent": True
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer to the query")
    reasoning_steps: int = Field(..., description="Number of reasoning steps taken")
    agent_used: bool = Field(..., description="Whether the agent was used")
    session_id: Optional[str] = Field(None, description="Session ID for this conversation")
    error: Optional[str] = Field(None, description="Error message if any")
    skipped: bool = Field(False, description="Whether execution was skipped (e.g., no API keys)")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the analysis, our Q3 revenue of $125,000 shows strong performance...",
                "reasoning_steps": 4,
                "agent_used": True,
                "session_id": "user-session-123",
                "error": None,
                "skipped": False
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    agent_available: bool
    config: dict


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ReAct Pattern Implementation API",
        "version": "1.0.0",
        "module": "M10.1",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "metrics": "/metrics (if prometheus-client installed)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns system status and configuration info.
    """
    return HealthResponse(
        status="ok",
        agent_available=agent is not None,
        config=Config.get_info()
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query endpoint with optional agentic reasoning.

    Processes user queries using ReAct agent for complex multi-step reasoning,
    or falls back to simple processing if agent is unavailable or disabled.

    Args:
        request: QueryRequest with query, session_id, and use_agent flag

    Returns:
        QueryResponse with answer, reasoning steps, and metadata
    """
    try:
        # Check if agent is available
        if not agent:
            if not Config.OPENAI_API_KEY:
                return QueryResponse(
                    answer="",
                    reasoning_steps=0,
                    agent_used=False,
                    session_id=request.session_id,
                    error=None,
                    skipped=True
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Agent initialization failed - check logs"
                )

        if request.use_agent and Config.ENABLE_AGENT:
            # Use ReAct agent for complex queries
            logger.info(f"Processing query with agent: {request.query[:50]}...")

            result = agent.query(request.query, request.session_id)

            return QueryResponse(
                answer=result['output'],
                reasoning_steps=result['num_steps'],
                agent_used=True,
                session_id=request.session_id,
                error=result.get('error'),
                skipped=False
            )
        else:
            # Use simple processing (fallback)
            logger.info(f"Processing query without agent: {request.query[:50]}...")

            # Simple fallback response
            fallback_answer = f"Simple processing mode: Your query '{request.query}' would be handled by the static pipeline. Agent is disabled or not requested."

            return QueryResponse(
                answer=fallback_answer,
                reasoning_steps=1,
                agent_used=False,
                session_id=request.session_id,
                error=None,
                skipped=False
            )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# METRICS (Optional)
# ============================================================================

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    # Define metrics
    query_counter = Counter(
        'agent_query_requests_total',
        'Total number of query requests',
        ['agent_used', 'status']
    )

    query_duration = Histogram(
        'agent_query_duration_seconds',
        'Query processing duration in seconds',
        ['agent_used']
    )

    query_steps = Histogram(
        'agent_steps_total',
        'Number of reasoning steps per query',
        buckets=[1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
    )

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("Prometheus metrics enabled at /metrics")

except ImportError:
    logger.info("prometheus-client not installed - metrics endpoint disabled")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Run the application with uvicorn."""
    import sys

    # Check configuration
    if not Config.OPENAI_API_KEY:
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Agent will not be available. Set it in .env or environment.")
        print()

    print("=" * 60)
    print("ReAct Pattern Implementation API")
    print("Module 10.1: Agentic RAG & Tool Use")
    print("=" * 60)
    print()
    print("Configuration:")
    for key, value in Config.get_info().items():
        print(f"  {key}: {value}")
    print()
    print("Starting server on http://localhost:8000")
    print("  - Health check: http://localhost:8000/health")
    print("  - API docs: http://localhost:8000/docs")
    print("  - Query endpoint: POST http://localhost:8000/query")
    print()

    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
