"""
FastAPI wrapper for Conversational RAG with Memory.

Provides REST API endpoints for conversational queries with session management.
"""

import logging
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from src.l3_m10_conversational_rag_memory import ConversationalRAG
from config import get_clients, Config, validate_config
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OFFLINE mode (env-driven)
OFFLINE = os.getenv("OFFLINE", "false").lower() == "true"
if OFFLINE:
    logger.info("Running in OFFLINE mode - API/model calls will be skipped")

# Initialize FastAPI app
app = FastAPI(
    title="Conversational RAG with Memory",
    description="Multi-turn dialogue system with memory and reference resolution",
    version="1.0.0",
)

# Global state
clients = get_clients()
openai_client = clients["openai"]
redis_client = clients["redis"]

# Initialize RAG system
rag_system: Optional[ConversationalRAG] = None
if openai_client:
    rag_system = ConversationalRAG(
        llm_client=openai_client,
        redis_client=redis_client,
        short_term_size=Config.SHORT_TERM_BUFFER_SIZE,
        max_context_tokens=Config.MAX_CONTEXT_TOKENS,
        model=Config.DEFAULT_MODEL,
        spacy_model=Config.SPACY_MODEL,
    )
    logger.info("RAG system initialized")
else:
    logger.warning("RAG system not initialized - API keys missing")


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="User query text", min_length=1)
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation persistence"
    )
    create_session: bool = Field(
        False, description="Auto-create session if not provided"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    response: str = Field(..., description="Assistant response")
    session_id: Optional[str] = Field(None, description="Session ID used")
    memory_stats: Dict[str, Any] = Field(..., description="Current memory statistics")
    skipped: bool = Field(False, description="True if API call was skipped")
    reason: Optional[str] = Field(None, description="Reason for skipping")


class SessionStatsResponse(BaseModel):
    """Response model for session stats."""

    session_id: str
    exists: bool
    memory_stats: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""

    session_id: Optional[str] = Field(None, description="Session ID to reset")


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and configuration validation.
    """
    is_valid, warnings = validate_config()

    return {
        "status": "ok",
        "config_valid": is_valid,
        "warnings": warnings,
        "rag_initialized": rag_system is not None,
        "redis_available": redis_client is not None,
    }


# Main query endpoint
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Process a conversational query with memory and reference resolution.

    Args:
        request: Query request with text and optional session_id

    Returns:
        Assistant response with memory statistics

    Raises:
        HTTPException: If query processing fails
    """
    # Check OFFLINE mode
    if OFFLINE:
        return QueryResponse(
            response="[OFFLINE MODE] System would process: " + request.query,
            session_id=request.session_id,
            memory_stats={"short_term_turns": 0, "has_long_term_summary": False, "estimated_tokens": 0},
            skipped=True,
            reason="offline mode enabled",
        )

    # Check if system is available
    if not rag_system:
        return QueryResponse(
            response="System not available",
            session_id=request.session_id,
            memory_stats={},
            skipped=True,
            reason="no keys/service",
        )

    try:
        # Handle session ID
        session_id = request.session_id
        if request.create_session and not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {session_id}")

        # Process query
        response = rag_system.query(request.query, session_id=session_id)

        # Get memory stats
        stats = rag_system.get_memory_stats()

        return QueryResponse(
            response=response,
            session_id=session_id,
            memory_stats=stats,
            skipped=False,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


# Session management
@app.get("/session/{session_id}", response_model=SessionStatsResponse, tags=["Session"])
async def get_session_stats(session_id: str):
    """
    Get statistics for a specific session.

    Args:
        session_id: Session ID to check

    Returns:
        Session statistics
    """
    if not rag_system:
        return SessionStatsResponse(
            session_id=session_id, exists=False, memory_stats=None
        )

    exists = rag_system.session_manager.session_exists(session_id)

    if exists:
        # Load session to get stats
        loaded_memory = rag_system.session_manager.load_session(
            session_id,
            rag_system.memory.short_term_size,
            rag_system.memory.max_context_tokens,
            rag_system.llm_client,
            rag_system.model,
        )

        if loaded_memory:
            stats = {
                "short_term_turns": len(loaded_memory.short_term_buffer),
                "has_long_term_summary": bool(loaded_memory.long_term_summary),
                "estimated_tokens": loaded_memory._estimate_tokens(),
            }
            return SessionStatsResponse(
                session_id=session_id, exists=True, memory_stats=stats
            )

    return SessionStatsResponse(session_id=session_id, exists=False, memory_stats=None)


@app.post("/session/reset", tags=["Session"])
async def reset_session(request: ResetRequest):
    """
    Reset conversation memory for a session.

    Args:
        request: Reset request with optional session_id

    Returns:
        Success message
    """
    if not rag_system:
        return {"status": "skipped", "reason": "no keys/service"}

    if request.session_id:
        # Delete specific session
        deleted = rag_system.session_manager.delete_session(request.session_id)
        return {
            "status": "success",
            "message": f"Session {request.session_id} {'deleted' if deleted else 'not found'}",
        }
    else:
        # Reset in-memory state
        rag_system.reset_memory()
        return {"status": "success", "message": "Memory reset"}


# Optional metrics endpoint
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    # Metrics
    query_counter = Counter("queries_total", "Total number of queries")
    query_duration = Histogram("query_duration_seconds", "Query processing duration")
    session_counter = Counter("sessions_created_total", "Total sessions created")

    @app.get("/metrics", tags=["Metrics"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("Metrics endpoint enabled")

except ImportError:
    logger.info("prometheus-client not installed, metrics endpoint disabled")


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Conversational RAG with Memory API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "session_stats": "GET /session/{session_id}",
            "reset": "POST /session/reset",
        },
        "documentation": "/docs",
    }


# Uvicorn runner for local development
if __name__ == "__main__":
    import sys

    # Check configuration
    is_valid, warnings = validate_config()

    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if not openai_client:
        print("⚠️  Warning: OpenAI API key not configured")
        print("   API will return mock responses")
        print()

    # Run server
    print("Starting Conversational RAG API server...")
    print("Documentation available at: http://localhost:8000/docs")
    print()

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
