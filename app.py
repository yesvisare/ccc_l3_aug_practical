"""
FastAPI Application for Module 9.4: Advanced Reranking Strategies

Provides REST API endpoints for advanced reranking operations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import logging
import time

from src.l3_m9_advanced_reranking.l3_m9_advanced_reranking_strategies import (
    Document,
    EnsembleReranker,
    MMRReranker,
    TemporalReranker,
    PersonalizationReranker,
    AdvancedReranker
)
from src.l3_m9_advanced_reranking.config import get_config, has_models_available

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Reranking Strategies API",
    description="Module 9.4: Ensemble, MMR, Temporal, and Personalization reranking",
    version="1.0.0"
)

# Global configuration
config = get_config()
models_available = has_models_available()


# Pydantic models for request/response
class DocumentInput(BaseModel):
    """Input document model."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.5


class UserProfile(BaseModel):
    """User profile model."""
    user_id: str
    interaction_count: int = 0
    preferences: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request model for reranking query."""
    query: str
    documents: List[DocumentInput]
    user_profile: Optional[UserProfile] = None
    top_k: Optional[int] = None
    strategies: List[Literal["ensemble", "mmr", "temporal", "personalization"]] = Field(
        default=["ensemble", "mmr", "temporal", "personalization"]
    )
    mmr_lambda: Optional[float] = None
    temporal_decay_days: Optional[int] = None
    temporal_boost_factor: Optional[float] = None


class DocumentOutput(BaseModel):
    """Output document model."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    """Response model for reranking query."""
    documents: List[DocumentOutput]
    latency_ms: float
    strategy_used: str
    skipped: bool = False
    reason: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_available: bool
    timestamp: float


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status information
    """
    return HealthResponse(
        status="ok",
        models_available=models_available,
        timestamp=time.time()
    )


@app.post("/query", response_model=QueryResponse)
async def rerank_query(request: QueryRequest):
    """
    Rerank documents using advanced strategies.

    Args:
        request: Query request with documents and parameters

    Returns:
        QueryResponse with reranked documents
    """
    try:
        # Check if models are available
        if not models_available and "ensemble" in request.strategies:
            logger.warning("Models not available, skipping API call")
            return QueryResponse(
                documents=[
                    DocumentOutput(
                        id=doc.id,
                        text=doc.text,
                        metadata=doc.metadata,
                        score=doc.score
                    )
                    for doc in request.documents
                ],
                latency_ms=0.0,
                strategy_used="none",
                skipped=True,
                reason="no models available"
            )

        # Convert input documents to Document objects
        documents = [
            Document(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=doc.score
            )
            for doc in request.documents
        ]

        # Build configuration
        reranker_config = {
            "model_names": config.RERANKER_MODELS,
            "weights": config.ENSEMBLE_WEIGHTS,
            "aggregation": "weighted",
            "mmr_lambda": request.mmr_lambda or config.MMR_LAMBDA,
            "decay_days": request.temporal_decay_days or config.RECENCY_DECAY_DAYS,
            "boost_factor": request.temporal_boost_factor or config.RECENCY_BOOST_FACTOR,
            "min_interactions": config.MIN_USER_INTERACTIONS
        }

        # Initialize advanced reranker
        reranker = AdvancedReranker(
            enable_ensemble="ensemble" in request.strategies and models_available,
            enable_mmr="mmr" in request.strategies,
            enable_temporal="temporal" in request.strategies,
            enable_personalization="personalization" in request.strategies,
            config=reranker_config
        )

        # Convert user profile if provided
        user_profile_dict = None
        if request.user_profile:
            user_profile_dict = {
                "user_id": request.user_profile.user_id,
                "interaction_count": request.user_profile.interaction_count,
                "preferences": request.user_profile.preferences
            }

        # Perform reranking
        result = reranker.rerank(
            query=request.query,
            documents=documents,
            user_profile=user_profile_dict,
            top_k=request.top_k
        )

        # Convert result to response
        return QueryResponse(
            documents=[
                DocumentOutput(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=doc.score
                )
                for doc in result.documents
            ],
            latency_ms=result.latency_ms,
            strategy_used=result.strategy_used,
            debug_info=result.debug_info
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ensemble", response_model=QueryResponse)
async def ensemble_rerank(request: QueryRequest):
    """
    Rerank using ensemble strategy only.

    Args:
        request: Query request

    Returns:
        QueryResponse with ensemble-reranked documents
    """
    try:
        if not models_available:
            return QueryResponse(
                documents=[
                    DocumentOutput(
                        id=doc.id,
                        text=doc.text,
                        metadata=doc.metadata,
                        score=doc.score
                    )
                    for doc in request.documents
                ],
                latency_ms=0.0,
                strategy_used="ensemble",
                skipped=True,
                reason="no models available"
            )

        documents = [
            Document(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=doc.score
            )
            for doc in request.documents
        ]

        reranker = EnsembleReranker(
            model_names=config.RERANKER_MODELS,
            weights=config.ENSEMBLE_WEIGHTS,
            aggregation="weighted"
        )

        result = reranker.rerank(request.query, documents, top_k=request.top_k)

        return QueryResponse(
            documents=[
                DocumentOutput(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=doc.score
                )
                for doc in result.documents
            ],
            latency_ms=result.latency_ms,
            strategy_used=result.strategy_used,
            debug_info=result.debug_info
        )

    except Exception as e:
        logger.error(f"Error in ensemble reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mmr", response_model=QueryResponse)
async def mmr_rerank(request: QueryRequest):
    """
    Rerank using MMR diversity strategy.

    Args:
        request: Query request

    Returns:
        QueryResponse with MMR-reranked documents
    """
    try:
        documents = [
            Document(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=doc.score
            )
            for doc in request.documents
        ]

        mmr_lambda = request.mmr_lambda or config.MMR_LAMBDA
        reranker = MMRReranker(lambda_param=mmr_lambda)

        result = reranker.rerank(documents, top_k=request.top_k)

        return QueryResponse(
            documents=[
                DocumentOutput(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=doc.score
                )
                for doc in result.documents
            ],
            latency_ms=result.latency_ms,
            strategy_used=result.strategy_used,
            debug_info=result.debug_info
        )

    except Exception as e:
        logger.error(f"Error in MMR reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get Prometheus metrics (optional).

    Returns:
        Metrics in Prometheus format if prometheus-client is available
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        return {"error": "prometheus-client not installed"}


# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Advanced Reranking API server...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
