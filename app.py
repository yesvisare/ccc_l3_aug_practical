"""
FastAPI wrapper for Module 9: HyDE

Provides REST API endpoints for hypothesis generation and retrieval.
All business logic is in l2_m9_hypothetical_document_embeddings.py
"""

import logging
import time
from typing import Optional, List, Dict, Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

# Import module components
from src.l3_m9_hypothetical_document_embeddings import (
    AdaptiveHyDERetriever,
    HyDEGenerator,
    QueryClassifier
)
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Module 9: HyDE API",
    description="Hypothetical Document Embeddings for advanced retrieval",
    version="1.0.0"
)

# Global retriever (initialized on startup)
retriever: Optional[AdaptiveHyDERetriever] = None
hypothesis_generator: Optional[HyDEGenerator] = None
classifier: Optional[QueryClassifier] = None


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User question", min_length=1)
    top_k: int = Field(default=10, description="Number of results", ge=1, le=100)
    force_method: Optional[Literal["hyde", "traditional", "hybrid"]] = Field(
        default=None,
        description="Force specific retrieval method (overrides automatic routing)"
    )
    domain_context: Optional[str] = Field(
        default=None,
        description="Optional domain context for hypothesis generation"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    results: List[Dict[str, Any]]
    hypothesis: Optional[str]
    routing: Dict[str, Any]
    performance: Dict[str, float]
    metadata: Dict[str, Any]


class HypothesisRequest(BaseModel):
    """Request model for hypothesis generation endpoint."""
    query: str = Field(..., description="User question", min_length=1)
    domain_context: Optional[str] = Field(
        default=None,
        description="Optional domain context"
    )


class HypothesisResponse(BaseModel):
    """Response model for hypothesis generation."""
    hypothesis: str
    success: bool
    generation_time_ms: float
    tokens_used: int


class ClassifyRequest(BaseModel):
    """Request model for query classification."""
    query: str = Field(..., description="User question", min_length=1)


class ClassifyResponse(BaseModel):
    """Response model for query classification."""
    use_hyde: bool
    confidence: float
    beneficial_signals: int
    harmful_signals: int
    reasoning: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    components: Dict[str, bool]
    timestamp: float


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global retriever, hypothesis_generator, classifier

    logger.info("Initializing HyDE API...")

    # Validate config
    status = config.validate_config()

    if not status["openai"]:
        logger.error("⚠️ OPENAI_API_KEY not set. API will return errors.")
        return

    # Initialize components
    try:
        retriever = AdaptiveHyDERetriever(
            openai_api_key=config.OPENAI_API_KEY,
            pinecone_api_key=config.PINECONE_API_KEY,
            pinecone_index_name=config.PINECONE_INDEX_NAME
        )
        logger.info("✓ Adaptive retriever initialized")

        hypothesis_generator = HyDEGenerator(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.HYDE_MODEL,
            temperature=config.HYDE_TEMPERATURE,
            max_tokens=config.HYDE_MAX_TOKENS
        )
        logger.info("✓ Hypothesis generator initialized")

        classifier = QueryClassifier()
        logger.info("✓ Query classifier initialized")

        logger.info("✓ HyDE API ready!")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "Module 9: HyDE API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "hypothesis": "/hypothesis",
            "classify": "/classify"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    status = config.validate_config()

    return HealthResponse(
        status="ok" if status["openai"] else "degraded",
        components={
            "openai": status["openai"],
            "pinecone": status["pinecone"],
            "retriever": retriever is not None,
            "generator": hypothesis_generator is not None,
            "classifier": classifier is not None
        },
        timestamp=time.time()
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query endpoint - main retrieval interface.

    Automatically routes to best retrieval method based on query type.
    """
    if not retriever:
        # Graceful degradation
        if not config.OPENAI_API_KEY:
            return QueryResponse(
                results=[],
                hypothesis=None,
                routing={
                    "method_used": "none",
                    "classification": {},
                    "reasoning": "No OpenAI API key configured"
                },
                performance={"total_time_ms": 0},
                metadata={"skipped": True, "reason": "no keys/service"}
            )
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        result = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            force_method=request.force_method
        )

        return QueryResponse(
            results=result["results"],
            hypothesis=result.get("hypothesis"),
            routing=result["routing"],
            performance=result["performance"],
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hypothesis", response_model=HypothesisResponse)
async def hypothesis_endpoint(request: HypothesisRequest):
    """
    Generate hypothetical document answer for a query.

    Useful for testing and debugging hypothesis quality.
    """
    if not hypothesis_generator:
        if not config.OPENAI_API_KEY:
            return HypothesisResponse(
                hypothesis=request.query,
                success=False,
                generation_time_ms=0,
                tokens_used=0
            )
        raise HTTPException(status_code=503, detail="Generator not initialized")

    try:
        result = hypothesis_generator.generate_hypothesis(
            query=request.query,
            domain_context=request.domain_context
        )

        return HypothesisResponse(
            hypothesis=result["hypothesis"],
            success=result["success"],
            generation_time_ms=result["generation_time_ms"],
            tokens_used=result["tokens_used"]
        )

    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(request: ClassifyRequest):
    """
    Classify query to determine if HyDE should be used.

    Useful for understanding routing decisions.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    try:
        result = classifier.should_use_hyde(request.query)

        return ClassifyResponse(
            use_hyde=result["use_hyde"],
            confidence=result["confidence"],
            beneficial_signals=result["beneficial_signals"],
            harmful_signals=result["harmful_signals"],
            reasoning=result["reasoning"]
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    # Metrics
    request_count = Counter(
        "hyde_requests_total",
        "Total number of requests",
        ["endpoint", "method"]
    )

    request_duration = Histogram(
        "hyde_request_duration_seconds",
        "Request duration in seconds",
        ["endpoint"]
    )

    hypothesis_generation_duration = Histogram(
        "hyde_hypothesis_generation_seconds",
        "Hypothesis generation duration in seconds"
    )

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("✓ Prometheus metrics enabled at /metrics")

except ImportError:
    logger.debug("Prometheus client not installed (optional)")


# Uvicorn runner
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
