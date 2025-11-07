"""
Module 9.2: Multi-Hop & Recursive Retrieval - FastAPI Application
Provides REST API endpoints for multi-hop retrieval with health checks and metrics.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Import our implementation
from l2_m9_multi_hop_recursive_retrieval import (
    Document,
    ReferenceExtractor,
    KnowledgeGraphManager,
    MultiHopRetriever,
    RetrievalResult,
    load_example_data
)
from config import (
    get_clients,
    has_required_services,
    MAX_HOP_DEPTH,
    RELEVANCE_THRESHOLD,
    BEAM_WIDTH,
    MAX_TOKENS_PER_QUERY,
    TOP_K_INITIAL,
    TOP_K_PER_HOP
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "retriever": None,
    "graph_manager": None,
    "services_available": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("Initializing multi-hop retrieval system...")

    try:
        # Get clients
        pinecone_index, neo4j_driver, llm_client = get_clients()

        # Initialize components
        graph_manager = KnowledgeGraphManager(neo4j_driver)
        reference_extractor = ReferenceExtractor(llm_client, use_llm=bool(llm_client))
        retriever = MultiHopRetriever(
            vector_index=pinecone_index,
            graph_manager=graph_manager,
            reference_extractor=reference_extractor,
            max_hop_depth=MAX_HOP_DEPTH,
            relevance_threshold=RELEVANCE_THRESHOLD,
            beam_width=BEAM_WIDTH,
            max_tokens=MAX_TOKENS_PER_QUERY
        )

        # Store in app state
        app_state["retriever"] = retriever
        app_state["graph_manager"] = graph_manager
        app_state["services_available"] = has_required_services()

        # Load example data if available
        try:
            documents = load_example_data("example_data.json")
            for doc in documents:
                graph_manager.add_document(doc)
            logger.info(f"Loaded {len(documents)} example documents")
        except Exception as e:
            logger.warning(f"Could not load example data: {e}")

        logger.info("✓ System initialized")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        app_state["services_available"] = False

    yield

    # Cleanup
    logger.info("Shutting down...")
    if app_state["graph_manager"]:
        try:
            # Close Neo4j driver if present
            if hasattr(app_state["graph_manager"], "driver") and app_state["graph_manager"].driver:
                app_state["graph_manager"].driver.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


app = FastAPI(
    title="Multi-Hop & Recursive Retrieval API",
    description="Advanced RAG retrieval following document references across multiple hops",
    version="1.0.0",
    lifespan=lifespan
)

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    query_counter = Counter("multihop_queries_total", "Total queries processed")
    query_duration = Histogram("multihop_query_duration_seconds", "Query duration")
    hop_counter = Histogram("multihop_hops_per_query", "Hops per query")
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.info("Prometheus metrics disabled (prometheus-client not installed)")


# Request/Response Models

class QueryRequest(BaseModel):
    """Request model for multi-hop query."""
    query: str = Field(..., description="Search query")
    top_k_initial: int = Field(TOP_K_INITIAL, ge=1, le=50, description="Initial retrieval top-k")
    top_k_per_hop: int = Field(TOP_K_PER_HOP, ge=1, le=20, description="Per-hop top-k")
    max_hop_depth: Optional[int] = Field(None, ge=1, le=10, description="Override max hop depth")
    relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override relevance threshold")


class DocumentResponse(BaseModel):
    """Document in response."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    hop_distance: int
    references: List[str]


class QueryResponse(BaseModel):
    """Response model for multi-hop query."""
    documents: List[DocumentResponse]
    hop_count: int
    total_documents: int
    execution_time_ms: float
    graph_traversed: Dict[str, List[str]]
    skipped: bool = False
    reason: Optional[str] = None


class IngestRequest(BaseModel):
    """Request model for ingesting documents."""
    documents: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, bool]


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "retriever": app_state["retriever"] is not None,
            "graph_manager": app_state["graph_manager"] is not None,
            "all_services": app_state["services_available"]
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Perform multi-hop retrieval query.

    Returns documents retrieved through multi-hop traversal with citation chains.
    """
    if not app_state["retriever"]:
        # Graceful degradation
        return QueryResponse(
            documents=[],
            hop_count=0,
            total_documents=0,
            execution_time_ms=0.0,
            graph_traversed={},
            skipped=True,
            reason="Retriever not initialized (missing services)"
        )

    try:
        start_time = time.time()

        # Override settings if provided
        retriever = app_state["retriever"]
        original_depth = retriever.max_hop_depth
        original_threshold = retriever.relevance_threshold

        if request.max_hop_depth:
            retriever.max_hop_depth = request.max_hop_depth
        if request.relevance_threshold:
            retriever.relevance_threshold = request.relevance_threshold

        # Perform retrieval
        result = retriever.retrieve(
            query=request.query,
            top_k_initial=request.top_k_initial,
            top_k_per_hop=request.top_k_per_hop
        )

        # Restore original settings
        retriever.max_hop_depth = original_depth
        retriever.relevance_threshold = original_threshold

        # Update metrics
        if METRICS_ENABLED:
            query_counter.inc()
            query_duration.observe(time.time() - start_time)
            hop_counter.observe(result.hop_count)

        # Convert to response model
        doc_responses = [
            DocumentResponse(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=doc.score,
                hop_distance=doc.hop_distance,
                references=doc.references
            )
            for doc in result.documents
        ]

        return QueryResponse(
            documents=doc_responses,
            hop_count=result.hop_count,
            total_documents=result.total_documents,
            execution_time_ms=result.execution_time_ms,
            graph_traversed=result.graph_traversed,
            skipped=False
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@app.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the knowledge graph.

    Adds documents and their references to the graph for retrieval.
    """
    if not app_state["graph_manager"]:
        return {
            "skipped": True,
            "reason": "Graph manager not initialized (missing services)",
            "count": 0
        }

    try:
        graph_manager = app_state["graph_manager"]
        count = 0

        for doc_data in request.documents:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
                references=doc_data.get("references", [])
            )
            graph_manager.add_document(doc)
            count += 1

        logger.info(f"Ingested {count} documents")

        return {
            "status": "success",
            "count": count,
            "skipped": False
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


if METRICS_ENABLED:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# Local development runner
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Multi-Hop Retrieval API server...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
