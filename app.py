"""
FastAPI wrapper for M11.4 Vector Index Sharding.

Endpoints:
- GET /health: Health check
- POST /query: Single-tenant or cross-shard query
- POST /ingest: Ingest documents for tenant
- GET /metrics: Shard health monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from config import Config, get_clients, has_required_services
from src.l3_m11_vector_index_sharding import ShardManager, ShardedRAG, monitor_shard_health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="M11.4 Vector Index Sharding",
    description="Production-grade sharded vector search for multi-tenant SaaS",
    version="1.0.0"
)

# Initialize clients and services
pinecone_client, redis_client, openai_client = get_clients()
shard_manager = ShardManager(redis_client, num_shards=Config.NUM_SHARDS, shard_prefix=Config.SHARD_PREFIX)
rag = ShardedRAG(pinecone_client, openai_client, shard_manager, vector_dimension=Config.VECTOR_DIMENSION)


# Request/Response models
class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}


class IngestRequest(BaseModel):
    tenant_id: str
    documents: List[Document]


class QueryRequest(BaseModel):
    query_text: str
    tenant_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    cross_shard: bool = False


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    latency_ms: Optional[float] = None
    shard: Optional[str] = None
    shards_queried: Optional[int] = None
    skipped: bool = False
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    config: Dict[str, Any]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with service availability."""
    return {
        "status": "ok",
        "services": {
            "pinecone": pinecone_client is not None,
            "redis": redis_client is not None,
            "openai": openai_client is not None
        },
        "config": {
            "num_shards": Config.NUM_SHARDS,
            "shard_prefix": Config.SHARD_PREFIX,
            "vector_dimension": Config.VECTOR_DIMENSION
        }
    }


@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents for a tenant.

    Routes tenant to their assigned shard and creates namespace.
    """
    if not has_required_services():
        return {
            "success": False,
            "skipped": True,
            "reason": "no keys/service",
            "message": "Configure PINECONE_API_KEY and OPENAI_API_KEY"
        }

    try:
        # Convert Pydantic models to dicts
        documents = [doc.model_dump() for doc in request.documents]

        result = rag.upsert_documents(
            tenant_id=request.tenant_id,
            documents=documents
        )

        return result

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents.

    - tenant_id provided: Single-tenant query (fast, ~350ms P95)
    - cross_shard=True: Query all shards (slower, admin use)
    """
    if not has_required_services():
        return QueryResponse(
            results=[],
            skipped=True,
            reason="no keys/service"
        )

    try:
        if request.cross_shard:
            # Cross-shard query (admin)
            result = rag.query_cross_shard(
                query_text=request.query_text,
                top_k=request.top_k
            )
            return QueryResponse(
                results=result.get("results", []),
                latency_ms=result.get("latency_ms"),
                shards_queried=result.get("shards_queried"),
                skipped=result.get("skipped", False),
                reason=result.get("reason")
            )
        elif request.tenant_id:
            # Single-tenant query (fast path)
            result = rag.query_single_tenant(
                tenant_id=request.tenant_id,
                query_text=request.query_text,
                top_k=request.top_k
            )
            return QueryResponse(
                results=result.get("results", []),
                latency_ms=result.get("latency_ms"),
                shard=result.get("shard"),
                skipped=result.get("skipped", False),
                reason=result.get("reason")
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide tenant_id or set cross_shard=True"
            )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Shard health metrics.

    Returns:
    - Namespace counts per shard
    - Vector counts per shard
    - Rebalancing alerts
    """
    if not pinecone_client:
        return {
            "skipped": True,
            "reason": "Pinecone not configured"
        }

    try:
        # Shard distribution
        shard_stats = shard_manager.get_shard_stats()

        # Health monitoring
        health = monitor_shard_health(shard_manager, pinecone_client)

        return {
            "distribution": shard_stats,
            "health": health,
            "thresholds": {
                "max_vectors_per_shard": Config.MAX_VECTORS_PER_SHARD,
                "max_namespaces_per_shard": Config.MAX_NAMESPACES_PER_SHARD,
                "rebalance_threshold": Config.REBALANCE_NAMESPACE_THRESHOLD
            }
        }

    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shard/{tenant_id}")
async def get_tenant_shard(tenant_id: str):
    """Get shard assignment for specific tenant."""
    shard_id = shard_manager.get_shard_for_tenant(tenant_id)
    index_name = shard_manager.get_shard_index_name(tenant_id)

    return {
        "tenant_id": tenant_id,
        "shard_id": shard_id,
        "index_name": index_name
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting M11.4 Vector Index Sharding API...")
    print(f"Shards: {Config.NUM_SHARDS}")
    print(f"Services: Pinecone={pinecone_client is not None}, Redis={redis_client is not None}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
