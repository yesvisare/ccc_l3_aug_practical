"""
FastAPI application for Module 11.3: Resource Management & Throttling
Provides REST API with quota enforcement and fair queuing.
"""

import os
import asyncio
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our modules
try:
    from config import get_redis_client, Config, configure_logging
    from l2_m11_resource_management_throttling import (
        QuotaManager,
        QuotaType,
        FairTenantQueue,
        QueuedRequest,
        QueueWorker,
        process_rag_request
    )
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not import dependencies: {e}")
    REDIS_AVAILABLE = False

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Module 11.3: Resource Management & Throttling",
    description="Multi-tenant quota management with fair queuing",
    version="1.0.0"
)

# Global state (initialized on startup)
redis_client = None
quota_manager = None
fair_queue = None
queue_worker = None


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = "gpt-3.5-turbo"
    context: Optional[str] = ""
    use_tools: bool = False
    use_embeddings: bool = False


class QueryResponse(BaseModel):
    status: str
    request_id: Optional[str] = None
    response: Optional[str] = None
    action: Optional[str] = None
    message: Optional[str] = None


class QuotaUpdateRequest(BaseModel):
    tenant_id: str
    tier: str


# Startup/Shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection and components"""
    global redis_client, quota_manager, fair_queue, queue_worker

    if not REDIS_AVAILABLE:
        logger.warning("Redis dependencies not available - running in limited mode")
        return

    try:
        # Initialize Redis
        redis_client = get_redis_client()
        logger.info("✓ Redis connected")

        # Initialize quota manager
        quota_manager = QuotaManager(redis_client)
        logger.info("✓ Quota manager initialized")

        # Initialize fair queue
        fair_queue = FairTenantQueue(
            redis_client,
            max_queue_size=Config.MAX_QUEUE_SIZE_PER_TENANT
        )
        logger.info("✓ Fair queue initialized")

        # Initialize queue worker (run in background)
        queue_worker = QueueWorker(
            fair_queue,
            quota_manager,
            process_rag_request,
            workers=Config.QUEUE_WORKERS,
            poll_interval=Config.QUEUE_POLL_INTERVAL
        )
        # Start workers in background
        asyncio.create_task(queue_worker.start())
        logger.info(f"✓ Queue worker started ({Config.QUEUE_WORKERS} workers)")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        logger.warning("Running in limited mode without Redis")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    if queue_worker:
        queue_worker.stop()
        logger.info("Queue worker stopped")


# Dependency: Extract tenant ID
def get_tenant_id(x_tenant_id: Optional[str] = Header(None)) -> str:
    """Extract tenant ID from request header"""
    if not x_tenant_id:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Tenant-ID header"
        )
    return x_tenant_id


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "redis_available": redis_client is not None
    }

    if redis_client:
        try:
            redis_client.ping()
            status["redis_status"] = "connected"
        except Exception as e:
            status["redis_status"] = f"error: {e}"

    return status


# Query Endpoint (with quota enforcement)
@app.post("/query", response_model=QueryResponse)
async def submit_query(
    query_request: QueryRequest,
    request: Request,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Submit a query with quota enforcement.

    If tenant is under quota, processes immediately.
    If over quota, queues the request for later processing.
    """
    if not redis_client or not quota_manager or not fair_queue:
        return QueryResponse(
            status="skipped",
            message="⚠️ Skipping API calls (no Redis service)"
        )

    try:
        # Check quota (hourly limit is most restrictive)
        under_quota, current, limit = quota_manager.check_quota(
            tenant_id,
            QuotaType.QUERIES_HOURLY,
            increment=1
        )

        if under_quota:
            # Process immediately
            quota_manager.record_query(tenant_id, tokens_used=1000)

            return QueryResponse(
                status="success",
                request_id=f"req_{datetime.utcnow().timestamp()}",
                response=f"Processed: {query_request.query[:50]}...",
                action="processed_immediately"
            )
        else:
            # Over quota - try to queue
            if Config.ENABLE_QUEUING:
                import uuid
                import time

                queued_request = QueuedRequest(
                    request_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    query=query_request.query,
                    queued_at=time.time()
                )

                queued = await fair_queue.enqueue(queued_request)

                if queued:
                    stats = fair_queue.get_queue_stats()
                    return JSONResponse(
                        status_code=202,  # Accepted
                        content={
                            "status": "queued",
                            "request_id": queued_request.request_id,
                            "action": "queued",
                            "message": f"Request queued. Current position: ~{stats['total_queued_requests']}",
                            "estimated_wait_seconds": stats["total_queued_requests"] * 2
                        },
                        headers={
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": "0",
                            "Retry-After": "30"
                        }
                    )
                else:
                    # Queue full
                    return JSONResponse(
                        status_code=429,
                        content={
                            "status": "rejected",
                            "message": "Quota exceeded and queue is full. Please retry later.",
                            "action": "queue_full"
                        },
                        headers={
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": "0",
                            "Retry-After": "3600"
                        }
                    )
            else:
                # Queuing disabled - reject immediately
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "rejected",
                        "message": f"Quota exceeded: {current}/{limit} queries this hour",
                        "action": "rejected"
                    },
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "Retry-After": "3600"
                    }
                )

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Quota Status Endpoint
@app.get("/quota/status")
async def get_quota_status(tenant_id: str = Depends(get_tenant_id)):
    """Get current quota status for tenant"""
    if not quota_manager:
        return {
            "skipped": True,
            "reason": "no Redis service"
        }

    try:
        status = quota_manager.get_quota_status(tenant_id)
        return status
    except Exception as e:
        logger.error(f"Error getting quota status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Queue Stats Endpoint
@app.get("/queue/stats")
async def get_queue_stats():
    """Get queue statistics (admin endpoint)"""
    if not fair_queue:
        return {
            "skipped": True,
            "reason": "no Redis service"
        }

    try:
        stats = fair_queue.get_queue_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Admin: Update Quota
@app.post("/admin/quota/update")
async def admin_quota_update(
    update_request: QuotaUpdateRequest,
    x_admin_key: Optional[str] = Header(None)
):
    """
    Admin endpoint to update tenant quota tier.

    Requires admin API key in X-Admin-Key header.
    """
    if not quota_manager:
        return {
            "skipped": True,
            "reason": "no Redis service"
        }

    # Verify admin key
    if x_admin_key != Config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing admin API key"
        )

    try:
        # Update tenant tier
        quota_manager.set_tenant_tier(
            update_request.tenant_id,
            update_request.tier
        )

        # Get new status
        new_status = quota_manager.get_quota_status(update_request.tenant_id)

        return {
            "success": True,
            "tenant_id": update_request.tenant_id,
            "new_tier": update_request.tier,
            "new_status": new_status,
            "effective_immediately": True
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating quota: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    # Metrics
    requests_total = Counter(
        'requests_total',
        'Total requests',
        ['tenant_id', 'status']
    )
    quota_usage = Gauge(
        'quota_usage',
        'Current quota usage',
        ['tenant_id', 'quota_type']
    )
    queue_depth = Gauge(
        'queue_depth',
        'Current queue depth',
        ['tenant_id']
    )

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return generate_latest()

except ImportError:
    logger.info("prometheus-client not installed, /metrics endpoint disabled")


# Main entry point
if __name__ == "__main__":
    print("=" * 60)
    print("Module 11.3: Resource Management & Throttling API")
    print("=" * 60)
    print(f"\nStarting server on http://localhost:8000")
    print(f"Docs available at http://localhost:8000/docs")
    print(f"\nRedis: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
    print(f"Queue workers: {Config.QUEUE_WORKERS}")
    print(f"Max queue size: {Config.MAX_TOTAL_QUEUE_SIZE}")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
