"""
FastAPI wrapper for Usage Metering & Analytics Module.

Provides REST API endpoints for metering operations:
- Health check
- Usage tracking
- Quota checking
- Invoice generation

No business logic - delegates to l2_m12_usage_metering_analytics.py
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from l2_m12_usage_metering_analytics import (
    UsageEvent, TenantQuota, ClickHouseSchema,
    UsageTracker, QuotaManager, BillingExporter, CostCalculator
)
from config import get_clickhouse_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
client = None
tracker = None
quota_manager = None
billing_exporter = None


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global client, tracker, quota_manager, billing_exporter

    # Startup
    logger.info("Starting Usage Metering API...")
    client = get_clickhouse_client()

    if client:
        ClickHouseSchema.initialize_schema(client)
        tracker = UsageTracker(client)
        await tracker.start()
        quota_manager = QuotaManager(client)
        billing_exporter = BillingExporter(client)
        logger.info("✓ Connected to ClickHouse")
    else:
        logger.warning("⚠️ ClickHouse not available - running in degraded mode")
        tracker = UsageTracker(client=None)
        await tracker.start()

    yield

    # Shutdown
    if tracker:
        await tracker.stop()
    logger.info("Usage Metering API stopped")


app = FastAPI(
    title="Usage Metering & Analytics API",
    description="Production-grade usage metering for multi-tenant SaaS",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class TrackUsageRequest(BaseModel):
    """Request model for tracking usage."""
    tenant_id: str = Field(..., description="Tenant identifier")
    event_type: str = Field(..., description="Event type: query, token_input, token_output, storage")
    quantity: float = Field(..., gt=0, description="Usage quantity")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class QuotaCheckRequest(BaseModel):
    """Request model for quota checking."""
    tenant_id: str = Field(..., description="Tenant identifier")


class SetQuotaRequest(BaseModel):
    """Request model for setting quota."""
    tenant_id: str = Field(..., description="Tenant identifier")
    queries_per_day: int = Field(..., gt=0, description="Daily query limit")
    tokens_per_day: int = Field(..., gt=0, description="Daily token limit")
    storage_gb: float = Field(..., gt=0, description="Storage limit in GB")
    overage_allowed: bool = Field(default=True, description="Allow overages")


class InvoiceRequest(BaseModel):
    """Request model for invoice generation."""
    tenant_id: str = Field(..., description="Tenant identifier")
    year: int = Field(..., ge=2020, le=2100, description="Invoice year")
    month: int = Field(..., ge=1, le=12, description="Invoice month")


# Endpoints
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "clickhouse_available": client is not None,
        "tracker_running": tracker is not None and tracker.running,
    }


@app.post("/track", status_code=status.HTTP_200_OK)
async def track_usage(request: TrackUsageRequest):
    """
    Track a usage event.

    Non-blocking operation with <5ms overhead.
    Falls back to local storage if ClickHouse unavailable.
    """
    if not tracker:
        return {
            "skipped": True,
            "reason": "Tracker not initialized"
        }

    try:
        import uuid
        event = UsageEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=request.tenant_id,
            event_type=request.event_type,
            quantity=request.quantity,
            timestamp=datetime.now(),
            metadata=request.metadata
        )

        await tracker.track(event)

        # Calculate cost
        cost = CostCalculator.calculate_event_cost(event)

        return {
            "success": True,
            "event_id": event.event_id,
            "cost_usd": round(cost, 4),
            "storage_mode": "clickhouse" if client else "fallback_file"
        }

    except Exception as e:
        logger.error(f"Error tracking usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track usage: {str(e)}"
        )


@app.post("/quota/check", status_code=status.HTTP_200_OK)
async def check_quota(request: QuotaCheckRequest):
    """Check tenant quota status."""
    if not quota_manager or not client:
        return {
            "skipped": True,
            "reason": "Quota manager not available (ClickHouse required)"
        }

    try:
        status_data = quota_manager.check_quota(request.tenant_id)
        return status_data

    except Exception as e:
        logger.error(f"Error checking quota: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check quota: {str(e)}"
        )


@app.post("/quota/set", status_code=status.HTTP_200_OK)
async def set_quota(request: SetQuotaRequest):
    """Set or update tenant quota."""
    if not quota_manager or not client:
        return {
            "skipped": True,
            "reason": "Quota manager not available (ClickHouse required)"
        }

    try:
        quota = TenantQuota(
            tenant_id=request.tenant_id,
            queries_per_day=request.queries_per_day,
            tokens_per_day=request.tokens_per_day,
            storage_gb=request.storage_gb,
            overage_allowed=request.overage_allowed
        )

        success = quota_manager.set_quota(quota)

        return {
            "success": success,
            "tenant_id": request.tenant_id,
            "quota": {
                "queries_per_day": request.queries_per_day,
                "tokens_per_day": request.tokens_per_day,
                "storage_gb": request.storage_gb,
            }
        }

    except Exception as e:
        logger.error(f"Error setting quota: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set quota: {str(e)}"
        )


@app.post("/invoice/generate", status_code=status.HTTP_200_OK)
async def generate_invoice(request: InvoiceRequest):
    """Generate monthly invoice for a tenant."""
    if not billing_exporter or not client:
        return {
            "skipped": True,
            "reason": "Billing exporter not available (ClickHouse required)"
        }

    try:
        invoice = billing_exporter.export_monthly_invoice(
            request.tenant_id,
            request.year,
            request.month
        )

        if "error" in invoice:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=invoice["error"]
            )

        return invoice

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating invoice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate invoice: {str(e)}"
        )


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def get_metrics():
    """
    Optional Prometheus-compatible metrics endpoint.

    Returns basic operational metrics.
    """
    metrics = {
        "clickhouse_connected": 1 if client else 0,
        "tracker_running": 1 if (tracker and tracker.running) else 0,
        "buffer_size": len(tracker.buffer) if tracker else 0,
    }

    return metrics


# Local development runner
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
