"""
Module 12.4: Tenant Lifecycle Management - FastAPI Application
API wrapper for tenant lifecycle operations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from config import Config, get_stripe_client
import l2_m12_tenant_lifecycle_management as lifecycle

# Configure logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tenant Lifecycle Management API",
    description="Module 12.4: Complete tenant lifecycle operations",
    version="1.0.0"
)

# Optional Prometheus metrics
if Config.ENABLE_METRICS:
    try:
        from prometheus_client import Counter, Histogram, make_asgi_app

        lifecycle_operations = Counter(
            'lifecycle_operations_total',
            'Total lifecycle operations',
            ['operation', 'status']
        )
        operation_duration = Histogram(
            'lifecycle_operation_duration_seconds',
            'Lifecycle operation duration',
            ['operation']
        )

        # Mount metrics endpoint
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
    except ImportError:
        logger.warning("prometheus_client not installed, metrics disabled")


# Request/Response Models

class TenantData(BaseModel):
    """Tenant data model."""
    tenant_id: str
    name: str
    email: str
    current_plan: str
    state: str = "active"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    current_usage: Optional[Dict[str, int]] = None


class UpgradeRequest(BaseModel):
    """Upgrade request model."""
    tenant: TenantData
    to_plan: str = Field(..., description="Target plan (starter, professional, enterprise)")


class DowngradeRequest(BaseModel):
    """Downgrade request model."""
    tenant: TenantData
    to_plan: str = Field(..., description="Target plan (free, starter, professional)")
    schedule_for_period_end: bool = Field(
        default=True,
        description="Apply downgrade at billing period end"
    )


class ExportRequest(BaseModel):
    """Data export request model."""
    tenant: TenantData
    export_type: str = Field(default="full", description="Export type (full, incremental)")
    requested_by: str = Field(default="api", description="User requesting export")


class DeletionRequest(BaseModel):
    """Deletion request model."""
    tenant: TenantData
    requested_by: str = Field(..., description="User requesting deletion")


class ReactivationRequest(BaseModel):
    """Reactivation request model."""
    tenant: TenantData
    reactivation_plan: Optional[str] = Field(None, description="Plan to reactivate on")


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "tenant-lifecycle-management"}


@app.post("/upgrade")
async def upgrade_tenant(request: UpgradeRequest):
    """
    Upgrade tenant to higher plan tier.

    Provisions resources before billing changes to avoid service interruption.
    Includes automatic rollback on failure.
    """
    try:
        # Get Stripe client if configured
        stripe_client = get_stripe_client()

        if not stripe_client:
            logger.warning("Stripe not configured, upgrade will skip billing updates")

        # Execute upgrade
        result = lifecycle.upgrade_tenant(
            tenant_data=request.tenant.dict(),
            to_plan=request.to_plan,
            plan_hierarchy=Config.PLAN_HIERARCHY,
            plan_limits=Config.PLAN_LIMITS,
            stripe_client=stripe_client
        )

        # Track metrics if enabled
        if Config.ENABLE_METRICS:
            status = "success" if result.get("success") else "failure"
            lifecycle_operations.labels(operation="upgrade", status=status).inc()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Upgrade failed: {str(e)}")
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="upgrade", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/downgrade")
async def downgrade_tenant(request: DowngradeRequest):
    """
    Downgrade tenant to lower plan tier.

    Validates current usage fits new plan limits.
    Can schedule downgrade for billing period end.
    """
    try:
        # Get Stripe client if configured
        stripe_client = get_stripe_client()

        if not stripe_client and request.tenant.stripe_subscription_id:
            logger.warning("Stripe not configured, downgrade will skip billing updates")

        # Execute downgrade
        result = lifecycle.downgrade_tenant(
            tenant_data=request.tenant.dict(),
            to_plan=request.to_plan,
            plan_hierarchy=Config.PLAN_HIERARCHY,
            plan_limits=Config.PLAN_LIMITS,
            stripe_client=stripe_client
        )

        # Track metrics if enabled
        if Config.ENABLE_METRICS:
            status = "success" if result.get("success") else "failure"
            lifecycle_operations.labels(operation="downgrade", status=status).inc()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Downgrade failed: {str(e)}")
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="downgrade", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_data(request: ExportRequest, background_tasks: BackgroundTasks):
    """
    Export tenant data for GDPR compliance.

    Generates chunked export in background job.
    Returns job ID for status tracking.
    """
    try:
        # Initiate export
        result = lifecycle.export_tenant_data(
            tenant_data=request.tenant.dict(),
            export_type=request.export_type
        )

        # Track metrics if enabled
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="export", status="initiated").inc()

        # In production: background_tasks would trigger actual Celery job
        # background_tasks.add_task(process_export_job, result["export_id"])

        return result

    except Exception as e:
        logger.error(f"Export initiation failed: {str(e)}")
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="export", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete_tenant(request: DeletionRequest):
    """
    Soft-delete tenant with retention period.

    Tenant can be reactivated during retention period (default 30 days).
    After retention, data is permanently deleted.
    """
    try:
        # Execute soft delete
        result = lifecycle.delete_tenant(
            tenant_data=request.tenant.dict(),
            requested_by=request.requested_by
        )

        # Track metrics if enabled
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="delete", status="soft_deleted").inc()

        return result

    except Exception as e:
        logger.error(f"Deletion failed: {str(e)}")
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="delete", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reactivate")
async def reactivate_tenant(request: ReactivationRequest):
    """
    Reactivate suspended or soft-deleted tenant.

    Used for win-back campaigns and recovery workflows.
    Validates tenant can be reactivated (within retention period).
    """
    try:
        # Get Stripe client if configured
        stripe_client = get_stripe_client()

        if not stripe_client:
            logger.warning("Stripe not configured, reactivation will skip billing updates")

        # Execute reactivation
        result = lifecycle.reactivate_tenant(
            tenant_data=request.tenant.dict(),
            reactivation_plan=request.reactivation_plan,
            stripe_client=stripe_client
        )

        # Track metrics if enabled
        if Config.ENABLE_METRICS:
            status = "success" if result.get("success") else "failure"
            lifecycle_operations.labels(operation="reactivate", status=status).inc()

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Reactivation failed: {str(e)}")
        if Config.ENABLE_METRICS:
            lifecycle_operations.labels(operation="reactivate", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plans")
async def get_plans():
    """Get available plans and their limits."""
    return {
        "plan_hierarchy": Config.PLAN_HIERARCHY,
        "plan_limits": Config.PLAN_LIMITS
    }


@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "retention_days": Config.SOFT_DELETE_RETENTION_DAYS,
        "export_chunk_size_mb": Config.DATA_EXPORT_CHUNK_SIZE_MB,
        "max_concurrent_jobs": Config.MAX_CONCURRENT_LIFECYCLE_JOBS,
        "auto_downgrade_enabled": Config.ENABLE_AUTO_DOWNGRADE,
        "reactivation_enabled": Config.ENABLE_REACTIVATION_WORKFLOW,
        "metrics_enabled": Config.ENABLE_METRICS,
        "stripe_configured": Config.STRIPE_API_KEY is not None
    }


# Development server runner
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Tenant Lifecycle Management API...")
    logger.info(f"Stripe configured: {Config.STRIPE_API_KEY is not None}")
    logger.info(f"Metrics enabled: {Config.ENABLE_METRICS}")

    uvicorn.run(
        "app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD,
        log_level=Config.LOG_LEVEL.lower()
    )
