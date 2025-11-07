"""
FastAPI wrapper for M12.3 Self-Service Tenant Onboarding.

Provides REST API endpoints:
- GET /health - Health check
- POST /signup - Tenant signup
- POST /webhook/stripe - Stripe webhook handler
- POST /provision - Manual provisioning trigger (admin)
- GET /tenant/{tenant_id} - Get tenant status
- GET /metrics - Activation metrics
"""

from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime

# Import module functions
from l2_m12_self_service_tenant_onboarding import (
    create_skeleton_tenant,
    generate_stripe_checkout_url,
    verify_stripe_webhook,
    handle_checkout_completed,
    trigger_provisioning_task,
    provision_tenant,
    send_welcome_email,
    get_setup_wizard_steps,
    track_activation_event,
    calculate_activation_metrics,
    check_provisioning_timeout,
    TenantStatus,
    PlanType,
)

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Self-Service Tenant Onboarding API",
    description="Module 12.3: Automated SaaS customer onboarding",
    version="1.0.0"
)

# In-memory storage (use database in production)
TENANT_STORE: Dict[str, Any] = {}
ACTIVATION_EVENTS: List[Dict[str, Any]] = []

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    signup_counter = Counter('tenant_signups_total', 'Total tenant signups')
    provisioning_counter = Counter('tenant_provisioning_total', 'Total provisioning attempts')
    activation_counter = Counter('tenant_activations_total', 'Total tenant activations')

    METRICS_ENABLED = True
except ImportError:
    logger.warning("prometheus_client not installed, metrics disabled")
    METRICS_ENABLED = False


# ===========================
# Request/Response Models
# ===========================

class SignupRequest(BaseModel):
    """Signup request model."""
    email: EmailStr
    company_name: str
    password: str
    plan: str = "starter"


class SignupResponse(BaseModel):
    """Signup response model."""
    tenant_id: str
    status: str
    checkout_url: str
    message: str


class ProvisionRequest(BaseModel):
    """Manual provisioning request (admin)."""
    tenant_id: str
    force: bool = False


class TenantStatusResponse(BaseModel):
    """Tenant status response."""
    tenant_id: str
    status: str
    email: str
    company_name: str
    plan: str
    created_at: str
    activated_at: Optional[str] = None


class ActivationEventRequest(BaseModel):
    """Activation event tracking request."""
    tenant_id: str
    event_type: str
    metadata: Optional[Dict[str, Any]] = None


# ===========================
# Endpoints
# ===========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    config_status = config.verify_config()

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "services": config_status,
        "tenants_count": len(TENANT_STORE),
    }


@app.post("/signup", response_model=SignupResponse)
async def signup(request: SignupRequest):
    """
    Public signup endpoint.

    Creates skeleton tenant and returns Stripe checkout URL.
    """
    try:
        # Get Stripe client
        stripe_client = config.get_stripe_client()

        if not stripe_client and config.STRIPE_SECRET_KEY:
            logger.warning("⚠️ Stripe client initialization failed")

        # Create skeleton tenant
        tenant = create_skeleton_tenant(
            email=request.email,
            company_name=request.company_name,
            password=request.password,
            plan=request.plan
        )

        # Store tenant
        TENANT_STORE[tenant['tenant_id']] = tenant

        # Generate checkout URL
        checkout_url = generate_stripe_checkout_url(
            tenant['tenant_id'],
            request.plan,
            stripe_client
        )

        # Track signup event
        ACTIVATION_EVENTS.append(
            track_activation_event(tenant['tenant_id'], 'signup_completed')
        )

        if METRICS_ENABLED:
            signup_counter.inc()

        logger.info(f"Signup completed for {request.email}: {tenant['tenant_id']}")

        return SignupResponse(
            tenant_id=tenant['tenant_id'],
            status=tenant['status'],
            checkout_url=checkout_url,
            message="Please complete payment to activate your account"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
    background_tasks: BackgroundTasks = None
):
    """
    Stripe webhook handler.

    Handles checkout.session.completed events.
    """
    try:
        payload = await request.body()

        # Verify webhook signature
        if not verify_stripe_webhook(
            payload.decode('utf-8'),
            stripe_signature or "",
            config.STRIPE_WEBHOOK_SECRET
        ):
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Parse event
        event = json.loads(payload)
        event_type = event.get('type')

        if event_type == 'checkout.session.completed':
            # Update tenant status
            updated_tenant = handle_checkout_completed(event['data'], TENANT_STORE)

            # Track payment event
            ACTIVATION_EVENTS.append(
                track_activation_event(updated_tenant['tenant_id'], 'payment_confirmed')
            )

            # Trigger provisioning (background)
            if background_tasks:
                background_tasks.add_task(
                    _provision_tenant_background,
                    updated_tenant['tenant_id']
                )

            logger.info(f"Webhook processed: {event_type} for {updated_tenant['tenant_id']}")

            return {"status": "success", "tenant_id": updated_tenant['tenant_id']}

        return {"status": "ignored", "event_type": event_type}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/provision")
async def manual_provision(request: ProvisionRequest):
    """
    Manual provisioning trigger (admin endpoint).

    Use for retry after failures or manual intervention.
    """
    try:
        if request.tenant_id not in TENANT_STORE:
            raise HTTPException(status_code=404, detail="Tenant not found")

        tenant = TENANT_STORE[request.tenant_id]

        # Check if already active
        if tenant['status'] == TenantStatus.ACTIVE.value and not request.force:
            return {
                "status": "already_active",
                "tenant_id": request.tenant_id,
                "message": "Tenant already active. Use force=true to re-provision."
            }

        # Load sample data
        try:
            with open('example_data.json', 'r') as f:
                data = json.load(f)
                sample_docs = data.get('sample_documents', [])
        except Exception as e:
            logger.warning(f"Could not load sample data: {e}")
            sample_docs = []

        # Provision
        provisioned = provision_tenant(
            request.tenant_id,
            TENANT_STORE,
            pinecone_client=config.get_pinecone_client(),
            stripe_client=config.get_stripe_client(),
            sample_docs=sample_docs
        )

        # Send welcome email
        send_welcome_email(provisioned, config.get_sendgrid_client())

        if METRICS_ENABLED:
            provisioning_counter.inc()
            if provisioned['status'] == TenantStatus.ACTIVE.value:
                activation_counter.inc()

        return {
            "status": "success",
            "tenant_id": request.tenant_id,
            "tenant_status": provisioned['status']
        }

    except Exception as e:
        logger.error(f"Provisioning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tenant/{tenant_id}", response_model=TenantStatusResponse)
async def get_tenant(tenant_id: str):
    """Get tenant status."""
    if tenant_id not in TENANT_STORE:
        raise HTTPException(status_code=404, detail="Tenant not found")

    tenant = TENANT_STORE[tenant_id]

    # Check for timeout
    if check_provisioning_timeout(tenant):
        tenant['status'] = TenantStatus.FAILED.value
        tenant['error_message'] = "Provisioning timeout (>5 minutes)"

    return TenantStatusResponse(
        tenant_id=tenant['tenant_id'],
        status=tenant['status'],
        email=tenant['email'],
        company_name=tenant['company_name'],
        plan=tenant['plan'],
        created_at=tenant['created_at'],
        activated_at=tenant.get('activated_at')
    )


@app.get("/wizard/steps")
async def get_wizard_steps():
    """Get setup wizard steps."""
    return {"steps": get_setup_wizard_steps()}


@app.post("/activation/track")
async def track_event(request: ActivationEventRequest):
    """Track activation event."""
    event = track_activation_event(
        request.tenant_id,
        request.event_type,
        request.metadata
    )
    ACTIVATION_EVENTS.append(event)

    return {"status": "tracked", "event": event}


@app.get("/activation/metrics")
async def get_activation_metrics():
    """Get activation funnel metrics."""
    metrics = calculate_activation_metrics(ACTIVATION_EVENTS)

    return {
        "metrics": metrics,
        "total_events": len(ACTIVATION_EVENTS),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not METRICS_ENABLED:
        return {"skipped": True, "reason": "prometheus_client not installed"}

    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ===========================
# Background Tasks
# ===========================

def _provision_tenant_background(tenant_id: str):
    """Background provisioning task."""
    try:
        logger.info(f"Background provisioning started for {tenant_id}")

        # Load sample data
        try:
            with open('example_data.json', 'r') as f:
                data = json.load(f)
                sample_docs = data.get('sample_documents', [])
        except Exception as e:
            logger.warning(f"Could not load sample data: {e}")
            sample_docs = []

        # Provision
        provisioned = provision_tenant(
            tenant_id,
            TENANT_STORE,
            pinecone_client=config.get_pinecone_client(),
            stripe_client=config.get_stripe_client(),
            sample_docs=sample_docs
        )

        # Send welcome email
        send_welcome_email(provisioned, config.get_sendgrid_client())

        # Track activation
        ACTIVATION_EVENTS.append(
            track_activation_event(tenant_id, 'first_login')
        )

        if METRICS_ENABLED:
            provisioning_counter.inc()
            activation_counter.inc()

        logger.info(f"Background provisioning completed for {tenant_id}")

    except Exception as e:
        logger.error(f"Background provisioning failed for {tenant_id}: {e}")


# ===========================
# Startup/Shutdown
# ===========================

@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    logger.info("Starting Self-Service Tenant Onboarding API")
    config_status = config.verify_config()

    for service, available in config_status.items():
        status = "✓" if available else "⚠"
        logger.info(f"{status} {service}: {'configured' if available else 'not configured'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks."""
    logger.info("Shutting down Self-Service Tenant Onboarding API")


# ===========================
# Local Development
# ===========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
