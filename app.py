"""
FastAPI application for M12.2 Billing Integration.
Provides REST endpoints for billing operations and webhook handling.
"""

from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import logging

from src.l3_m12_billing_integration import (
    StripeBillingManager,
    UsageSyncService,
    DunningManager,
    verify_webhook_signature
)
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="M12.2 Billing Integration API",
    description="Automated subscription and usage-based billing with Stripe",
    version="1.0.0"
)

# Initialize services
billing_manager = StripeBillingManager()
usage_sync = UsageSyncService(billing_manager)
dunning_manager = DunningManager()


# Pydantic Models
class CustomerCreate(BaseModel):
    tenant_id: str
    email: EmailStr
    name: str


class SubscriptionCreate(BaseModel):
    customer_id: str
    plan_type: str
    tenant_id: str
    trial_days: int = 14


class UsageReport(BaseModel):
    tenant_id: str
    subscription_id: str
    query_count: int


class PaymentFailure(BaseModel):
    tenant_id: str
    failure_count: int
    invoice_amount: float


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stripe_configured = Config.is_stripe_configured()

    return {
        "status": "ok",
        "stripe_configured": stripe_configured,
        "module": "M12.2 Billing Integration"
    }


# Customer Management
@app.post("/customers")
async def create_customer(customer: CustomerCreate):
    """
    Create a new Stripe customer.

    Args:
        customer: Customer details (tenant_id, email, name)

    Returns:
        Customer ID and status
    """
    if not Config.is_stripe_configured():
        return {
            "skipped": True,
            "reason": "No Stripe API key configured"
        }

    customer_id = billing_manager.create_customer(
        tenant_id=customer.tenant_id,
        email=customer.email,
        name=customer.name
    )

    if not customer_id:
        raise HTTPException(status_code=500, detail="Failed to create customer")

    return {
        "customer_id": customer_id,
        "tenant_id": customer.tenant_id,
        "status": "created"
    }


# Subscription Management
@app.post("/subscriptions")
async def create_subscription(subscription: SubscriptionCreate):
    """
    Create a new subscription.

    Args:
        subscription: Subscription details (customer_id, plan_type, etc.)

    Returns:
        Subscription details
    """
    if not Config.is_stripe_configured():
        return {
            "skipped": True,
            "reason": "No Stripe API key configured"
        }

    result = billing_manager.create_subscription(
        customer_id=subscription.customer_id,
        plan_type=subscription.plan_type,
        tenant_id=subscription.tenant_id,
        trial_days=subscription.trial_days
    )

    if not result:
        raise HTTPException(status_code=500, detail="Failed to create subscription")

    return result


@app.delete("/subscriptions/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    cancel_at_period_end: bool = True
):
    """
    Cancel a subscription.

    Args:
        subscription_id: Stripe subscription ID
        cancel_at_period_end: If True, service continues until period end

    Returns:
        Cancellation details
    """
    if not Config.is_stripe_configured():
        return {
            "skipped": True,
            "reason": "No Stripe API key configured"
        }

    result = billing_manager.cancel_subscription(
        subscription_id=subscription_id,
        cancel_at_period_end=cancel_at_period_end
    )

    if not result:
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")

    return result


# Usage Reporting
@app.post("/usage/report")
async def report_usage(usage: UsageReport):
    """
    Report usage for a tenant.

    Args:
        usage: Usage details (tenant_id, subscription_id, query_count)

    Returns:
        Success status
    """
    if not Config.is_stripe_configured():
        return {
            "skipped": True,
            "reason": "No Stripe API key configured"
        }

    success = billing_manager.report_usage(
        subscription_id=usage.subscription_id,
        quantity=usage.query_count
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to report usage")

    return {
        "tenant_id": usage.tenant_id,
        "subscription_id": usage.subscription_id,
        "quantity": usage.query_count,
        "status": "reported"
    }


@app.post("/usage/sync")
async def sync_daily_usage(usage_data: List[UsageReport]):
    """
    Sync daily usage for multiple tenants.

    Args:
        usage_data: List of usage reports

    Returns:
        Sync results
    """
    if not Config.is_stripe_configured():
        return {
            "skipped": True,
            "reason": "No Stripe API key configured"
        }

    # Convert to dict format expected by sync service
    data_dicts = [
        {
            "tenant_id": u.tenant_id,
            "subscription_id": u.subscription_id,
            "query_count": u.query_count
        }
        for u in usage_data
    ]

    results = usage_sync.sync_daily_usage(data_dicts)

    return {
        "synced_count": len(results),
        "results": results
    }


# Dunning Management
@app.post("/dunning/process")
async def process_payment_failure(failure: PaymentFailure):
    """
    Process a payment failure and execute dunning logic.

    Args:
        failure: Payment failure details

    Returns:
        Dunning action taken
    """
    result = dunning_manager.process_failed_payment(
        tenant_id=failure.tenant_id,
        failure_count=failure.failure_count,
        invoice_amount=failure.invoice_amount
    )

    return result


@app.post("/dunning/reactivate/{tenant_id}")
async def reactivate_tenant(tenant_id: str):
    """
    Reactivate a tenant after successful payment.

    Args:
        tenant_id: Internal tenant identifier

    Returns:
        Reactivation details
    """
    result = dunning_manager.reactivate_tenant(tenant_id)
    return result


# Webhook Handling
@app.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")
):
    """
    Handle Stripe webhook events.

    Critical events:
    - invoice.payment_succeeded: Payment completed
    - invoice.payment_failed: Payment failed (trigger dunning)
    - customer.subscription.deleted: Subscription cancelled
    - customer.subscription.updated: Plan changed

    Args:
        request: FastAPI request object
        background_tasks: Background task handler
        stripe_signature: Stripe signature header

    Returns:
        Success/error status
    """
    if not Config.STRIPE_WEBHOOK_SECRET:
        logger.warning("⚠️ No webhook secret configured, skipping signature verification")
        return {"status": "skipped", "reason": "No webhook secret"}

    # Get raw body
    payload = await request.body()

    # Verify signature
    event = verify_webhook_signature(
        payload=payload,
        signature=stripe_signature or "",
        webhook_secret=Config.STRIPE_WEBHOOK_SECRET
    )

    if not event:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # Handle event
    event_type = event["type"]
    event_data = event["data"]["object"]

    logger.info(f"📨 Received webhook: {event_type}")

    # Process in background to respond quickly (<5 seconds)
    background_tasks.add_task(process_webhook_event, event_type, event_data)

    return {"status": "success"}


async def process_webhook_event(event_type: str, event_data: dict):
    """
    Process webhook event in background.

    Args:
        event_type: Type of webhook event
        event_data: Event payload
    """
    try:
        if event_type == "invoice.payment_succeeded":
            await handle_payment_succeeded(event_data)

        elif event_type == "invoice.payment_failed":
            await handle_payment_failed(event_data)

        elif event_type == "customer.subscription.deleted":
            await handle_subscription_deleted(event_data)

        elif event_type == "customer.subscription.updated":
            await handle_subscription_updated(event_data)

        else:
            logger.info(f"ℹ️ Unhandled event type: {event_type}")

    except Exception as e:
        logger.error(f"✗ Error processing webhook {event_type}: {e}")


async def handle_payment_succeeded(invoice_data: dict):
    """Handle successful payment"""
    tenant_id = invoice_data.get("metadata", {}).get("tenant_id")
    logger.info(f"✓ Payment succeeded for tenant {tenant_id}")
    # In production: Update database, send receipt email


async def handle_payment_failed(invoice_data: dict):
    """Handle failed payment"""
    tenant_id = invoice_data.get("metadata", {}).get("tenant_id")
    attempt_count = invoice_data.get("attempt_count", 1)
    logger.warning(f"⚠️ Payment failed for tenant {tenant_id} (attempt {attempt_count})")
    # In production: Trigger dunning logic


async def handle_subscription_deleted(subscription_data: dict):
    """Handle subscription cancellation"""
    tenant_id = subscription_data.get("metadata", {}).get("tenant_id")
    logger.info(f"ℹ️ Subscription cancelled for tenant {tenant_id}")
    # In production: Update database, disable access


async def handle_subscription_updated(subscription_data: dict):
    """Handle subscription updates"""
    tenant_id = subscription_data.get("metadata", {}).get("tenant_id")
    logger.info(f"ℹ️ Subscription updated for tenant {tenant_id}")
    # In production: Update database with new plan


# Optional: Metrics endpoint
try:
    from prometheus_client import Counter, make_asgi_app

    webhook_events = Counter(
        'stripe_webhook_events',
        'Stripe webhook events processed',
        ['event_type', 'status']
    )

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

except ImportError:
    logger.info("ℹ️ prometheus-client not installed, /metrics endpoint unavailable")


# Uvicorn runner
if __name__ == "__main__":
    import uvicorn

    print("Starting M12.2 Billing Integration API...")
    print(f"Stripe configured: {Config.is_stripe_configured()}")
    print(f"Listening on {Config.API_HOST}:{Config.API_PORT}")

    uvicorn.run(
        "app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )
