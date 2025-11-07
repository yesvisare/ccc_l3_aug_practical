"""
Module 12.2: Billing Integration
Automate subscription and usage-based billing with Stripe.

This module implements:
- Customer management (sync tenants to Stripe)
- Subscription lifecycle (create, cancel, update)
- Usage reporting (ClickHouse to Stripe sync)
- Payment event handling
- Dunning logic for failed payments
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import stripe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StripeBillingManager:
    """Manages all Stripe billing operations"""

    def __init__(self, stripe_api_key: Optional[str] = None):
        """
        Initialize Stripe billing manager.

        Args:
            stripe_api_key: Stripe secret key (defaults to STRIPE_SECRET_KEY env var)
        """
        self.stripe_api_key = stripe_api_key or os.getenv("STRIPE_SECRET_KEY")
        if not self.stripe_api_key:
            logger.warning("⚠️ No Stripe API key configured. Billing operations will fail.")
        else:
            stripe.api_key = self.stripe_api_key

    def create_customer(
        self,
        tenant_id: str,
        email: str,
        name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Create a Stripe customer for a tenant.

        Args:
            tenant_id: Internal tenant identifier
            email: Customer email address
            name: Customer or company name
            metadata: Additional metadata to store

        Returns:
            Stripe customer ID or None if failed
        """
        if not self.stripe_api_key:
            logger.error("⚠️ Skipping customer creation (no Stripe key)")
            return None

        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    "tenant_id": tenant_id,
                    "created_at": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )

            logger.info(f"✓ Created Stripe customer {customer.id} for tenant {tenant_id}")
            return customer.id

        except stripe.error.StripeError as e:
            logger.error(f"✗ Failed to create customer: {e}")
            return None

    def get_or_create_customer(
        self,
        tenant_id: str,
        email: str,
        name: str
    ) -> Optional[str]:
        """
        Get existing customer or create new one (idempotent).

        Args:
            tenant_id: Internal tenant identifier
            email: Customer email address
            name: Customer or company name

        Returns:
            Stripe customer ID or None if failed
        """
        if not self.stripe_api_key:
            logger.error("⚠️ Skipping customer lookup (no Stripe key)")
            return None

        try:
            # Search for existing customer by email
            customers = stripe.Customer.list(email=email, limit=1)

            if customers.data:
                customer = customers.data[0]

                # Update metadata if tenant_id missing
                if customer.metadata.get("tenant_id") != tenant_id:
                    stripe.Customer.modify(
                        customer.id,
                        metadata={"tenant_id": tenant_id}
                    )

                logger.info(f"✓ Found existing customer {customer.id} for tenant {tenant_id}")
                return customer.id
            else:
                # Create new customer
                return self.create_customer(tenant_id, email, name)

        except stripe.error.StripeError as e:
            logger.error(f"✗ Failed to get or create customer: {e}")
            return None

    def attach_payment_method(
        self,
        customer_id: str,
        payment_method_id: str
    ) -> bool:
        """
        Attach a payment method to customer and set as default.

        Args:
            customer_id: Stripe customer ID
            payment_method_id: Payment method ID from Stripe.js frontend

        Returns:
            True if successful, False otherwise
        """
        if not self.stripe_api_key:
            logger.error("⚠️ Skipping payment method attachment (no Stripe key)")
            return False

        try:
            # Attach payment method to customer
            stripe.PaymentMethod.attach(
                payment_method_id,
                customer=customer_id
            )

            # Set as default payment method
            stripe.Customer.modify(
                customer_id,
                invoice_settings={"default_payment_method": payment_method_id}
            )

            logger.info(f"✓ Attached payment method to customer {customer_id}")
            return True

        except stripe.error.StripeError as e:
            logger.error(f"✗ Failed to attach payment method: {e}")
            return False

    def create_subscription(
        self,
        customer_id: str,
        plan_type: str,
        tenant_id: str,
        trial_days: int = 14
    ) -> Optional[Dict[str, Any]]:
        """
        Create a subscription with base + usage-based pricing.

        Args:
            customer_id: Stripe customer ID
            plan_type: 'starter', 'pro', or 'enterprise'
            tenant_id: Internal tenant identifier
            trial_days: Trial period in days (0 for no trial)

        Returns:
            Subscription details or None if failed
        """
        if not self.stripe_api_key:
            logger.error("⚠️ Skipping subscription creation (no Stripe key)")
            return None

        # Define pricing structure (would be price IDs from Stripe Dashboard)
        PLANS = {
            "starter": {
                "base_price": "price_starter_base",
                "included_queries": 10000,
                "overage_price": "price_starter_overage"
            },
            "pro": {
                "base_price": "price_pro_base",
                "included_queries": 100000,
                "overage_price": "price_pro_overage"
            },
            "enterprise": {
                "base_price": "price_enterprise_base",
                "included_queries": 1000000,
                "overage_price": "price_enterprise_overage"
            }
        }

        plan = PLANS.get(plan_type)
        if not plan:
            logger.error(f"✗ Unknown plan type: {plan_type}")
            return None

        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[
                    {"price": plan["base_price"]},
                    {
                        "price": plan["overage_price"],
                        "metadata": {
                            "tenant_id": tenant_id,
                            "metric": "queries",
                            "included_quantity": plan["included_queries"]
                        }
                    }
                ],
                metadata={
                    "tenant_id": tenant_id,
                    "plan_type": plan_type
                },
                trial_period_days=trial_days if trial_days > 0 else None,
                payment_behavior="default_incomplete"
            )

            logger.info(f"✓ Created subscription {subscription.id} for tenant {tenant_id}")

            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end,
                "trial_end": subscription.trial_end
            }

        except stripe.error.StripeError as e:
            logger.error(f"✗ Failed to create subscription: {e}")
            return None

    def cancel_subscription(
        self,
        subscription_id: str,
        cancel_at_period_end: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Cancel a subscription.

        Args:
            subscription_id: Stripe subscription ID
            cancel_at_period_end: If True, service continues until end of period

        Returns:
            Cancellation details or None if failed
        """
        if not self.stripe_api_key:
            logger.error("⚠️ Skipping subscription cancellation (no Stripe key)")
            return None

        try:
            if cancel_at_period_end:
                # Cancel at end of period (graceful)
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                # Cancel immediately
                subscription = stripe.Subscription.delete(subscription_id)

            logger.info(f"✓ Cancelled subscription {subscription_id}")

            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "cancel_at": subscription.cancel_at
            }

        except stripe.error.StripeError as e:
            logger.error(f"✗ Failed to cancel subscription: {e}")
            return None

    def report_usage(
        self,
        subscription_id: str,
        quantity: int,
        timestamp: Optional[int] = None
    ) -> bool:
        """
        Report usage to Stripe for billing.

        Args:
            subscription_id: Stripe subscription ID
            quantity: Usage quantity (e.g., number of queries)
            timestamp: Unix timestamp (defaults to now)

        Returns:
            True if successful, False otherwise
        """
        if not self.stripe_api_key:
            logger.error("⚠️ Skipping usage reporting (no Stripe key)")
            return False

        try:
            # Get subscription to find metered item
            subscription = stripe.Subscription.retrieve(subscription_id)
            usage_item_id = None

            for item in subscription["items"]["data"]:
                if item["price"]["recurring"].get("usage_type") == "metered":
                    usage_item_id = item.id
                    break

            if not usage_item_id:
                logger.error(f"✗ No metered item found for subscription {subscription_id}")
                return False

            # Create usage record
            stripe.SubscriptionItem.create_usage_record(
                usage_item_id,
                quantity=quantity,
                timestamp=timestamp or int(datetime.utcnow().timestamp()),
                action="set"  # "set" replaces, "increment" adds
            )

            logger.info(f"✓ Reported {quantity} units for subscription {subscription_id}")
            return True

        except stripe.error.StripeError as e:
            logger.error(f"✗ Failed to report usage: {e}")
            return False


class UsageSyncService:
    """Syncs usage from data sources to Stripe"""

    def __init__(self, billing_manager: StripeBillingManager):
        """
        Initialize usage sync service.

        Args:
            billing_manager: StripeBillingManager instance
        """
        self.billing = billing_manager

    def sync_daily_usage(
        self,
        usage_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sync usage data to Stripe for all tenants.

        Args:
            usage_data: List of dicts with keys: tenant_id, query_count, subscription_id

        Returns:
            List of sync results
        """
        results = []

        for data in usage_data:
            tenant_id = data.get("tenant_id")
            query_count = data.get("query_count", 0)
            subscription_id = data.get("subscription_id")

            if not subscription_id:
                logger.warning(f"⚠️ No subscription ID for tenant {tenant_id}")
                continue

            success = self.billing.report_usage(
                subscription_id=subscription_id,
                quantity=query_count
            )

            results.append({
                "tenant_id": tenant_id,
                "query_count": query_count,
                "success": success
            })

        logger.info(f"✓ Synced usage for {len(results)} tenants")
        return results


class DunningManager:
    """Manages payment retry logic and service suspension"""

    def __init__(self):
        """Initialize dunning manager"""
        pass

    def process_failed_payment(
        self,
        tenant_id: str,
        failure_count: int,
        invoice_amount: float
    ) -> Dict[str, Any]:
        """
        Process payment failure with escalating strategy.

        Args:
            tenant_id: Internal tenant identifier
            failure_count: Number of consecutive failures
            invoice_amount: Amount owed

        Returns:
            Action taken
        """
        action = None

        if failure_count == 1:
            # Day 1: Soft reminder
            action = "reminder_sent"
            logger.info(f"📧 Sending payment reminder to tenant {tenant_id}")

        elif failure_count == 2:
            # Day 4: Warning
            action = "warning_sent"
            logger.warning(f"⚠️ Sending payment warning to tenant {tenant_id}")

        elif failure_count == 3:
            # Day 7: Final warning + soft suspension
            action = "rate_limit_reduced"
            logger.warning(f"⚠️ Reducing rate limits for tenant {tenant_id}")

        elif failure_count >= 4:
            # Day 8+: Full suspension
            action = "service_suspended"
            logger.error(f"🚫 Suspending service for tenant {tenant_id}")

        return {
            "tenant_id": tenant_id,
            "action": action,
            "failure_count": failure_count,
            "invoice_amount": invoice_amount
        }

    def reactivate_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """
        Reactivate tenant after successful payment.

        Args:
            tenant_id: Internal tenant identifier

        Returns:
            Reactivation details
        """
        logger.info(f"✓ Reactivating tenant {tenant_id}")

        return {
            "tenant_id": tenant_id,
            "action": "reactivated",
            "timestamp": datetime.utcnow().isoformat()
        }


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    webhook_secret: str
) -> Optional[Dict[str, Any]]:
    """
    Verify Stripe webhook signature (CRITICAL for security).

    Args:
        payload: Raw request body
        signature: Stripe-Signature header value
        webhook_secret: Webhook signing secret from Stripe Dashboard

    Returns:
        Verified event dict or None if invalid
    """
    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=signature,
            secret=webhook_secret
        )
        return event
    except ValueError:
        logger.error("✗ Invalid webhook payload")
        return None
    except stripe.error.SignatureVerificationError:
        logger.error("✗ Invalid webhook signature")
        return None


# CLI Usage Examples
if __name__ == "__main__":
    print("Module 12.2: Billing Integration")
    print("=" * 50)

    # Example: Create customer
    billing = StripeBillingManager()

    if not os.getenv("STRIPE_SECRET_KEY"):
        print("⚠️ No STRIPE_SECRET_KEY found in environment")
        print("Set STRIPE_SECRET_KEY to test billing operations")
    else:
        print("\n1. Creating test customer...")
        customer_id = billing.create_customer(
            tenant_id="test_tenant_001",
            email="test@example.com",
            name="Test Company"
        )
        print(f"   Customer ID: {customer_id}")

        if customer_id:
            print("\n2. Creating subscription...")
            subscription = billing.create_subscription(
                customer_id=customer_id,
                plan_type="pro",
                tenant_id="test_tenant_001",
                trial_days=14
            )
            if subscription:
                print(f"   Subscription ID: {subscription['subscription_id']}")
                print(f"   Status: {subscription['status']}")

        print("\n3. Testing dunning logic...")
        dunning = DunningManager()
        result = dunning.process_failed_payment(
            tenant_id="test_tenant_001",
            failure_count=1,
            invoice_amount=99.00
        )
        print(f"   Action: {result['action']}")

    print("\n" + "=" * 50)
    print("Examples complete. See README.md for full usage.")
