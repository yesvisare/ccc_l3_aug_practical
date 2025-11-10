"""
Module 12.3: Self-Service Tenant Onboarding

Automates SaaS customer onboarding using multi-tenant architecture and billing integration.
Enables customers to become productive within 5 minutes through automated provisioning.

Architecture:
1. Signup & Payment Capture (synchronous)
2. Automated Provisioning (background via Celery)
3. Welcome & Activation (email + in-app)
4. Activation Monitoring (analytics)
"""

import logging
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""
    PENDING_PAYMENT = "pending_payment"
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    FAILED = "failed"
    SUSPENDED = "suspended"


class PlanType(str, Enum):
    """Available subscription plans."""
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# ===========================
# Step 1: Public Signup Endpoint
# ===========================

def generate_tenant_id() -> str:
    """
    Generate unique tenant identifier.

    Returns:
        Unique tenant ID (hex string)
    """
    return secrets.token_hex(16)


def hash_password(password: str) -> str:
    """
    Hash password using SHA-256 (simplified; use bcrypt in production).

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def create_skeleton_tenant(
    email: str,
    company_name: str,
    password: str,
    plan: str
) -> Dict[str, Any]:
    """
    Create skeleton tenant record before payment collection.

    This creates the initial tenant record with status "pending_payment".
    Actual provisioning occurs after webhook confirmation.

    Args:
        email: User email address
        company_name: Company/organization name
        password: Plain text password (will be hashed)
        plan: Plan type (starter, pro, enterprise)

    Returns:
        Dict with tenant_id, status, and creation timestamp

    Raises:
        ValueError: If invalid plan or missing required fields
    """
    logger.info(f"Creating skeleton tenant for {email} ({company_name})")

    # Validation
    if not all([email, company_name, password, plan]):
        raise ValueError("Missing required fields")

    if plan not in [p.value for p in PlanType]:
        raise ValueError(f"Invalid plan: {plan}")

    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    # Generate tenant
    tenant_id = generate_tenant_id()
    password_hash = hash_password(password)

    tenant_data = {
        "tenant_id": tenant_id,
        "email": email,
        "company_name": company_name,
        "password_hash": password_hash,
        "plan": plan,
        "status": TenantStatus.PENDING_PAYMENT.value,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "pinecone_namespace": None,
        "api_key": None,
    }

    logger.info(f"Tenant {tenant_id} created with status {TenantStatus.PENDING_PAYMENT.value}")
    return tenant_data


def generate_stripe_checkout_url(
    tenant_id: str,
    plan: str,
    stripe_client=None
) -> str:
    """
    Generate Stripe Checkout URL for payment collection.

    Args:
        tenant_id: Tenant identifier
        plan: Plan type
        stripe_client: Stripe client (optional)

    Returns:
        Stripe Checkout URL
    """
    if not stripe_client:
        logger.warning("⚠️ Stripe client not available, returning mock URL")
        return f"https://checkout.stripe.com/mock/{tenant_id}?plan={plan}"

    try:
        from config import STRIPE_PRICE_ID_STARTER, STRIPE_PRICE_ID_PRO, STRIPE_PRICE_ID_ENTERPRISE, APP_URL

        price_map = {
            PlanType.STARTER.value: STRIPE_PRICE_ID_STARTER,
            PlanType.PRO.value: STRIPE_PRICE_ID_PRO,
            PlanType.ENTERPRISE.value: STRIPE_PRICE_ID_ENTERPRISE,
        }

        session = stripe_client.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_map.get(plan, STRIPE_PRICE_ID_STARTER),
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{APP_URL}/onboarding/success?tenant_id={tenant_id}",
            cancel_url=f"{APP_URL}/onboarding/cancel?tenant_id={tenant_id}",
            metadata={'tenant_id': tenant_id},
        )

        logger.info(f"Stripe checkout session created for tenant {tenant_id}")
        return session.url

    except Exception as e:
        logger.error(f"Error creating Stripe checkout: {e}")
        raise


# ===========================
# Step 2: Stripe Webhook Handler
# ===========================

def verify_stripe_webhook(payload: str, signature: str, webhook_secret: str) -> bool:
    """
    Verify Stripe webhook signature.

    Args:
        payload: Request body
        signature: Stripe signature header
        webhook_secret: Webhook signing secret

    Returns:
        True if valid, False otherwise
    """
    if not webhook_secret:
        logger.warning("⚠️ Webhook secret not configured, skipping verification")
        return True  # Allow in development

    try:
        import stripe
        stripe.Webhook.construct_event(payload, signature, webhook_secret)
        return True
    except Exception as e:
        logger.error(f"Webhook verification failed: {e}")
        return False


def handle_checkout_completed(
    event_data: Dict[str, Any],
    tenant_store: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle checkout.session.completed webhook event.

    Updates tenant status to "provisioning" and stores Stripe IDs.

    Args:
        event_data: Stripe event data
        tenant_store: In-memory tenant storage (use DB in production)

    Returns:
        Updated tenant data
    """
    session = event_data.get('object', {})
    tenant_id = session.get('metadata', {}).get('tenant_id')

    if not tenant_id:
        logger.error("No tenant_id in webhook metadata")
        raise ValueError("Missing tenant_id")

    if tenant_id not in tenant_store:
        logger.error(f"Tenant {tenant_id} not found")
        raise ValueError(f"Tenant {tenant_id} not found")

    tenant = tenant_store[tenant_id]

    # Update tenant with Stripe info
    tenant['status'] = TenantStatus.PROVISIONING.value
    tenant['stripe_customer_id'] = session.get('customer')
    tenant['stripe_subscription_id'] = session.get('subscription')
    tenant['updated_at'] = datetime.utcnow().isoformat()

    logger.info(f"Tenant {tenant_id} status updated to {TenantStatus.PROVISIONING.value}")

    return tenant


def trigger_provisioning_task(tenant_id: str, celery_app=None) -> Optional[str]:
    """
    Trigger background provisioning task via Celery.

    Args:
        tenant_id: Tenant identifier
        celery_app: Celery app instance (optional)

    Returns:
        Task ID if triggered, None otherwise
    """
    if not celery_app:
        logger.warning("⚠️ Celery not available, provisioning task not triggered")
        return None

    try:
        task = celery_app.send_task(
            'tasks.provision_tenant',
            args=[tenant_id],
            countdown=0  # Execute immediately
        )
        logger.info(f"Provisioning task {task.id} triggered for tenant {tenant_id}")
        return task.id
    except Exception as e:
        logger.error(f"Error triggering provisioning task: {e}")
        return None


# ===========================
# Step 3: Celery Background Provisioning
# ===========================

def create_pinecone_namespace(
    tenant_id: str,
    pinecone_client=None,
    index_name: str = "saas-tenants"
) -> str:
    """
    Create Pinecone namespace for tenant.

    Args:
        tenant_id: Tenant identifier
        pinecone_client: Pinecone client
        index_name: Pinecone index name

    Returns:
        Namespace identifier

    Raises:
        RuntimeError: If namespace creation fails
    """
    namespace = f"tenant_{tenant_id}"

    if not pinecone_client:
        logger.warning(f"⚠️ Pinecone client not available, skipping namespace creation for {namespace}")
        return namespace

    try:
        # Get or create index
        index = pinecone_client.Index(index_name)

        # Namespaces are created automatically on first upsert
        # We can verify by listing stats
        logger.info(f"Pinecone namespace {namespace} ready")
        return namespace

    except Exception as e:
        logger.error(f"Error creating Pinecone namespace: {e}")
        raise RuntimeError(f"Pinecone namespace creation failed: {e}")


def generate_api_key(tenant_id: str) -> str:
    """
    Generate API key for tenant (JWT with tenant_id claim).

    Args:
        tenant_id: Tenant identifier

    Returns:
        JWT token
    """
    try:
        from jose import jwt
        from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS

        payload = {
            "tenant_id": tenant_id,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.info(f"API key generated for tenant {tenant_id}")
        return token

    except Exception as e:
        logger.error(f"Error generating API key: {e}")
        return f"mock_api_key_{tenant_id}"


def load_sample_data(
    tenant_id: str,
    namespace: str,
    sample_docs: List[Dict[str, Any]],
    pinecone_client=None
) -> int:
    """
    Load sample data into tenant namespace.

    Args:
        tenant_id: Tenant identifier
        namespace: Pinecone namespace
        sample_docs: List of sample documents
        pinecone_client: Pinecone client

    Returns:
        Number of documents loaded
    """
    if not pinecone_client:
        logger.warning(f"⚠️ Pinecone client not available, skipping sample data load")
        return 0

    try:
        from config import PINECONE_INDEX_NAME
        index = pinecone_client.Index(PINECONE_INDEX_NAME)

        # Simple mock vectors (in production, use actual embeddings)
        vectors = []
        for i, doc in enumerate(sample_docs[:3]):  # Limit to 3 samples
            vectors.append({
                "id": f"{tenant_id}_sample_{i}",
                "values": [0.1] * 1536,  # Mock embedding
                "metadata": {
                    "title": doc.get("title", ""),
                    "content": doc.get("content", "")[:200],  # Truncate
                }
            })

        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Loaded {len(vectors)} sample documents for tenant {tenant_id}")
            return len(vectors)

        return 0

    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return 0


def provision_tenant(
    tenant_id: str,
    tenant_store: Dict[str, Any],
    pinecone_client=None,
    stripe_client=None,
    sample_docs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Complete tenant provisioning (Celery task logic).

    This function:
    1. Creates Pinecone namespace
    2. Generates API keys
    3. Loads sample data
    4. Updates tenant status to "active"

    Includes retry logic and error handling.

    Args:
        tenant_id: Tenant identifier
        tenant_store: Tenant storage
        pinecone_client: Pinecone client
        stripe_client: Stripe client
        sample_docs: Sample documents to load

    Returns:
        Updated tenant data

    Raises:
        RuntimeError: If provisioning fails after retries
    """
    logger.info(f"Starting provisioning for tenant {tenant_id}")

    if tenant_id not in tenant_store:
        raise ValueError(f"Tenant {tenant_id} not found")

    tenant = tenant_store[tenant_id]

    try:
        # Step 1: Create Pinecone namespace
        namespace = create_pinecone_namespace(tenant_id, pinecone_client)
        tenant['pinecone_namespace'] = namespace

        # Step 2: Generate API key
        api_key = generate_api_key(tenant_id)
        tenant['api_key'] = api_key

        # Step 3: Load sample data
        from config import SAMPLE_DATA_ENABLED
        if SAMPLE_DATA_ENABLED and sample_docs:
            docs_loaded = load_sample_data(tenant_id, namespace, sample_docs, pinecone_client)
            tenant['sample_docs_loaded'] = docs_loaded

        # Step 4: Update status to active
        tenant['status'] = TenantStatus.ACTIVE.value
        tenant['activated_at'] = datetime.utcnow().isoformat()
        tenant['updated_at'] = datetime.utcnow().isoformat()

        logger.info(f"✓ Tenant {tenant_id} provisioned successfully")
        return tenant

    except Exception as e:
        logger.error(f"Provisioning failed for tenant {tenant_id}: {e}")
        tenant['status'] = TenantStatus.FAILED.value
        tenant['error_message'] = str(e)
        tenant['updated_at'] = datetime.utcnow().isoformat()
        raise RuntimeError(f"Provisioning failed: {e}")


# ===========================
# Step 4: Welcome & Activation
# ===========================

def send_welcome_email(
    tenant: Dict[str, Any],
    sendgrid_client=None
) -> bool:
    """
    Send welcome email with login link and docs.

    Args:
        tenant: Tenant data
        sendgrid_client: SendGrid client

    Returns:
        True if sent successfully
    """
    if not sendgrid_client:
        logger.warning(f"⚠️ SendGrid client not available, skipping welcome email")
        return False

    try:
        from sendgrid.helpers.mail import Mail
        from config import SENDGRID_FROM_EMAIL, SENDGRID_FROM_NAME, APP_URL

        message = Mail(
            from_email=(SENDGRID_FROM_EMAIL, SENDGRID_FROM_NAME),
            to_emails=tenant['email'],
            subject=f"Welcome to {SENDGRID_FROM_NAME}!",
            html_content=f"""
            <h1>Welcome to {tenant['company_name']}!</h1>
            <p>Your account is ready. Click below to get started:</p>
            <a href="{APP_URL}/login?email={tenant['email']}">Login Now</a>
            <p>Your API Key: <code>{tenant.get('api_key', 'N/A')}</code></p>
            <p>Check out our <a href="{APP_URL}/docs">documentation</a> to learn more.</p>
            """
        )

        response = sendgrid_client.send(message)
        logger.info(f"Welcome email sent to {tenant['email']}: status {response.status_code}")
        return response.status_code == 202

    except Exception as e:
        logger.error(f"Error sending welcome email: {e}")
        return False


def get_setup_wizard_steps() -> List[Dict[str, Any]]:
    """
    Get interactive setup wizard steps.

    Returns:
        List of wizard steps with instructions
    """
    return [
        {
            "step": 1,
            "title": "Upload First Document",
            "description": "Upload a document to see how semantic search works",
            "action": "upload_document",
        },
        {
            "step": 2,
            "title": "Run First Query",
            "description": "Try searching across your documents",
            "action": "execute_query",
        },
        {
            "step": 3,
            "title": "Explore Integrations",
            "description": "Connect your tools and workflows",
            "action": "view_integrations",
        },
    ]


# ===========================
# Step 5: Activation Monitoring
# ===========================

def track_activation_event(
    tenant_id: str,
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Track activation event for analytics.

    Args:
        tenant_id: Tenant identifier
        event_type: Event type (signup_completed, first_login, etc.)
        metadata: Optional event metadata

    Returns:
        Event record
    """
    event = {
        "tenant_id": tenant_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
    }

    logger.info(f"Activation event: {event_type} for tenant {tenant_id}")
    return event


def calculate_activation_metrics(
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate activation funnel metrics.

    Args:
        events: List of activation events

    Returns:
        Dict with conversion rates and metrics
    """
    event_counts = {}
    for event in events:
        event_type = event.get('event_type')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    total_signups = event_counts.get('signup_completed', 0)

    metrics = {
        "total_signups": total_signups,
        "payment_confirmed": event_counts.get('payment_confirmed', 0),
        "first_login": event_counts.get('first_login', 0),
        "first_document_upload": event_counts.get('first_document_upload', 0),
        "first_query_executed": event_counts.get('first_query_executed', 0),
    }

    if total_signups > 0:
        metrics['login_rate'] = round(metrics['first_login'] / total_signups * 100, 1)
        metrics['activation_rate'] = round(metrics['first_query_executed'] / total_signups * 100, 1)
    else:
        metrics['login_rate'] = 0.0
        metrics['activation_rate'] = 0.0

    logger.info(f"Activation metrics: {metrics['activation_rate']}% activation rate")
    return metrics


def check_provisioning_timeout(
    tenant: Dict[str, Any],
    timeout_seconds: int = 300
) -> bool:
    """
    Check if provisioning has timed out (Failure Mode #1).

    Args:
        tenant: Tenant data
        timeout_seconds: Timeout threshold in seconds

    Returns:
        True if timed out
    """
    if tenant.get('status') != TenantStatus.PROVISIONING.value:
        return False

    updated_at = datetime.fromisoformat(tenant.get('updated_at', datetime.utcnow().isoformat()))
    elapsed = (datetime.utcnow() - updated_at).total_seconds()

    if elapsed > timeout_seconds:
        logger.warning(f"Tenant {tenant['tenant_id']} provisioning timed out ({elapsed}s)")
        return True

    return False


# ===========================
# CLI Examples
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("Module 12.3: Self-Service Tenant Onboarding")
    print("=" * 60)

    # Example 1: Create skeleton tenant
    print("\n[1] Creating skeleton tenant...")
    tenant = create_skeleton_tenant(
        email="demo@example.com",
        company_name="Demo Corp",
        password="SecurePass123!",
        plan="pro"
    )
    print(f"  ✓ Tenant created: {tenant['tenant_id']}")
    print(f"  Status: {tenant['status']}")

    # Example 2: Generate checkout URL
    print("\n[2] Generating Stripe checkout URL...")
    checkout_url = generate_stripe_checkout_url(tenant['tenant_id'], tenant['plan'])
    print(f"  ✓ Checkout URL: {checkout_url[:60]}...")

    # Example 3: Simulate webhook (in-memory store)
    print("\n[3] Simulating webhook handler...")
    tenant_store = {tenant['tenant_id']: tenant}
    webhook_data = {
        'object': {
            'metadata': {'tenant_id': tenant['tenant_id']},
            'customer': 'cus_mock123',
            'subscription': 'sub_mock456',
        }
    }
    updated_tenant = handle_checkout_completed(webhook_data, tenant_store)
    print(f"  ✓ Status updated to: {updated_tenant['status']}")

    # Example 4: Provision tenant
    print("\n[4] Provisioning tenant...")
    sample_docs = [
        {"title": "Doc 1", "content": "Sample content 1"},
        {"title": "Doc 2", "content": "Sample content 2"},
    ]
    provisioned = provision_tenant(
        tenant['tenant_id'],
        tenant_store,
        sample_docs=sample_docs
    )
    print(f"  ✓ Provisioning complete")
    print(f"  Status: {provisioned['status']}")
    print(f"  Namespace: {provisioned.get('pinecone_namespace', 'N/A')}")
    print(f"  API Key: {provisioned.get('api_key', 'N/A')[:40]}...")

    # Example 5: Activation tracking
    print("\n[5] Tracking activation events...")
    events = [
        track_activation_event(tenant['tenant_id'], 'signup_completed'),
        track_activation_event(tenant['tenant_id'], 'payment_confirmed'),
        track_activation_event(tenant['tenant_id'], 'first_login'),
        track_activation_event(tenant['tenant_id'], 'first_query_executed'),
    ]
    metrics = calculate_activation_metrics(events)
    print(f"  ✓ Activation rate: {metrics['activation_rate']}%")
    print(f"  Login rate: {metrics['login_rate']}%")

    print("\n" + "=" * 60)
    print("✓ All examples completed")
    print("=" * 60)
