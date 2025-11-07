"""
Configuration module for M12.3 Self-Service Tenant Onboarding.
Loads environment variables and provides client accessors.
"""
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ===========================
# Environment Variables
# ===========================

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tenants.db")

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID_STARTER = os.getenv("STRIPE_PRICE_ID_STARTER", "price_starter")
STRIPE_PRICE_ID_PRO = os.getenv("STRIPE_PRICE_ID_PRO", "price_pro")
STRIPE_PRICE_ID_ENTERPRISE = os.getenv("STRIPE_PRICE_ID_ENTERPRISE", "price_enterprise")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "saas-tenants")

# Redis/Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Email (SendGrid)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "onboarding@example.com")
SENDGRID_FROM_NAME = os.getenv("SENDGRID_FROM_NAME", "SaaS Platform")

# Application
APP_URL = os.getenv("APP_URL", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "720"))  # 30 days

# Provisioning defaults
PROVISIONING_TIMEOUT_SECONDS = int(os.getenv("PROVISIONING_TIMEOUT_SECONDS", "300"))  # 5 minutes
MAX_PROVISIONING_RETRIES = int(os.getenv("MAX_PROVISIONING_RETRIES", "3"))
SAMPLE_DATA_ENABLED = os.getenv("SAMPLE_DATA_ENABLED", "true").lower() == "true"

# Rate limiting
SIGNUP_RATE_LIMIT_PER_HOUR = int(os.getenv("SIGNUP_RATE_LIMIT_PER_HOUR", "10"))

# ===========================
# Client Accessors
# ===========================

def get_stripe_client():
    """Get configured Stripe client."""
    if not STRIPE_SECRET_KEY:
        logger.warning("Stripe secret key not configured")
        return None

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        return stripe
    except ImportError:
        logger.error("Stripe library not installed")
        return None


def get_pinecone_client():
    """Get configured Pinecone client."""
    if not PINECONE_API_KEY:
        logger.warning("Pinecone API key not configured")
        return None

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc
    except ImportError:
        logger.error("Pinecone library not installed")
        return None
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return None


def get_redis_client():
    """Get configured Redis client."""
    try:
        import redis
        return redis.from_url(REDIS_URL)
    except ImportError:
        logger.warning("Redis library not installed")
        return None
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return None


def get_sendgrid_client():
    """Get configured SendGrid client."""
    if not SENDGRID_API_KEY:
        logger.warning("SendGrid API key not configured")
        return None

    try:
        from sendgrid import SendGridAPIClient
        return SendGridAPIClient(SENDGRID_API_KEY)
    except ImportError:
        logger.error("SendGrid library not installed")
        return None


def get_clients() -> Dict[str, Any]:
    """
    Get all configured clients.
    Returns dict with 'stripe', 'pinecone', 'redis', 'sendgrid' keys.
    """
    return {
        "stripe": get_stripe_client(),
        "pinecone": get_pinecone_client(),
        "redis": get_redis_client(),
        "sendgrid": get_sendgrid_client(),
    }


def verify_config() -> Dict[str, bool]:
    """
    Verify configuration completeness.
    Returns dict of service availability.
    """
    return {
        "stripe": bool(STRIPE_SECRET_KEY),
        "pinecone": bool(PINECONE_API_KEY),
        "redis": bool(REDIS_URL),
        "sendgrid": bool(SENDGRID_API_KEY),
        "database": bool(DATABASE_URL),
    }


if __name__ == "__main__":
    # Quick config check
    logging.basicConfig(level=logging.INFO)
    config_status = verify_config()
    print("Configuration Status:")
    for service, available in config_status.items():
        status = "✓" if available else "✗"
        print(f"  {status} {service}")
