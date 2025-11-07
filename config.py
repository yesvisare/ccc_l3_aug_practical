"""
Module 12.4: Tenant Lifecycle Management - Configuration
Loads environment variables and provides SDK clients.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration container for tenant lifecycle management."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/tenant_lifecycle_db")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Stripe
    STRIPE_API_KEY: Optional[str] = os.getenv("STRIPE_API_KEY")
    STRIPE_WEBHOOK_SECRET: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")

    # Lifecycle settings
    SOFT_DELETE_RETENTION_DAYS: int = int(os.getenv("SOFT_DELETE_RETENTION_DAYS", "30"))
    DATA_EXPORT_CHUNK_SIZE_MB: int = int(os.getenv("DATA_EXPORT_CHUNK_SIZE_MB", "50"))
    MAX_CONCURRENT_LIFECYCLE_JOBS: int = int(os.getenv("MAX_CONCURRENT_LIFECYCLE_JOBS", "10"))

    # Storage
    EXPORT_STORAGE_TYPE: str = os.getenv("EXPORT_STORAGE_TYPE", "local")
    EXPORT_STORAGE_PATH: str = os.getenv("EXPORT_STORAGE_PATH", "/tmp/tenant_exports")

    # AWS S3 (optional)
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_S3_BUCKET: Optional[str] = os.getenv("AWS_S3_BUCKET")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Feature flags
    ENABLE_AUTO_DOWNGRADE: bool = os.getenv("ENABLE_AUTO_DOWNGRADE", "true").lower() == "true"
    ENABLE_REACTIVATION_WORKFLOW: bool = os.getenv("ENABLE_REACTIVATION_WORKFLOW", "true").lower() == "true"
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"

    # Plan hierarchy
    PLAN_HIERARCHY = ["free", "starter", "professional", "enterprise"]

    # Plan limits (example defaults)
    PLAN_LIMITS = {
        "free": {"users": 5, "storage_gb": 1, "api_calls_per_day": 100},
        "starter": {"users": 20, "storage_gb": 10, "api_calls_per_day": 1000},
        "professional": {"users": 100, "storage_gb": 100, "api_calls_per_day": 10000},
        "enterprise": {"users": -1, "storage_gb": -1, "api_calls_per_day": -1}  # -1 = unlimited
    }


def get_stripe_client():
    """
    Get configured Stripe client.

    Returns:
        stripe module with API key set, or None if not configured
    """
    if not Config.STRIPE_API_KEY:
        return None

    try:
        import stripe
        stripe.api_key = Config.STRIPE_API_KEY
        return stripe
    except ImportError:
        return None


def get_redis_client():
    """
    Get configured Redis client.

    Returns:
        Redis client or None if connection fails
    """
    try:
        import redis
        client = redis.from_url(Config.REDIS_URL)
        # Test connection
        client.ping()
        return client
    except Exception:
        return None


def get_database_engine():
    """
    Get SQLAlchemy database engine.

    Returns:
        SQLAlchemy engine or None if connection fails
    """
    try:
        from sqlalchemy import create_engine
        engine = create_engine(Config.DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return engine
    except Exception:
        return None


def get_clients() -> Dict[str, Any]:
    """
    Get all configured service clients.

    Returns:
        Dictionary of available clients
    """
    return {
        "stripe": get_stripe_client(),
        "redis": get_redis_client(),
        "database": get_database_engine()
    }


# Export config instance
config = Config()
