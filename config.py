"""
Configuration management for Module 11.2: Tenant-Specific Customization
Loads environment variables and provides client setup for database and cache.
"""
import os
import logging
from typing import Optional, Tuple, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Application configuration from environment variables."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/tenant_config_db")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "300"))

    # Application
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # API Keys (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Default Tenant Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_ALPHA: float = float(os.getenv("DEFAULT_ALPHA", "0.5"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "500"))

    # Resource Limits
    MAX_TEMPERATURE: float = float(os.getenv("MAX_TEMPERATURE", "2.0"))
    MIN_TEMPERATURE: float = float(os.getenv("MIN_TEMPERATURE", "0.0"))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "20"))
    MIN_TOP_K: int = int(os.getenv("MIN_TOP_K", "1"))

    # Approved models whitelist
    APPROVED_MODELS: list = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]


def get_database_client() -> Optional[Any]:
    """
    Get database connection/engine.
    Returns None if connection fails.
    """
    try:
        from sqlalchemy import create_engine
        engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
        logger.info("Database client initialized")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database client: {e}")
        return None


def get_redis_client() -> Optional[Any]:
    """
    Get Redis client.
    Returns None if connection fails.
    """
    try:
        import redis
        client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        client.ping()
        logger.info("Redis client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
        return None


def get_clients() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize and return both database and Redis clients.

    Returns:
        Tuple of (db_engine, redis_client). Either can be None if unavailable.
    """
    db_engine = get_database_client()
    redis_client = get_redis_client()
    return db_engine, redis_client


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
