"""
Configuration management for Module 11.3
Loads environment variables and provides client initialization.
"""

import os
import redis
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for Module 11.3"""

    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DECODE_RESPONSES: bool = True

    # Queue configuration
    MAX_QUEUE_SIZE_PER_TENANT: int = int(os.getenv("MAX_QUEUE_SIZE_PER_TENANT", "100"))
    MAX_TOTAL_QUEUE_SIZE: int = int(os.getenv("MAX_TOTAL_QUEUE_SIZE", "5000"))
    QUEUE_WORKERS: int = int(os.getenv("QUEUE_WORKERS", "5"))
    QUEUE_POLL_INTERVAL: float = float(os.getenv("QUEUE_POLL_INTERVAL", "1.0"))

    # Quota defaults
    FREE_TIER_HOURLY_LIMIT: int = int(os.getenv("FREE_TIER_HOURLY_LIMIT", "100"))
    PRO_TIER_HOURLY_LIMIT: int = int(os.getenv("PRO_TIER_HOURLY_LIMIT", "1000"))
    ENTERPRISE_TIER_HOURLY_LIMIT: int = int(os.getenv("ENTERPRISE_TIER_HOURLY_LIMIT", "10000"))

    # Admin configuration
    ADMIN_API_KEY: Optional[str] = os.getenv("ADMIN_API_KEY")

    # Throttling configuration
    ENABLE_QUEUING: bool = os.getenv("ENABLE_QUEUING", "true").lower() == "true"
    MAX_QUEUE_WAIT_SECONDS: int = int(os.getenv("MAX_QUEUE_WAIT_SECONDS", "30"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def get_redis_client() -> redis.Redis:
    """
    Get configured Redis client.

    Returns:
        redis.Redis: Configured Redis client

    Raises:
        redis.ConnectionError: If unable to connect to Redis
    """
    try:
        client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=Config.REDIS_DECODE_RESPONSES,
            socket_connect_timeout=5,
            socket_timeout=5
        )

        # Test connection
        client.ping()
        logger.info(f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        return client

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.error(f"Make sure Redis is running at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        raise


def configure_logging():
    """Configure logging based on config"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    # Test configuration
    print("=== Configuration Test ===\n")

    configure_logging()

    print("Configuration loaded:")
    print(f"  Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
    print(f"  Queue workers: {Config.QUEUE_WORKERS}")
    print(f"  Max queue size: {Config.MAX_TOTAL_QUEUE_SIZE}")
    print(f"  Enable queuing: {Config.ENABLE_QUEUING}")
    print(f"  Log level: {Config.LOG_LEVEL}")

    # Test Redis connection
    try:
        client = get_redis_client()
        print("\n✓ Redis connection successful")
        info = client.info("server")
        print(f"  Redis version: {info.get('redis_version', 'unknown')}")
    except Exception as e:
        print(f"\n⚠️ Redis connection failed: {e}")
        print("  Run: docker run -d -p 6379:6379 redis:7-alpine")
