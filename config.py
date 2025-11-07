"""
Configuration module for M11.4 Vector Index Sharding.
Loads environment variables and provides client initialization.
"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration settings for vector index sharding."""

    # Pinecone settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Sharding configuration
    NUM_SHARDS: int = int(os.getenv("NUM_SHARDS", "4"))
    SHARD_PREFIX: str = os.getenv("SHARD_PREFIX", "tenant-shard")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "1536"))
    METRIC: str = os.getenv("METRIC", "cosine")

    # Performance thresholds
    MAX_VECTORS_PER_SHARD: int = int(os.getenv("MAX_VECTORS_PER_SHARD", "300000"))
    MAX_NAMESPACES_PER_SHARD: int = int(os.getenv("MAX_NAMESPACES_PER_SHARD", "20"))
    P95_LATENCY_THRESHOLD_MS: int = int(os.getenv("P95_LATENCY_THRESHOLD_MS", "500"))

    # Rebalancing settings
    REBALANCE_NAMESPACE_THRESHOLD: int = int(os.getenv("REBALANCE_NAMESPACE_THRESHOLD", "18"))


def get_clients() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """
    Initialize and return Pinecone, Redis, and OpenAI clients.

    Returns:
        Tuple of (pinecone_client, redis_client, openai_client) or None for missing keys
    """
    pinecone_client = None
    redis_client = None
    openai_client = None

    # Initialize Pinecone
    if Config.PINECONE_API_KEY:
        try:
            from pinecone import Pinecone
            pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
    else:
        logger.warning("PINECONE_API_KEY not set")

    # Initialize Redis
    try:
        import redis
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
            decode_responses=True
        )
        # Test connection
        redis_client.ping()
        logger.info("Redis client initialized")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None

    # Initialize OpenAI
    if Config.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    else:
        logger.warning("OPENAI_API_KEY not set")

    return pinecone_client, redis_client, openai_client


def has_required_services() -> bool:
    """Check if minimum required services are configured."""
    return bool(Config.PINECONE_API_KEY and Config.OPENAI_API_KEY)
