"""
Configuration management for Module 9: HyDE

Reads from .env file and provides utility functions for initializing clients.
"""

import os
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hyde-index")

# Redis (optional, for caching)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Model configurations
HYDE_MODEL = os.getenv("HYDE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# HyDE parameters
HYDE_TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.3"))
HYDE_MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "200"))

# Retrieval parameters
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
HYDE_WEIGHT = float(os.getenv("HYDE_WEIGHT", "0.6"))
TRADITIONAL_WEIGHT = float(os.getenv("TRADITIONAL_WEIGHT", "0.4"))

# Cache settings
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour

# Query classification
HYDE_THRESHOLD = float(os.getenv("HYDE_THRESHOLD", "0.1"))

# Server settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


def validate_config() -> Dict[str, bool]:
    """
    Validate configuration and return status.

    Returns:
        Dict with validation results for each component
    """
    status = {
        "openai": OPENAI_API_KEY is not None,
        "pinecone": PINECONE_API_KEY is not None and PINECONE_INDEX_NAME is not None,
        "redis": True  # Redis is optional
    }

    if not status["openai"]:
        logger.warning("⚠️ OPENAI_API_KEY not set")

    if not status["pinecone"]:
        logger.warning("⚠️ Pinecone credentials not set (vector search will be skipped)")

    return status


def get_clients() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Initialize and return clients for OpenAI, Pinecone, and Redis.

    Returns:
        Tuple of (openai_client, pinecone_index, redis_client)
        Returns None for any client that cannot be initialized
    """
    from openai import OpenAI

    # OpenAI client
    openai_client = None
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✓ OpenAI client initialized")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize OpenAI client: {e}")

    # Pinecone client
    pinecone_index = None
    if PINECONE_API_KEY and PINECONE_INDEX_NAME:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"✓ Pinecone index initialized: {PINECONE_INDEX_NAME}")
        except ImportError:
            logger.warning("⚠️ Pinecone not installed. Install with: pip install pinecone-client")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize Pinecone: {e}")

    # Redis client (optional)
    redis_client = None
    try:
        import redis
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        # Test connection
        redis_client.ping()
        logger.info("✓ Redis client initialized")
    except ImportError:
        logger.debug("Redis not installed (optional)")
    except Exception as e:
        logger.debug(f"Redis not available (optional): {e}")

    return openai_client, pinecone_index, redis_client


def get_hyde_config() -> Dict[str, Any]:
    """
    Get HyDE configuration parameters.

    Returns:
        Dict with HyDE configuration
    """
    return {
        "model": HYDE_MODEL,
        "temperature": HYDE_TEMPERATURE,
        "max_tokens": HYDE_MAX_TOKENS,
        "embedding_model": EMBEDDING_MODEL,
        "hyde_weight": HYDE_WEIGHT,
        "traditional_weight": TRADITIONAL_WEIGHT,
        "hyde_threshold": HYDE_THRESHOLD,
        "default_top_k": DEFAULT_TOP_K,
        "cache_ttl": CACHE_TTL_SECONDS
    }


if __name__ == "__main__":
    print("Configuration Status")
    print("=" * 50)

    status = validate_config()
    for component, is_valid in status.items():
        symbol = "✓" if is_valid else "✗"
        print(f"{symbol} {component}: {'OK' if is_valid else 'Missing'}")

    print("\nHyDE Configuration:")
    config = get_hyde_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nAttempting to initialize clients...")
    clients = get_clients()
    print("Done!")
