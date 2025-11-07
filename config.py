"""
Configuration module for Conversational RAG with Memory.

Loads environment variables and provides client initialization.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the module."""

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_SESSION_TTL: int = int(os.getenv("REDIS_SESSION_TTL", "604800"))  # 7 days

    # Memory Configuration
    SHORT_TERM_BUFFER_SIZE: int = int(os.getenv("SHORT_TERM_BUFFER_SIZE", "5"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
    SUMMARY_MODEL: str = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

    # Default LLM for queries
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    # spaCy model
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")


def get_openai_client():
    """Initialize OpenAI client."""
    if not Config.OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=Config.OPENAI_API_KEY)
    except ImportError:
        return None


def get_anthropic_client():
    """Initialize Anthropic client."""
    if not Config.ANTHROPIC_API_KEY:
        return None

    try:
        import anthropic
        return anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    except ImportError:
        return None


def get_redis_client():
    """Initialize Redis client."""
    try:
        import redis
        return redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=True
        )
    except ImportError:
        return None
    except Exception:
        return None


def get_clients() -> Dict[str, Any]:
    """
    Get all initialized clients.

    Returns:
        Dictionary of available clients
    """
    return {
        "openai": get_openai_client(),
        "anthropic": get_anthropic_client(),
        "redis": get_redis_client(),
    }


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration and return status.

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    if not Config.OPENAI_API_KEY and not Config.ANTHROPIC_API_KEY:
        warnings.append("No LLM API key configured")

    redis_client = get_redis_client()
    if redis_client is None:
        warnings.append("Redis not available - session persistence disabled")
    else:
        try:
            redis_client.ping()
        except Exception:
            warnings.append("Redis connection failed - check REDIS_HOST and REDIS_PORT")

    return len(warnings) == 0, warnings
