"""
Configuration management for Enterprise RAG SaaS.

Handles environment variables, client initialization, and system constants.
Supports graceful degradation when services are unavailable.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Environment Variables
# ============================================================================

class Config:
    """Centralized configuration from environment variables."""

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Service Endpoints
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "compliance-copilot")

    # System Defaults
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    DEFAULT_RETRIEVAL_MODE: str = os.getenv("DEFAULT_RETRIEVAL_MODE", "basic")
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))

    # Resource Limits
    MAX_QUERIES_PER_HOUR: int = int(os.getenv("MAX_QUERIES_PER_HOUR", "100"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    MAX_DOCUMENTS_PER_TENANT: int = int(os.getenv("MAX_DOCUMENTS_PER_TENANT", "10000"))
    QUERY_TIMEOUT_SECONDS: float = float(os.getenv("QUERY_TIMEOUT_SECONDS", "30.0"))

    # Database (PostgreSQL for tenant configs)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # Observability
    OTLP_ENDPOINT: Optional[str] = os.getenv("OTLP_ENDPOINT")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Application Settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def validate(cls) -> Dict[str, bool]:
        """
        Check which services are available.
        Returns dict of service -> availability status.
        """
        return {
            "openai": cls.OPENAI_API_KEY is not None,
            "pinecone": cls.PINECONE_API_KEY is not None,
            "anthropic": cls.ANTHROPIC_API_KEY is not None,
            "database": cls.DATABASE_URL is not None,
            "telemetry": cls.OTLP_ENDPOINT is not None
        }

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.ENVIRONMENT == "production"


# ============================================================================
# Client Initialization
# ============================================================================

def get_openai_client():
    """
    Initialize OpenAI client with error handling.
    Returns None if API key not configured.
    """
    if not Config.OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured")
        return None

    try:
        import openai
        openai.api_key = Config.OPENAI_API_KEY
        logger.info("OpenAI client initialized")
        return openai
    except ImportError:
        logger.warning("openai package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


def get_pinecone_client():
    """
    Initialize Pinecone client with error handling.
    Returns None if API key not configured.
    """
    if not Config.PINECONE_API_KEY:
        logger.warning("Pinecone API key not configured")
        return None

    try:
        import pinecone
        pinecone.init(
            api_key=Config.PINECONE_API_KEY,
            environment=Config.PINECONE_ENVIRONMENT
        )
        logger.info("Pinecone client initialized")
        return pinecone
    except ImportError:
        logger.warning("pinecone-client package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {e}")
        return None


def get_database_connection():
    """
    Initialize database connection (PostgreSQL).
    Returns None if DATABASE_URL not configured.
    """
    if not Config.DATABASE_URL:
        logger.warning("Database URL not configured")
        return None

    try:
        import psycopg2
        conn = psycopg2.connect(Config.DATABASE_URL)
        logger.info("Database connection established")
        return conn
    except ImportError:
        logger.warning("psycopg2 package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


def get_clients() -> Dict[str, Any]:
    """
    Initialize all available clients.

    Returns:
        Dictionary with client instances (None for unavailable services)
    """
    return {
        "openai": get_openai_client(),
        "pinecone": get_pinecone_client(),
        "database": get_database_connection()
    }


# ============================================================================
# System Constants
# ============================================================================

# Cost estimates (USD per 1K tokens)
COST_PER_1K_TOKENS = {
    "gpt-3.5-turbo": 0.002,
    "gpt-4": 0.03,
    "gpt-4-turbo": 0.01
}

# Monthly infrastructure cost breakdown (example)
MONTHLY_COST_BREAKDOWN = {
    "database": {"min": 50, "max": 200, "unit": "USD"},
    "vector_store": {"min": 70, "max": 500, "unit": "USD"},
    "llm_api": {"min": 100, "max": 2000, "unit": "USD"},
    "observability": {"min": 50, "max": 300, "unit": "USD"}
}

# Performance benchmarks
LATENCY_TARGETS = {
    "p50": 300,  # ms
    "p95": 500,  # ms
    "p99": 1000  # ms
}

# Scaling thresholds
SCALING_THRESHOLDS = {
    "min_customers": 5,
    "max_customers": 100,
    "min_market_size": 100,
    "max_latency_ms": 500
}


def print_config_status():
    """Print current configuration status for debugging."""
    print("=" * 60)
    print("Configuration Status")
    print("=" * 60)

    status = Config.validate()
    for service, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {service.capitalize()}: {'Available' if available else 'Not configured'}")

    print(f"\nEnvironment: {Config.ENVIRONMENT}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Log Level: {Config.LOG_LEVEL}")
    print("=" * 60)


if __name__ == "__main__":
    # Quick config check
    logging.basicConfig(level=logging.INFO)
    print_config_status()

    # Attempt to initialize clients
    clients = get_clients()
    print(f"\nInitialized clients: {sum(1 for c in clients.values() if c is not None)}/{len(clients)}")
