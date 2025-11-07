"""
Configuration module for tenant isolation system.

Loads environment variables and provides client initialization.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration constants and environment variables."""

    # Database configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "tenant_isolation")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

    # Pinecone configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_SHARED_INDEX = os.getenv("PINECONE_SHARED_INDEX", "shared-index-1")

    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Capacity limits
    MAX_NAMESPACES_PER_INDEX = int(os.getenv("MAX_NAMESPACES_PER_INDEX", "90"))
    NAMESPACE_ALERT_THRESHOLD = float(os.getenv("NAMESPACE_ALERT_THRESHOLD", "0.8"))

    # Cost tracking defaults
    COST_PER_1K_EMBED_TOKENS = float(os.getenv("COST_PER_1K_EMBED_TOKENS", "0.0001"))
    COST_PER_1K_LLM_TOKENS = float(os.getenv("COST_PER_1K_LLM_TOKENS", "0.002"))
    COST_PER_QUERY_PINECONE = float(os.getenv("COST_PER_QUERY_PINECONE", "0.00001"))

    # Monthly fixed costs (USD)
    MONTHLY_PINECONE_BASE = float(os.getenv("MONTHLY_PINECONE_BASE", "50.0"))
    MONTHLY_POSTGRES_BASE = float(os.getenv("MONTHLY_POSTGRES_BASE", "30.0"))
    MONTHLY_MONITORING_BASE = float(os.getenv("MONTHLY_MONITORING_BASE", "35.0"))

    @classmethod
    def get_total_fixed_costs(cls) -> float:
        """Calculate total monthly fixed costs."""
        return cls.MONTHLY_PINECONE_BASE + cls.MONTHLY_POSTGRES_BASE + cls.MONTHLY_MONITORING_BASE


def get_clients() -> Dict[str, Any]:
    """
    Initialize and return external service clients.

    Returns clients only if API keys are available. Gracefully handles missing keys.

    Returns:
        Dict with optional clients (pinecone, openai, postgres)
    """
    clients = {}

    # Pinecone client
    if Config.PINECONE_API_KEY:
        try:
            # Note: Import only if key exists to avoid import errors
            from pinecone import Pinecone
            pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            clients["pinecone"] = pc
            logger.info("Pinecone client initialized")
        except ImportError:
            logger.warning("Pinecone library not installed, skipping client init")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
    else:
        logger.warning("PINECONE_API_KEY not set, skipping Pinecone client")

    # OpenAI client
    if Config.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            clients["openai"] = client
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("OpenAI library not installed, skipping client init")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    else:
        logger.warning("OPENAI_API_KEY not set, skipping OpenAI client")

    # PostgreSQL client
    if Config.POSTGRES_PASSWORD:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                dbname=Config.POSTGRES_DB,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD
            )
            clients["postgres"] = conn
            logger.info("PostgreSQL client initialized")
        except ImportError:
            logger.warning("psycopg2 library not installed, skipping PostgreSQL client")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL client: {e}")
    else:
        logger.warning("POSTGRES_PASSWORD not set, skipping PostgreSQL client")

    return clients


def validate_config() -> bool:
    """
    Validate that required configuration is present.

    Returns:
        bool: True if config is valid for production use
    """
    required = []
    warnings = []

    # Check critical settings
    if not Config.PINECONE_API_KEY:
        warnings.append("PINECONE_API_KEY not set")

    if not Config.OPENAI_API_KEY:
        warnings.append("OPENAI_API_KEY not set")

    if not Config.POSTGRES_PASSWORD:
        warnings.append("POSTGRES_PASSWORD not set")

    if warnings:
        logger.warning(f"Config validation warnings: {', '.join(warnings)}")
        return False

    logger.info("Config validation passed")
    return True


if __name__ == "__main__":
    print("=== Configuration Status ===\n")

    print("Environment Variables:")
    print(f"  POSTGRES_HOST: {Config.POSTGRES_HOST}")
    print(f"  POSTGRES_PORT: {Config.POSTGRES_PORT}")
    print(f"  POSTGRES_DB: {Config.POSTGRES_DB}")
    print(f"  PINECONE_API_KEY: {'✓ Set' if Config.PINECONE_API_KEY else '✗ Not set'}")
    print(f"  OPENAI_API_KEY: {'✓ Set' if Config.OPENAI_API_KEY else '✗ Not set'}")

    print(f"\nCapacity Limits:")
    print(f"  Max namespaces per index: {Config.MAX_NAMESPACES_PER_INDEX}")
    print(f"  Alert threshold: {Config.NAMESPACE_ALERT_THRESHOLD * 100}%")

    print(f"\nMonthly Fixed Costs:")
    print(f"  Pinecone base: ${Config.MONTHLY_PINECONE_BASE}")
    print(f"  PostgreSQL: ${Config.MONTHLY_POSTGRES_BASE}")
    print(f"  Monitoring: ${Config.MONTHLY_MONITORING_BASE}")
    print(f"  Total: ${Config.get_total_fixed_costs()}")

    print(f"\nValidation: {'✓ PASS' if validate_config() else '✗ WARNINGS'}")

    print("\nInitializing clients...")
    clients = get_clients()
    for name, client in clients.items():
        print(f"  {name}: ✓ Initialized")
