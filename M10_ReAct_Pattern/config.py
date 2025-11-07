"""
Configuration module for ReAct Pattern Implementation.
Reads from .env and provides typed configuration access.
"""
import os
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class Config:
    """Configuration settings for the ReAct agent system."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Agent Configuration
    AGENT_MODEL: str = os.getenv("AGENT_MODEL", "gpt-4")
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
    AGENT_TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))

    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "level1-rag")

    # External API Configuration
    INDUSTRY_API_KEY: str = os.getenv("INDUSTRY_API_KEY", "")
    INDUSTRY_API_URL: str = os.getenv("INDUSTRY_API_URL", "https://api.benchmarks.com")

    # Application Settings
    ENABLE_AGENT: bool = os.getenv("ENABLE_AGENT", "true").lower() == "true"
    FALLBACK_TO_STATIC: bool = os.getenv("FALLBACK_TO_STATIC", "true").lower() == "true"

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration is present."""
        if not cls.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set")
            return False
        return True

    @classmethod
    def get_info(cls) -> dict:
        """Get configuration info (without sensitive data)."""
        return {
            "agent_model": cls.AGENT_MODEL,
            "agent_max_iterations": cls.AGENT_MAX_ITERATIONS,
            "agent_timeout_seconds": cls.AGENT_TIMEOUT_SECONDS,
            "enable_agent": cls.ENABLE_AGENT,
            "fallback_to_static": cls.FALLBACK_TO_STATIC,
            "has_openai_key": bool(cls.OPENAI_API_KEY),
            "has_pinecone_key": bool(cls.PINECONE_API_KEY),
        }


def get_openai_client():
    """Get configured OpenAI client."""
    try:
        from openai import OpenAI
        if not Config.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
            return None
        return OpenAI(api_key=Config.OPENAI_API_KEY)
    except ImportError:
        logger.error("OpenAI package not installed")
        return None


def get_pinecone_client():
    """Get configured Pinecone client."""
    try:
        import pinecone
        if not Config.PINECONE_API_KEY:
            logger.warning("Pinecone API key not configured")
            return None
        pinecone.init(
            api_key=Config.PINECONE_API_KEY,
            environment=Config.PINECONE_ENVIRONMENT
        )
        return pinecone.Index(Config.PINECONE_INDEX_NAME)
    except ImportError:
        logger.error("Pinecone package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        return None


# Validate configuration on import
if __name__ != "__main__":
    if not Config.validate():
        logger.warning("Configuration validation failed - some features may not work")
