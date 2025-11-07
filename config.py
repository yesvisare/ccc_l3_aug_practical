"""
Configuration management for Multi-Agent Orchestration module.
Loads environment variables and provides client factories.
"""
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Application configuration singleton."""

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # LangChain tracing (optional)
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "multi-agent-orchestration")

    # Application settings
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "60"))
    MAX_COST_PER_QUERY: float = float(os.getenv("MAX_COST_PER_QUERY", "0.10"))

    @classmethod
    def is_configured(cls) -> bool:
        """Check if required API keys are configured."""
        return bool(cls.OPENAI_API_KEY and cls.OPENAI_API_KEY != "")

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            "openai_model": cls.OPENAI_MODEL,
            "openai_temperature": cls.OPENAI_TEMPERATURE,
            "max_iterations": cls.MAX_ITERATIONS,
            "timeout_seconds": cls.TIMEOUT_SECONDS,
            "max_cost_per_query": cls.MAX_COST_PER_QUERY,
            "tracing_enabled": cls.LANGCHAIN_TRACING_V2
        }


def get_openai_client():
    """
    Factory function to create OpenAI client.

    Returns:
        OpenAI client instance or None if not configured

    Raises:
        ValueError: If API key is not configured
    """
    if not Config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. Please set it in .env file."
        )

    try:
        from openai import OpenAI
        return OpenAI(api_key=Config.OPENAI_API_KEY)
    except ImportError:
        logger.error("OpenAI package not installed. Run: pip install openai")
        raise


def get_langchain_llm():
    """
    Factory function to create LangChain ChatOpenAI instance.

    Returns:
        ChatOpenAI instance or None if not configured

    Raises:
        ValueError: If API key is not configured
    """
    if not Config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. Please set it in .env file."
        )

    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    except ImportError:
        logger.error("langchain-openai package not installed. Run: pip install langchain-openai")
        raise


# Export logger for use in other modules
__all__ = ["Config", "get_openai_client", "get_langchain_llm", "logger"]
