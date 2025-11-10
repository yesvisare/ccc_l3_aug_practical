"""
Configuration module for Query Decomposition & Planning.

Loads environment variables and provides client initialization.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


class Config:
    """Configuration constants and defaults."""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model Configuration
    DECOMPOSITION_MODEL: str = os.getenv("DECOMPOSITION_MODEL", "gpt-4-turbo-preview")
    SYNTHESIS_MODEL: str = os.getenv("SYNTHESIS_MODEL", "gpt-4-turbo-preview")
    DECOMPOSITION_TEMPERATURE: float = float(os.getenv("DECOMPOSITION_TEMPERATURE", "0.0"))
    SYNTHESIS_TEMPERATURE: float = float(os.getenv("SYNTHESIS_TEMPERATURE", "0.3"))

    # Execution Limits
    MAX_SUB_QUERIES: int = int(os.getenv("MAX_SUB_QUERIES", "6"))
    MAX_CONCURRENT_RETRIEVALS: int = int(os.getenv("MAX_CONCURRENT_RETRIEVALS", "5"))
    RETRIEVAL_TIMEOUT_SEC: int = int(os.getenv("RETRIEVAL_TIMEOUT_SEC", "30"))

    # Token Limits
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))

    # Thresholds
    COMPLEXITY_THRESHOLD: float = float(os.getenv("COMPLEXITY_THRESHOLD", "0.7"))
    MIN_LATENCY_BUDGET_MS: int = int(os.getenv("MIN_LATENCY_BUDGET_MS", "700"))

    # Fallback Configuration
    ENABLE_FALLBACK: bool = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"

    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not cls.OPENAI_API_KEY:
            return False
        return True

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Dictionary of configuration values
        """
        return {
            "decomposition_model": cls.DECOMPOSITION_MODEL,
            "synthesis_model": cls.SYNTHESIS_MODEL,
            "max_sub_queries": cls.MAX_SUB_QUERIES,
            "max_concurrent_retrievals": cls.MAX_CONCURRENT_RETRIEVALS,
            "retrieval_timeout_sec": cls.RETRIEVAL_TIMEOUT_SEC,
            "max_context_tokens": cls.MAX_CONTEXT_TOKENS,
            "enable_fallback": cls.ENABLE_FALLBACK,
            "enable_metrics": cls.ENABLE_METRICS,
        }


def get_openai_client() -> Optional[AsyncOpenAI]:
    """
    Get configured OpenAI async client.

    Returns:
        AsyncOpenAI client if API key is available, None otherwise
    """
    if not Config.OPENAI_API_KEY:
        return None

    return AsyncOpenAI(api_key=Config.OPENAI_API_KEY)


def get_clients() -> Dict[str, Any]:
    """
    Get all configured SDK clients.

    Returns:
        Dictionary mapping client names to initialized clients
    """
    clients = {}

    openai_client = get_openai_client()
    if openai_client:
        clients["openai"] = openai_client

    return clients


# Module-level defaults
DEFAULT_DECOMPOSITION_PARAMS = {
    "model": Config.DECOMPOSITION_MODEL,
    "temperature": Config.DECOMPOSITION_TEMPERATURE,
    "max_sub_queries": Config.MAX_SUB_QUERIES,
}

DEFAULT_EXECUTION_PARAMS = {
    "max_concurrent": Config.MAX_CONCURRENT_RETRIEVALS,
    "timeout": Config.RETRIEVAL_TIMEOUT_SEC,
}

DEFAULT_SYNTHESIS_PARAMS = {
    "model": Config.SYNTHESIS_MODEL,
    "temperature": Config.SYNTHESIS_TEMPERATURE,
    "max_tokens": Config.MAX_CONTEXT_TOKENS,
}


if __name__ == "__main__":
    print("Configuration Status:")
    print(f"  OpenAI API Key: {'✓ Present' if Config.OPENAI_API_KEY else '✗ Missing'}")
    print(f"  Configuration Valid: {Config.validate()}")
    print("\nConfiguration Values:")
    for key, value in Config.get_config_dict().items():
        print(f"  {key}: {value}")
