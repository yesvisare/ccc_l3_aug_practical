"""
Configuration management for Module 10.2: Tool Calling & Function Execution

Reads environment variables and provides configuration constants.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "agentic_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Slack Configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Vector Database (Pinecone/Weaviate/etc.)
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY", "")
VECTOR_DB_ENDPOINT = os.getenv("VECTOR_DB_ENDPOINT", "")

# External API Configuration
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")
EXTERNAL_API_BASE_URL = os.getenv("EXTERNAL_API_BASE_URL", "https://api.example.com")


# ============================================================================
# TOOL EXECUTION DEFAULTS
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_COUNT = 2
MAX_CONCURRENT_TOOLS = 5
SANDBOX_ENABLED = True


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

MAX_AGENT_ITERATIONS = 10
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.0"))


# ============================================================================
# API RATE LIMITING
# ============================================================================

API_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))
API_RATE_LIMIT_PER_HOUR = int(os.getenv("API_RATE_LIMIT_PER_HOUR", "1000"))


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "tool_execution.log")


# ============================================================================
# CLIENT BUILDERS
# ============================================================================

def get_database_config() -> Dict[str, Any]:
    """
    Get database connection configuration.

    Returns:
        Dict with database connection parameters
    """
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "database": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD
    }


def get_slack_client():
    """
    Get Slack client if credentials available.

    Returns:
        Slack WebClient or None
    """
    if not SLACK_BOT_TOKEN:
        return None

    try:
        from slack_sdk import WebClient
        return WebClient(token=SLACK_BOT_TOKEN)
    except ImportError:
        return None


def get_openai_client():
    """
    Get OpenAI client if API key available.

    Returns:
        OpenAI client or None
    """
    if not OPENAI_API_KEY:
        return None

    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        return openai
    except ImportError:
        return None


def is_configured(service: str) -> bool:
    """
    Check if a service is properly configured.

    Args:
        service: Service name (database, slack, openai, vector_db)

    Returns:
        True if configured with necessary credentials
    """
    service_checks = {
        "database": bool(DB_PASSWORD),
        "slack": bool(SLACK_BOT_TOKEN or SLACK_WEBHOOK_URL),
        "openai": bool(OPENAI_API_KEY),
        "vector_db": bool(VECTOR_DB_API_KEY and VECTOR_DB_ENDPOINT)
    }

    return service_checks.get(service, False)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> Dict[str, bool]:
    """
    Validate configuration and return status.

    Returns:
        Dict mapping service names to configuration status
    """
    return {
        "database": is_configured("database"),
        "slack": is_configured("slack"),
        "openai": is_configured("openai"),
        "vector_db": is_configured("vector_db")
    }


if __name__ == "__main__":
    """Print configuration status."""
    print("=== Configuration Status ===\n")

    status = validate_config()
    for service, configured in status.items():
        icon = "✅" if configured else "⚠️"
        print(f"{icon} {service.capitalize()}: {'Configured' if configured else 'Not configured'}")

    print("\nDefaults:")
    print(f"  Max Timeout: {DEFAULT_TIMEOUT_SECONDS}s")
    print(f"  Max Retries: {DEFAULT_RETRY_COUNT}")
    print(f"  Max Concurrent Tools: {MAX_CONCURRENT_TOOLS}")
    print(f"  Sandbox Enabled: {SANDBOX_ENABLED}")
