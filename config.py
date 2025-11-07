"""
Configuration management for Usage Metering & Analytics.
Loads environment variables and provides client connections.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for usage metering system."""

    # ClickHouse connection
    CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT: int = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    CLICKHOUSE_USER: str = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    CLICKHOUSE_DATABASE: str = os.getenv("CLICKHOUSE_DATABASE", "metering")

    # Buffering configuration (matches <5ms overhead requirement)
    BUFFER_SIZE: int = int(os.getenv("BUFFER_SIZE", "100"))
    FLUSH_INTERVAL_SECONDS: float = float(os.getenv("FLUSH_INTERVAL_SECONDS", "1.0"))

    # Fallback storage
    FALLBACK_FILE_PATH: str = os.getenv("FALLBACK_FILE_PATH", "./usage_events_fallback.jsonl")

    # Pricing configuration (per-unit costs)
    PRICING_CONFIG: Dict[str, float] = {
        "query": float(os.getenv("PRICE_PER_QUERY", "0.01")),
        "token_input": float(os.getenv("PRICE_PER_1K_INPUT_TOKENS", "0.003")),
        "token_output": float(os.getenv("PRICE_PER_1K_OUTPUT_TOKENS", "0.015")),
        "storage_gb": float(os.getenv("PRICE_PER_GB_STORAGE", "0.10")),
    }

    # Quota defaults
    DEFAULT_QUOTA_QUERIES_PER_DAY: int = int(os.getenv("DEFAULT_QUOTA_QUERIES", "1000"))
    DEFAULT_QUOTA_TOKENS_PER_DAY: int = int(os.getenv("DEFAULT_QUOTA_TOKENS", "100000"))

    # Retention policy (36 months as per script)
    RETENTION_MONTHS: int = int(os.getenv("RETENTION_MONTHS", "36"))

    # Grafana (optional)
    GRAFANA_URL: Optional[str] = os.getenv("GRAFANA_URL")
    GRAFANA_API_KEY: Optional[str] = os.getenv("GRAFANA_API_KEY")


def get_clickhouse_client():
    """
    Get ClickHouse client connection.
    Returns None if connection fails (graceful degradation).
    """
    try:
        from clickhouse_driver import Client

        client = Client(
            host=Config.CLICKHOUSE_HOST,
            port=Config.CLICKHOUSE_PORT,
            user=Config.CLICKHOUSE_USER,
            password=Config.CLICKHOUSE_PASSWORD,
            database=Config.CLICKHOUSE_DATABASE,
        )

        # Test connection
        client.execute("SELECT 1")
        return client
    except Exception as e:
        print(f"⚠️ ClickHouse connection failed: {e}")
        return None


def get_config() -> Dict[str, Any]:
    """Return configuration as dictionary for easy access."""
    return {
        "clickhouse": {
            "host": Config.CLICKHOUSE_HOST,
            "port": Config.CLICKHOUSE_PORT,
            "database": Config.CLICKHOUSE_DATABASE,
        },
        "buffering": {
            "buffer_size": Config.BUFFER_SIZE,
            "flush_interval": Config.FLUSH_INTERVAL_SECONDS,
        },
        "pricing": Config.PRICING_CONFIG,
        "quotas": {
            "queries_per_day": Config.DEFAULT_QUOTA_QUERIES_PER_DAY,
            "tokens_per_day": Config.DEFAULT_QUOTA_TOKENS_PER_DAY,
        },
        "retention_months": Config.RETENTION_MONTHS,
    }


if __name__ == "__main__":
    print("Configuration loaded:")
    import json
    print(json.dumps(get_config(), indent=2))
