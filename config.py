"""
Module 7.2: Application Performance Monitoring Configuration
Manages Datadog APM, OpenTelemetry, and profiling settings
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class APMConfig:
    """APM configuration for Datadog"""

    # Datadog credentials
    DD_API_KEY: str = os.getenv("DD_API_KEY", "")
    DD_SITE: str = os.getenv("DD_SITE", "datadoghq.com")  # or datadoghq.eu
    DD_SERVICE: str = os.getenv("DD_SERVICE", "compliance-copilot-rag")
    DD_ENV: str = os.getenv("DD_ENV", "production")
    DD_VERSION: str = os.getenv("DD_VERSION", "1.0.0")

    # APM Configuration
    DD_PROFILING_ENABLED: bool = os.getenv("DD_PROFILING_ENABLED", "true").lower() == "true"
    DD_PROFILING_CAPTURE_PCT: int = int(os.getenv("DD_PROFILING_CAPTURE_PCT", "1"))  # Profile 1% of requests
    DD_TRACE_SAMPLE_RATE: float = float(os.getenv("DD_TRACE_SAMPLE_RATE", "0.1"))  # 10% trace sampling
    DD_TRACE_ANALYTICS_ENABLED: bool = os.getenv("DD_TRACE_ANALYTICS_ENABLED", "true").lower() == "true"

    # Performance overhead limits
    DD_PROFILING_MAX_TIME_USAGE_PCT: float = float(os.getenv("DD_PROFILING_MAX_TIME_USAGE_PCT", "5.0"))
    DD_PROFILING_MEMORY_ENABLED: bool = os.getenv("DD_PROFILING_MEMORY_ENABLED", "true").lower() == "true"
    DD_PROFILING_CPU_ENABLED: bool = os.getenv("DD_PROFILING_CPU_ENABLED", "true").lower() == "true"

    # Query profiling (for DB queries)
    DD_TRACE_DATABASE_ENABLED: bool = os.getenv("DD_TRACE_DATABASE_ENABLED", "true").lower() == "true"
    DD_TRACE_ANALYTICS_SAMPLE_RATE: float = float(os.getenv("DD_TRACE_ANALYTICS_SAMPLE_RATE", "0.5"))

    def __post_init__(self):
        """Validate configuration"""
        if not self.DD_API_KEY:
            logger.warning("DD_API_KEY not set - APM will be disabled")

        # Ensure sampling is production-safe
        if self.DD_PROFILING_CAPTURE_PCT > 5:
            logger.warning(
                f"⚠️  WARNING: Profiling {self.DD_PROFILING_CAPTURE_PCT}% may impact performance. "
                f"Recommended: ≤5% for production"
            )

        if self.DD_TRACE_SAMPLE_RATE > 0.2:
            logger.warning(
                f"⚠️  WARNING: Sampling {self.DD_TRACE_SAMPLE_RATE * 100}% may increase costs. "
                f"Recommended: ≤20% for production"
            )

    @property
    def is_configured(self) -> bool:
        """Check if APM is properly configured"""
        return bool(self.DD_API_KEY)


@dataclass
class OpenTelemetryConfig:
    """OpenTelemetry configuration (M7.1 prerequisite)"""

    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "compliance-copilot-rag")
    OTEL_TRACE_ENABLED: bool = os.getenv("OTEL_TRACE_ENABLED", "true").lower() == "true"


@dataclass
class VectorDBConfig:
    """Vector database configuration (Pinecone example)"""

    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "compliance-docs")

    @property
    def is_configured(self) -> bool:
        """Check if Pinecone is properly configured"""
        return bool(self.PINECONE_API_KEY)


@dataclass
class AppConfig:
    """Application configuration"""

    # General settings
    APP_NAME: str = "Module 7.2: Application Performance Monitoring"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Performance settings
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "10"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))


# Initialize global configs
apm_config = APMConfig()
otel_config = OpenTelemetryConfig()
vectordb_config = VectorDBConfig()
app_config = AppConfig()


def get_clients():
    """
    Initialize and return configured clients

    Returns:
        dict: Dictionary of initialized clients (APM, VectorDB, etc.)
    """
    clients = {
        "apm_enabled": apm_config.is_configured,
        "vectordb_enabled": vectordb_config.is_configured,
    }

    # Initialize Pinecone if configured
    if vectordb_config.is_configured:
        try:
            import pinecone
            # Note: Pinecone client initialization would go here in real implementation
            clients["pinecone"] = None  # Placeholder
            logger.info("✅ Pinecone client configured")
        except ImportError:
            logger.warning("Pinecone library not installed")
            clients["vectordb_enabled"] = False

    return clients


def configure_logging():
    """Configure structured logging"""
    logging.basicConfig(
        level=getattr(logging, app_config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ddtrace").setLevel(logging.WARNING)


# Configure logging on import
configure_logging()


if __name__ == "__main__":
    """Test configuration"""
    print("=== APM Configuration ===")
    print(f"Service: {apm_config.DD_SERVICE}")
    print(f"Environment: {apm_config.DD_ENV}")
    print(f"Profiling: {apm_config.DD_PROFILING_ENABLED}")
    print(f"Sample Rate: {apm_config.DD_TRACE_SAMPLE_RATE * 100}%")
    print(f"Configured: {apm_config.is_configured}")
    print()
    print("=== OpenTelemetry Configuration ===")
    print(f"Endpoint: {otel_config.OTEL_EXPORTER_OTLP_ENDPOINT}")
    print(f"Service: {otel_config.OTEL_SERVICE_NAME}")
    print()
    print("=== Clients ===")
    clients = get_clients()
    for key, value in clients.items():
        print(f"{key}: {value}")
