"""
Configuration management for M12.2 Billing Integration.
Loads environment variables and provides client initialization.
"""

import os
from typing import Optional
from dotenv import load_dotenv
import stripe

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for billing integration"""

    # Stripe Configuration
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # Billing Defaults
    DEFAULT_TRIAL_DAYS: int = int(os.getenv("DEFAULT_TRIAL_DAYS", "14"))
    DEFAULT_PLAN: str = os.getenv("DEFAULT_PLAN", "pro")

    # Dunning Configuration
    DUNNING_RETRY_DAYS: list = [1, 4, 7, 8]  # Days when actions are taken
    MAX_PAYMENT_FAILURES: int = int(os.getenv("MAX_PAYMENT_FAILURES", "4"))

    # Database Configuration (optional - for full implementation)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT: int = int(os.getenv("CLICKHOUSE_PORT", "9000"))

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    @classmethod
    def is_stripe_configured(cls) -> bool:
        """Check if Stripe is properly configured"""
        return bool(cls.STRIPE_SECRET_KEY and cls.STRIPE_SECRET_KEY.startswith("sk_"))

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not cls.STRIPE_SECRET_KEY:
            errors.append("STRIPE_SECRET_KEY not set")
        elif not cls.STRIPE_SECRET_KEY.startswith("sk_"):
            errors.append("STRIPE_SECRET_KEY must start with 'sk_'")

        if not cls.STRIPE_WEBHOOK_SECRET:
            errors.append("STRIPE_WEBHOOK_SECRET not set (optional for local dev)")

        return len(errors) == 0, errors


def get_stripe_client() -> Optional[stripe]:
    """
    Get initialized Stripe client.

    Returns:
        Stripe module with API key set, or None if not configured
    """
    if not Config.is_stripe_configured():
        return None

    stripe.api_key = Config.STRIPE_SECRET_KEY
    return stripe


def print_config_status():
    """Print configuration status for debugging"""
    print("Configuration Status")
    print("=" * 50)

    is_valid, errors = Config.validate()

    print(f"Stripe Configured: {Config.is_stripe_configured()}")
    print(f"Stripe Key: {Config.STRIPE_SECRET_KEY[:10]}..." if Config.STRIPE_SECRET_KEY else "Stripe Key: Not set")
    print(f"Webhook Secret: {'Set' if Config.STRIPE_WEBHOOK_SECRET else 'Not set'}")
    print(f"Default Trial Days: {Config.DEFAULT_TRIAL_DAYS}")
    print(f"Default Plan: {Config.DEFAULT_PLAN}")
    print(f"Max Payment Failures: {Config.MAX_PAYMENT_FAILURES}")

    print("\nValidation:")
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")

    print("=" * 50)


if __name__ == "__main__":
    print_config_status()
