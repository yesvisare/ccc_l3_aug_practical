"""
Configuration management for Module 13.3: Launch Preparation & Marketing

Handles environment variables and provides default constants for:
- Analytics service credentials (Google Analytics, Mixpanel)
- Marketing automation tools (Mailchimp, HubSpot)
- Default pricing and conversion benchmarks
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for launch and marketing services"""

    # Analytics
    GOOGLE_ANALYTICS_MEASUREMENT_ID: Optional[str] = os.getenv('GOOGLE_ANALYTICS_MEASUREMENT_ID')
    MIXPANEL_PROJECT_TOKEN: Optional[str] = os.getenv('MIXPANEL_PROJECT_TOKEN')
    MIXPANEL_API_SECRET: Optional[str] = os.getenv('MIXPANEL_API_SECRET')

    # Marketing Automation
    MAILCHIMP_API_KEY: Optional[str] = os.getenv('MAILCHIMP_API_KEY')
    MAILCHIMP_SERVER_PREFIX: Optional[str] = os.getenv('MAILCHIMP_SERVER_PREFIX')
    HUBSPOT_API_KEY: Optional[str] = os.getenv('HUBSPOT_API_KEY')

    # Landing Page / Domain
    LANDING_PAGE_URL: str = os.getenv('LANDING_PAGE_URL', 'https://yourcompany.com')
    PRODUCT_APP_URL: str = os.getenv('PRODUCT_APP_URL', 'https://app.yourcompany.com')

    # Default Pricing Constants (from script)
    DEFAULT_HOURLY_LABOR_COST: float = 100.0
    DEFAULT_VALUE_CAPTURE_RATE: float = 0.25  # Capture 25% of value delivered
    DEFAULT_TARGET_GROSS_MARGIN: float = 0.67  # 67% gross margin target

    # Conversion Benchmarks (from script)
    BENCHMARK_VISITOR_TO_SIGNUP: float = 0.05  # 5% target
    BENCHMARK_SIGNUP_TO_ACTIVATED: float = 0.50  # 50% target
    BENCHMARK_ACTIVATED_TO_PAID: float = 0.20  # 20% target

    # CAC/LTV Targets
    TARGET_LTV_CAC_RATIO: float = 3.0  # 3:1 minimum for healthy unit economics
    TARGET_PAYBACK_PERIOD_MONTHS: int = 12  # Recover CAC within 12 months

    # Marketing Channel Budgets (monthly, in USD)
    DEFAULT_LINKEDIN_ADS_BUDGET: float = 1500.0
    DEFAULT_GOOGLE_ADS_BUDGET: float = 1000.0
    DEFAULT_CONTENT_MARKETING_BUDGET: float = 500.0

    # Default Pricing Tiers (from script examples)
    PRICING_TIERS: Dict[str, Dict[str, Any]] = {
        'starter': {
            'monthly_price': 199,
            'query_limit': 500,
            'document_limit': 5000,
            'user_limit': 2
        },
        'professional': {
            'monthly_price': 499,
            'query_limit': 2500,
            'document_limit': 25000,
            'user_limit': 10
        },
        'enterprise': {
            'monthly_price': 1499,
            'query_limit': 999999,
            'document_limit': 999999,
            'user_limit': 999
        }
    }


def get_clients() -> Dict[str, Any]:
    """
    Initialize and return client connections for external services

    Returns:
        Dictionary of initialized clients (or None if credentials missing)
    """
    clients = {
        'google_analytics': None,
        'mixpanel': None,
        'mailchimp': None,
        'hubspot': None
    }

    # Google Analytics (GA4)
    if Config.GOOGLE_ANALYTICS_MEASUREMENT_ID:
        # Note: GA4 typically uses gtag.js client-side, not a Python SDK
        # For server-side tracking, use Measurement Protocol
        logger.info("Google Analytics configured (client-side tracking)")
        clients['google_analytics'] = {
            'measurement_id': Config.GOOGLE_ANALYTICS_MEASUREMENT_ID,
            'type': 'client_side'
        }
    else:
        logger.warning("Google Analytics not configured - set GOOGLE_ANALYTICS_MEASUREMENT_ID")

    # Mixpanel
    if Config.MIXPANEL_PROJECT_TOKEN:
        try:
            # Mixpanel SDK would be initialized here if installed
            # from mixpanel import Mixpanel
            # clients['mixpanel'] = Mixpanel(Config.MIXPANEL_PROJECT_TOKEN)
            logger.info("Mixpanel token configured")
            clients['mixpanel'] = {
                'token': Config.MIXPANEL_PROJECT_TOKEN,
                'type': 'sdk_required'
            }
        except ImportError:
            logger.warning("Mixpanel SDK not installed - run: pip install mixpanel")
    else:
        logger.warning("Mixpanel not configured - set MIXPANEL_PROJECT_TOKEN")

    # Mailchimp
    if Config.MAILCHIMP_API_KEY and Config.MAILCHIMP_SERVER_PREFIX:
        try:
            # Mailchimp SDK would be initialized here if installed
            # import mailchimp_marketing as MailchimpMarketing
            # client = MailchimpMarketing.Client()
            # client.set_config({
            #     "api_key": Config.MAILCHIMP_API_KEY,
            #     "server": Config.MAILCHIMP_SERVER_PREFIX
            # })
            # clients['mailchimp'] = client
            logger.info("Mailchimp configured")
            clients['mailchimp'] = {
                'api_key': Config.MAILCHIMP_API_KEY[:10] + '...',
                'server': Config.MAILCHIMP_SERVER_PREFIX,
                'type': 'sdk_required'
            }
        except ImportError:
            logger.warning("Mailchimp SDK not installed - run: pip install mailchimp-marketing")
    else:
        logger.warning("Mailchimp not configured - set MAILCHIMP_API_KEY and MAILCHIMP_SERVER_PREFIX")

    # HubSpot
    if Config.HUBSPOT_API_KEY:
        try:
            # HubSpot SDK would be initialized here if installed
            # from hubspot import HubSpot
            # clients['hubspot'] = HubSpot(access_token=Config.HUBSPOT_API_KEY)
            logger.info("HubSpot configured")
            clients['hubspot'] = {
                'api_key': Config.HUBSPOT_API_KEY[:10] + '...',
                'type': 'sdk_required'
            }
        except ImportError:
            logger.warning("HubSpot SDK not installed - run: pip install hubspot-api-client")
    else:
        logger.warning("HubSpot not configured - set HUBSPOT_API_KEY")

    return clients


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration completeness

    Returns:
        Tuple of (is_valid, list of missing configs)
    """
    missing = []

    # Optional but recommended
    if not Config.GOOGLE_ANALYTICS_MEASUREMENT_ID:
        missing.append("GOOGLE_ANALYTICS_MEASUREMENT_ID (recommended for conversion tracking)")

    if not Config.MIXPANEL_PROJECT_TOKEN:
        missing.append("MIXPANEL_PROJECT_TOKEN (recommended for funnel analysis)")

    # URLs should be set
    if Config.LANDING_PAGE_URL == 'https://yourcompany.com':
        missing.append("LANDING_PAGE_URL (set your actual landing page URL)")

    if Config.PRODUCT_APP_URL == 'https://app.yourcompany.com':
        missing.append("PRODUCT_APP_URL (set your actual product app URL)")

    is_valid = len(missing) == 0

    if not is_valid:
        logger.info(f"Configuration incomplete - missing: {len(missing)} items")
        logger.info("This module can run without external services for calculations and planning")
    else:
        logger.info("✅ Configuration complete")

    return is_valid, missing


if __name__ == "__main__":
    print("=== Configuration Check ===\n")

    is_valid, missing = validate_config()

    print(f"Configuration valid: {is_valid}\n")

    if missing:
        print("Missing configurations:")
        for item in missing:
            print(f"  - {item}")
        print()

    print("Available clients:")
    clients = get_clients()
    for name, client in clients.items():
        status = "✅ Configured" if client else "⚠️  Not configured"
        print(f"  {name}: {status}")

    print("\nDefault Constants:")
    print(f"  Target LTV:CAC Ratio: {Config.TARGET_LTV_CAC_RATIO}:1")
    print(f"  Target Gross Margin: {Config.DEFAULT_TARGET_GROSS_MARGIN*100}%")
    print(f"  Visitor→Signup Benchmark: {Config.BENCHMARK_VISITOR_TO_SIGNUP*100}%")
    print(f"  Signup→Activated Benchmark: {Config.BENCHMARK_SIGNUP_TO_ACTIVATED*100}%")
    print(f"  Activated→Paid Benchmark: {Config.BENCHMARK_ACTIVATED_TO_PAID*100}%")
