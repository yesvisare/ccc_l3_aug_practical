"""
Configuration management for Module 13.4: Portfolio Showcase & Career Launch

Loads environment variables and provides configuration constants.
This module doesn't require external API keys since it's focused on portfolio creation.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")


# Portfolio Configuration Constants
class PortfolioConfig:
    """Configuration for portfolio generation"""

    # Time investment estimates (hours)
    ARCHITECTURE_DOC_HOURS = 12  # 10-15 hours
    DEMO_VIDEO_HOURS = 18  # 15-20 hours
    CASE_STUDY_HOURS = 9  # 8-10 hours
    INTERVIEW_PREP_HOURS = 13  # 10-15 hours
    GITHUB_OPTIMIZATION_HOURS = 7  # 5-8 hours

    TOTAL_MINIMUM_HOURS = 48
    TOTAL_RECOMMENDED_HOURS = 68

    # Target metrics
    TARGET_CALLBACK_RATE = 10.0  # Percentage
    TARGET_OFFER_RATE = 20.0  # Percentage
    ALERT_THRESHOLD_APPLICATIONS = 20
    ALERT_THRESHOLD_INTERVIEWS = 5

    # Demo video structure (minutes)
    DEMO_HOOK_DURATION = 2
    DEMO_SOLUTION_DURATION = 3
    DEMO_LIVE_DURATION = 7
    DEMO_IMPACT_DURATION = 2
    DEMO_CTA_DURATION = 1
    DEMO_TOTAL_DURATION = 15

    # Case study requirements
    MIN_CASE_STUDY_WORDS = 2000
    RECOMMENDED_CASE_STUDY_WORDS = 3000

    # Application strategy targets
    APPLICATIONS_HIGH_TOUCH = 10  # 1hr each, 30-40% callback
    APPLICATIONS_MEDIUM_TOUCH = 50  # 20min each, 15-20% callback
    APPLICATIONS_LOW_TOUCH = 100  # 5min each, 5-10% callback

    # Cost estimates (INR/month)
    COST_WITH_PORTFOLIO_MIN = 7000
    COST_WITH_PORTFOLIO_MAX = 17000
    COST_WITHOUT_PORTFOLIO_MIN = 2000
    COST_WITHOUT_PORTFOLIO_MAX = 5000


# Output paths
class OutputPaths:
    """File paths for generated artifacts"""

    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / "output"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    ARCHITECTURE_DOC = OUTPUT_DIR / "architecture_documentation.md"
    DEMO_SCRIPT = OUTPUT_DIR / "demo_video_script.json"
    CASE_STUDY = OUTPUT_DIR / "case_study.md"
    INTERVIEW_PREP = OUTPUT_DIR / "interview_preparation.json"
    README_TEMPLATE = OUTPUT_DIR / "README_optimized.md"
    METRICS_TRACKING = OUTPUT_DIR / "application_metrics.json"


# Optional integrations (not required for this module)
def get_optional_config() -> dict:
    """
    Get optional configuration for external services.

    This module focuses on portfolio creation and doesn't require external APIs.
    However, if you want to integrate with services for:
    - Diagram generation (diagrams library uses local rendering)
    - Video hosting (YouTube, Vimeo)
    - Blog publishing (Medium, Dev.to APIs)

    You can add API keys here.
    """
    return {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "medium_api_token": os.getenv("MEDIUM_API_TOKEN"),
        "devto_api_key": os.getenv("DEVTO_API_KEY"),
        "linkedin_access_token": os.getenv("LINKEDIN_ACCESS_TOKEN"),
    }


def validate_config() -> bool:
    """
    Validate configuration is properly loaded.

    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check output directory is writable
        test_file = OutputPaths.OUTPUT_DIR / ".test"
        test_file.touch()
        test_file.unlink()

        logger.info("Configuration validated successfully")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Portfolio Configuration ===\n")

    print(f"Time Investment:")
    print(f"  Minimum: {PortfolioConfig.TOTAL_MINIMUM_HOURS} hours")
    print(f"  Recommended: {PortfolioConfig.TOTAL_RECOMMENDED_HOURS} hours")
    print()

    print(f"Target Metrics:")
    print(f"  Callback rate: {PortfolioConfig.TARGET_CALLBACK_RATE}%")
    print(f"  Offer rate: {PortfolioConfig.TARGET_OFFER_RATE}%")
    print()

    print(f"Output Directory: {OutputPaths.OUTPUT_DIR}")
    print(f"Configuration valid: {validate_config()}")
