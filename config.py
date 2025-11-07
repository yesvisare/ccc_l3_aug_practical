"""
Module 9.4: Advanced Reranking Strategies - Configuration

Reads environment variables and provides configuration for reranking models.
"""

import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


class RerankerConfig:
    """Configuration for advanced reranking strategies."""

    # Cross-encoder model names
    RERANKER_MODELS: List[str] = [
        os.getenv("RERANKER_MODEL_1", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        os.getenv("RERANKER_MODEL_2", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        os.getenv("RERANKER_MODEL_3", "cross-encoder/ms-marco-TinyBERT-L-6"),
    ]

    # Ensemble weights (must sum to 1.0)
    ENSEMBLE_WEIGHTS: List[float] = [
        float(os.getenv("ENSEMBLE_WEIGHT_1", "0.4")),
        float(os.getenv("ENSEMBLE_WEIGHT_2", "0.4")),
        float(os.getenv("ENSEMBLE_WEIGHT_3", "0.2")),
    ]

    # MMR parameters
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.7"))

    # Temporal boosting parameters
    RECENCY_DECAY_DAYS: int = int(os.getenv("RECENCY_DECAY_DAYS", "30"))
    RECENCY_BOOST_FACTOR: float = float(os.getenv("RECENCY_BOOST_FACTOR", "1.5"))

    # Performance budgets (milliseconds)
    MAX_ENSEMBLE_LATENCY_MS: int = int(os.getenv("MAX_ENSEMBLE_LATENCY_MS", "400"))
    MAX_MMR_LATENCY_MS: int = int(os.getenv("MAX_MMR_LATENCY_MS", "20"))
    MAX_TEMPORAL_LATENCY_MS: int = int(os.getenv("MAX_TEMPORAL_LATENCY_MS", "5"))
    MAX_PERSONALIZATION_LATENCY_MS: int = int(os.getenv("MAX_PERSONALIZATION_LATENCY_MS", "30"))

    # Personalization settings
    MIN_USER_INTERACTIONS: int = int(os.getenv("MIN_USER_INTERACTIONS", "100"))
    PERSONALIZATION_MODEL_PATH: str = os.getenv(
        "PERSONALIZATION_MODEL_PATH",
        "./models/user_preference_model.pkl"
    )

    # Temporal query detection keywords
    TEMPORAL_KEYWORDS: List[str] = [
        "latest", "current", "recent", "new", "today", "yesterday",
        "this week", "this month", "2024", "2025", "now", "updated"
    ]

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration values."""
        try:
            # Check ensemble weights sum to 1.0
            weight_sum = sum(cls.ENSEMBLE_WEIGHTS)
            if not (0.99 <= weight_sum <= 1.01):
                logger.warning(
                    f"Ensemble weights sum to {weight_sum}, should be 1.0. "
                    f"Normalizing..."
                )
                # Normalize weights
                cls.ENSEMBLE_WEIGHTS = [w / weight_sum for w in cls.ENSEMBLE_WEIGHTS]

            # Check MMR lambda is in valid range
            if not (0.0 <= cls.MMR_LAMBDA <= 1.0):
                logger.error(f"MMR_LAMBDA must be between 0 and 1, got {cls.MMR_LAMBDA}")
                return False

            logger.info("Configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def get_config() -> RerankerConfig:
    """
    Get validated configuration instance.

    Returns:
        RerankerConfig: Configuration object
    """
    config = RerankerConfig()
    config.validate()
    return config


def has_models_available() -> bool:
    """
    Check if reranker models can be loaded.

    Returns:
        bool: True if models are available, False otherwise
    """
    try:
        from sentence_transformers import CrossEncoder
        # Try to load the first model
        model_name = RerankerConfig.RERANKER_MODELS[0]
        _ = CrossEncoder(model_name)
        return True
    except Exception as e:
        logger.warning(f"Models not available: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"Reranker models: {config.RERANKER_MODELS}")
    print(f"Ensemble weights: {config.ENSEMBLE_WEIGHTS}")
    print(f"MMR lambda: {config.MMR_LAMBDA}")
    print(f"Recency decay days: {config.RECENCY_DECAY_DAYS}")
    print(f"Models available: {has_models_available()}")
