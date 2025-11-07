"""
Module 9.2: Multi-Hop & Recursive Retrieval - Configuration
Manages environment variables and client initialization
"""

import os
import logging
from typing import Optional, Tuple, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multi-hop-retrieval")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Multi-Hop Configuration
MAX_HOP_DEPTH = int(os.getenv("MAX_HOP_DEPTH", "3"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
MAX_TOKENS_PER_QUERY = int(os.getenv("MAX_TOKENS_PER_QUERY", "8000"))
BEAM_WIDTH = int(os.getenv("BEAM_WIDTH", "5"))

# Vector Search Configuration
TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "10"))
TOP_K_PER_HOP = int(os.getenv("TOP_K_PER_HOP", "5"))

# Application Settings
PORT = int(os.getenv("PORT", "8000"))


def get_clients() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Initialize and return clients for Pinecone, Neo4j, and LLM.

    Returns:
        Tuple of (pinecone_index, neo4j_driver, llm_client)
        Any unavailable client returns None
    """
    pinecone_index = None
    neo4j_driver = None
    llm_client = None

    # Initialize Pinecone
    if PINECONE_API_KEY:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"✓ Pinecone client initialized for index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
    else:
        logger.warning("⚠️ Pinecone API key not found")

    # Initialize Neo4j
    if NEO4J_PASSWORD:
        try:
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Test connection
            neo4j_driver.verify_connectivity()
            logger.info(f"✓ Neo4j driver initialized at: {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            neo4j_driver = None
    else:
        logger.warning("⚠️ Neo4j password not found")

    # Initialize LLM client (prefer OpenAI, fallback to Anthropic)
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✓ OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    elif ANTHROPIC_API_KEY:
        try:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("✓ Anthropic client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
    else:
        logger.warning("⚠️ No LLM API keys found (OpenAI or Anthropic)")

    return pinecone_index, neo4j_driver, llm_client


def has_required_services() -> bool:
    """Check if all required services are configured."""
    return all([
        PINECONE_API_KEY,
        NEO4J_PASSWORD,
        (OPENAI_API_KEY or ANTHROPIC_API_KEY)
    ])
