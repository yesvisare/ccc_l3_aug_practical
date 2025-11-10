"""
Module 9.1: Query Decomposition & Planning

Advanced retrieval techniques for handling complex multi-part queries.
"""

from .pipeline import (
    QueryDecomposer,
    DependencyGraph,
    ParallelExecutionEngine,
    AnswerSynthesizer,
    QueryDecompositionPipeline,
    SubQuery,
    DecompositionResult,
    DecompositionError,
    DependencyError,
    SynthesisError,
)
from .config import Config, get_openai_client, get_clients

__version__ = "1.0.0"

__all__ = [
    "QueryDecomposer",
    "DependencyGraph",
    "ParallelExecutionEngine",
    "AnswerSynthesizer",
    "QueryDecompositionPipeline",
    "SubQuery",
    "DecompositionResult",
    "DecompositionError",
    "DependencyError",
    "SynthesisError",
    "Config",
    "get_openai_client",
    "get_clients",
]
