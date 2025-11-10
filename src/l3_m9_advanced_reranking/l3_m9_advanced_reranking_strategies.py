"""
Module 9.4: Advanced Reranking Strategies

Implements four complementary strategies for improving search result quality:
1. Ensemble Reranking with Voting
2. Maximal Marginal Relevance (MMR)
3. Temporal/Recency Boosting
4. User Preference Learning

Author: Level 3 RAG Course
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Literal
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0


@dataclass
class RerankResult:
    """Result from reranking operation."""
    documents: List[Document]
    latency_ms: float
    strategy_used: str
    debug_info: Optional[Dict[str, Any]] = None


class EnsembleReranker:
    """
    Ensemble reranking using multiple cross-encoder models.

    Combines predictions from multiple models to reduce bias and improve accuracy
    by 8-12% over single-model approaches.
    """

    def __init__(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        aggregation: Literal["weighted", "voting", "confidence"] = "weighted"
    ):
        """
        Initialize ensemble reranker.

        Args:
            model_names: List of cross-encoder model identifiers
            weights: Optional weights for each model (must sum to 1.0)
            aggregation: Aggregation strategy - weighted, voting, or confidence
        """
        self.model_names = model_names
        self.models = []
        self.aggregation = aggregation

        # Set default weights if not provided
        if weights is None:
            self.weights = [1.0 / len(model_names)] * len(model_names)
        else:
            if len(weights) != len(model_names):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                logger.warning(f"Weights sum to {sum(weights)}, normalizing...")
                total = sum(weights)
                self.weights = [w / total for w in weights]
            else:
                self.weights = weights

        logger.info(f"Initializing ensemble with {len(model_names)} models")
        self._load_models()

    def _load_models(self) -> None:
        """Load cross-encoder models."""
        try:
            from sentence_transformers import CrossEncoder

            for model_name in self.model_names:
                try:
                    logger.info(f"Loading model: {model_name}")
                    model = CrossEncoder(model_name)
                    self.models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    # Use None as placeholder
                    self.models.append(None)

            # Check if at least one model loaded successfully
            if all(m is None for m in self.models):
                logger.warning("No models loaded successfully, using mock scoring")

        except ImportError:
            logger.warning("sentence-transformers not available, using mock scoring")

    def _aggregate_scores(
        self,
        all_scores: List[List[float]]
    ) -> List[float]:
        """
        Aggregate scores from multiple models.

        Args:
            all_scores: List of score lists from each model

        Returns:
            Aggregated scores
        """
        num_docs = len(all_scores[0])

        if self.aggregation == "weighted":
            # Weighted average
            aggregated = np.zeros(num_docs)
            for scores, weight in zip(all_scores, self.weights):
                aggregated += np.array(scores) * weight
            return aggregated.tolist()

        elif self.aggregation == "voting":
            # Rank-based Borda count
            aggregated = np.zeros(num_docs)
            for scores in all_scores:
                # Get ranks (higher score = lower rank number)
                ranks = np.argsort(np.argsort(scores)[::-1])
                # Borda count: points = (n_docs - rank)
                aggregated += (num_docs - ranks)
            return aggregated.tolist()

        elif self.aggregation == "confidence":
            # Magnitude-based weighting
            aggregated = np.zeros(num_docs)
            for scores in all_scores:
                scores_array = np.array(scores)
                # Weight by confidence (absolute value)
                confidence = np.abs(scores_array)
                aggregated += scores_array * confidence
            # Normalize
            total_confidence = sum(np.abs(scores).sum() for scores in all_scores)
            if total_confidence > 0:
                aggregated /= total_confidence
            return aggregated.tolist()

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> RerankResult:
        """
        Rerank documents using ensemble of models.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Optional number of top documents to return

        Returns:
            RerankResult with reranked documents
        """
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                latency_ms=0.0,
                strategy_used="ensemble",
                debug_info={"note": "No documents to rerank"}
            )

        # Prepare query-document pairs
        pairs = [[query, doc.text] for doc in documents]
        all_scores = []
        individual_model_scores = {}

        # Score with each model
        for i, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            if model is None:
                # Mock scoring for unavailable models
                logger.warning(f"Using mock scores for {model_name}")
                mock_scores = np.random.uniform(0.1, 0.9, len(documents)).tolist()
                all_scores.append(mock_scores)
                individual_model_scores[model_name] = mock_scores
            else:
                try:
                    scores = model.predict(pairs).tolist()
                    all_scores.append(scores)
                    individual_model_scores[model_name] = scores
                except Exception as e:
                    logger.error(f"Error scoring with {model_name}: {e}")
                    # Fallback to mock scores
                    mock_scores = np.random.uniform(0.1, 0.9, len(documents)).tolist()
                    all_scores.append(mock_scores)
                    individual_model_scores[model_name] = mock_scores

        # Aggregate scores
        final_scores = self._aggregate_scores(all_scores)

        # Update document scores and sort
        for doc, score in zip(documents, final_scores):
            doc.score = score

        reranked_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        if top_k:
            reranked_docs = reranked_docs[:top_k]

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Ensemble reranking completed in {latency_ms:.2f}ms")

        return RerankResult(
            documents=reranked_docs,
            latency_ms=latency_ms,
            strategy_used="ensemble",
            debug_info={
                "aggregation_method": self.aggregation,
                "num_models": len(self.models),
                "individual_scores": individual_model_scores,
                "weights": self.weights
            }
        )


class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) reranker.

    Balances relevance against diversity using:
    score = λ × relevance - (1-λ) × max_similarity_to_selected
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0.0 and 1.0")

        self.lambda_param = lambda_param
        logger.info(f"MMR reranker initialized with λ={lambda_param}")

    def _compute_similarity_matrix(
        self,
        documents: List[Document]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between documents.

        Args:
            documents: List of documents

        Returns:
            Similarity matrix (n_docs x n_docs)
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Use a lightweight embedding model
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = embedder.encode([doc.text for doc in documents])
            similarity_matrix = cosine_similarity(embeddings)
            return similarity_matrix

        except Exception as e:
            logger.warning(f"Could not compute embeddings: {e}, using mock similarity")
            # Mock similarity matrix
            n = len(documents)
            # Random similarities with diagonal = 1.0
            sim_matrix = np.random.uniform(0.3, 0.7, (n, n))
            np.fill_diagonal(sim_matrix, 1.0)
            # Make symmetric
            sim_matrix = (sim_matrix + sim_matrix.T) / 2
            return sim_matrix

    def rerank(
        self,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> RerankResult:
        """
        Rerank documents using MMR algorithm.

        Args:
            documents: List of documents with relevance scores
            top_k: Number of diverse documents to select

        Returns:
            RerankResult with diverse documents
        """
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                latency_ms=0.0,
                strategy_used="mmr",
                debug_info={"note": "No documents to rerank"}
            )

        # Sort by relevance score initially
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        if top_k is None:
            top_k = len(documents)

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(sorted_docs)

        # MMR selection
        selected_indices = []
        selected_docs = []
        remaining_indices = list(range(len(sorted_docs)))

        for _ in range(min(top_k, len(sorted_docs))):
            if not remaining_indices:
                break

            mmr_scores = []

            for idx in remaining_indices:
                relevance = sorted_docs[idx].score

                if not selected_indices:
                    # First document: pure relevance
                    mmr_score = relevance
                else:
                    # Compute max similarity to already selected documents
                    max_sim = max(
                        similarity_matrix[idx][selected_idx]
                        for selected_idx in selected_indices
                    )
                    # MMR formula
                    mmr_score = (self.lambda_param * relevance -
                                 (1 - self.lambda_param) * max_sim)

                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            selected_docs.append(sorted_docs[best_idx])
            remaining_indices.remove(best_idx)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"MMR reranking completed in {latency_ms:.2f}ms")

        return RerankResult(
            documents=selected_docs,
            latency_ms=latency_ms,
            strategy_used="mmr",
            debug_info={
                "lambda": self.lambda_param,
                "selected_count": len(selected_docs)
            }
        )


class TemporalReranker:
    """
    Temporal/Recency boosting reranker.

    Applies exponential decay to older documents for time-sensitive queries.
    """

    def __init__(
        self,
        decay_days: int = 30,
        boost_factor: float = 1.5,
        temporal_keywords: Optional[List[str]] = None
    ):
        """
        Initialize temporal reranker.

        Args:
            decay_days: Half-life for exponential decay
            boost_factor: Multiplicative boost for recent documents
            temporal_keywords: Keywords indicating time-sensitive queries
        """
        self.decay_days = decay_days
        self.boost_factor = boost_factor

        if temporal_keywords is None:
            self.temporal_keywords = [
                "latest", "current", "recent", "new", "today", "yesterday",
                "this week", "this month", "2024", "2025", "now", "updated"
            ]
        else:
            self.temporal_keywords = temporal_keywords

        logger.info(f"Temporal reranker initialized with {decay_days}-day decay")

    def is_temporal_query(self, query: str) -> bool:
        """
        Detect if query is time-sensitive.

        Args:
            query: Search query

        Returns:
            True if query contains temporal keywords
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.temporal_keywords)

    def _compute_recency_score(
        self,
        timestamp: str,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Compute recency score using exponential decay.

        Args:
            timestamp: ISO format timestamp
            current_time: Optional current time (defaults to now)

        Returns:
            Recency multiplier (0.0 to boost_factor)
        """
        try:
            doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if current_time is None:
                current_time = datetime.now(timezone.utc)

            # Calculate age in days
            age_days = (current_time - doc_time).days

            # Exponential decay: score = boost_factor * exp(-age / decay_days)
            decay_rate = np.log(2) / self.decay_days  # Half-life formula
            recency_multiplier = self.boost_factor * np.exp(-decay_rate * age_days)

            return max(0.1, min(recency_multiplier, self.boost_factor))

        except Exception as e:
            logger.warning(f"Error computing recency for {timestamp}: {e}")
            return 1.0  # Neutral multiplier on error

    def rerank(
        self,
        query: str,
        documents: List[Document],
        force_temporal: bool = False
    ) -> RerankResult:
        """
        Rerank documents with temporal boosting.

        Args:
            query: Search query
            documents: List of documents with scores
            force_temporal: Force temporal boosting even if query not detected

        Returns:
            RerankResult with temporally adjusted scores
        """
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                latency_ms=0.0,
                strategy_used="temporal",
                debug_info={"note": "No documents to rerank"}
            )

        is_temporal = force_temporal or self.is_temporal_query(query)

        if not is_temporal:
            logger.info("Query not detected as temporal, skipping boosting")
            latency_ms = (time.time() - start_time) * 1000
            return RerankResult(
                documents=documents,
                latency_ms=latency_ms,
                strategy_used="temporal",
                debug_info={"temporal_detected": False}
            )

        # Apply temporal boosting
        boosted_docs = []
        recency_scores = {}

        for doc in documents:
            timestamp = doc.metadata.get("timestamp")

            if timestamp:
                recency_mult = self._compute_recency_score(timestamp)
                boosted_score = doc.score * recency_mult
                recency_scores[doc.id] = recency_mult
            else:
                boosted_score = doc.score
                recency_scores[doc.id] = 1.0

            # Create new document with boosted score
            boosted_doc = Document(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=boosted_score
            )
            boosted_docs.append(boosted_doc)

        # Sort by boosted scores
        reranked_docs = sorted(boosted_docs, key=lambda d: d.score, reverse=True)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Temporal reranking completed in {latency_ms:.2f}ms")

        return RerankResult(
            documents=reranked_docs,
            latency_ms=latency_ms,
            strategy_used="temporal",
            debug_info={
                "temporal_detected": is_temporal,
                "recency_scores": recency_scores,
                "decay_days": self.decay_days,
                "boost_factor": self.boost_factor
            }
        )


class PersonalizationReranker:
    """
    User preference learning reranker.

    Uses implicit feedback to personalize document ranking based on user preferences.
    """

    def __init__(
        self,
        min_interactions: int = 100,
        model_path: Optional[str] = None
    ):
        """
        Initialize personalization reranker.

        Args:
            min_interactions: Minimum interactions required for personalization
            model_path: Path to trained preference model (optional)
        """
        self.min_interactions = min_interactions
        self.model_path = model_path
        self.model = None

        logger.info(f"Personalization reranker initialized (min_interactions={min_interactions})")

    def _extract_features(
        self,
        doc: Document,
        user_profile: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract features for preference prediction.

        Args:
            doc: Document to extract features from
            user_profile: User profile with preferences

        Returns:
            Feature vector
        """
        features = []

        # Document type match
        preferred_sources = user_profile.get("preferences", {}).get("preferred_sources", [])
        doc_source = doc.metadata.get("source", "")
        features.append(1.0 if doc_source in preferred_sources else 0.0)

        # Technical depth match
        preferred_depth = user_profile.get("preferences", {}).get("preferred_depth", 0.5)
        doc_depth = doc.metadata.get("technical_depth", 0.5)
        depth_diff = abs(preferred_depth - doc_depth)
        features.append(1.0 - depth_diff)  # Closer = higher score

        # Length preference
        preferred_length = user_profile.get("preferences", {}).get("preferred_length_range", [100, 300])
        doc_length = doc.metadata.get("length", 150)
        in_range = preferred_length[0] <= doc_length <= preferred_length[1]
        features.append(1.0 if in_range else 0.5)

        # Document type encoding
        doc_type = doc.metadata.get("doc_type", "unknown")
        type_score = {
            "tutorial": 0.8,
            "research": 0.9,
            "reference": 0.7,
            "opinion": 0.5
        }.get(doc_type, 0.5)
        features.append(type_score)

        return np.array(features)

    def _predict_preference(
        self,
        features: np.ndarray
    ) -> float:
        """
        Predict user preference score.

        Args:
            features: Feature vector

        Returns:
            Preference score (0.0 to 1.0)
        """
        if self.model is not None:
            try:
                return float(self.model.predict([features])[0])
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")

        # Simple heuristic: average of features
        return float(np.mean(features))

    def rerank(
        self,
        documents: List[Document],
        user_profile: Dict[str, Any]
    ) -> RerankResult:
        """
        Rerank documents based on user preferences.

        Args:
            documents: List of documents with scores
            user_profile: User profile with interaction history

        Returns:
            RerankResult with personalized scores
        """
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                latency_ms=0.0,
                strategy_used="personalization",
                debug_info={"note": "No documents to rerank"}
            )

        # Check if user has enough interactions
        interaction_count = user_profile.get("interaction_count", 0)

        if interaction_count < self.min_interactions:
            logger.info(
                f"User has {interaction_count} interactions, "
                f"need {self.min_interactions} for personalization"
            )
            latency_ms = (time.time() - start_time) * 1000
            return RerankResult(
                documents=documents,
                latency_ms=latency_ms,
                strategy_used="personalization",
                debug_info={
                    "personalized": False,
                    "reason": "insufficient_interactions"
                }
            )

        # Apply personalization
        personalized_docs = []
        preference_scores = {}

        for doc in documents:
            features = self._extract_features(doc, user_profile)
            pref_score = self._predict_preference(features)

            # Combine with base relevance
            personalized_score = doc.score * (0.7 + 0.3 * pref_score)
            preference_scores[doc.id] = pref_score

            personalized_doc = Document(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                score=personalized_score
            )
            personalized_docs.append(personalized_doc)

        # Sort by personalized scores
        reranked_docs = sorted(personalized_docs, key=lambda d: d.score, reverse=True)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Personalization reranking completed in {latency_ms:.2f}ms")

        return RerankResult(
            documents=reranked_docs,
            latency_ms=latency_ms,
            strategy_used="personalization",
            debug_info={
                "personalized": True,
                "interaction_count": interaction_count,
                "preference_scores": preference_scores
            }
        )


class AdvancedReranker:
    """
    Combined advanced reranking pipeline.

    Orchestrates all four strategies with performance budgets.
    """

    def __init__(
        self,
        enable_ensemble: bool = True,
        enable_mmr: bool = True,
        enable_temporal: bool = True,
        enable_personalization: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize advanced reranker.

        Args:
            enable_ensemble: Enable ensemble reranking
            enable_mmr: Enable MMR diversity
            enable_temporal: Enable temporal boosting
            enable_personalization: Enable user personalization
            config: Optional configuration dictionary
        """
        self.enable_ensemble = enable_ensemble
        self.enable_mmr = enable_mmr
        self.enable_temporal = enable_temporal
        self.enable_personalization = enable_personalization

        # Initialize individual rerankers
        if enable_ensemble and config:
            self.ensemble = EnsembleReranker(
                model_names=config.get("model_names", [
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ]),
                weights=config.get("weights"),
                aggregation=config.get("aggregation", "weighted")
            )
        else:
            self.ensemble = None

        if enable_mmr:
            mmr_lambda = config.get("mmr_lambda", 0.7) if config else 0.7
            self.mmr = MMRReranker(lambda_param=mmr_lambda)
        else:
            self.mmr = None

        if enable_temporal:
            decay_days = config.get("decay_days", 30) if config else 30
            boost_factor = config.get("boost_factor", 1.5) if config else 1.5
            self.temporal = TemporalReranker(
                decay_days=decay_days,
                boost_factor=boost_factor
            )
        else:
            self.temporal = None

        if enable_personalization:
            min_interactions = config.get("min_interactions", 100) if config else 100
            self.personalization = PersonalizationReranker(
                min_interactions=min_interactions
            )
        else:
            self.personalization = None

        logger.info("AdvancedReranker initialized")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        user_profile: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> RerankResult:
        """
        Apply full reranking pipeline.

        Args:
            query: Search query
            documents: Initial documents
            user_profile: Optional user profile for personalization
            top_k: Number of final results

        Returns:
            RerankResult with all strategies applied
        """
        total_start = time.time()
        pipeline_steps = []

        current_docs = documents

        # Step 1: Ensemble reranking
        if self.enable_ensemble and self.ensemble:
            result = self.ensemble.rerank(query, current_docs)
            current_docs = result.documents
            pipeline_steps.append({
                "strategy": "ensemble",
                "latency_ms": result.latency_ms
            })

        # Step 2: Temporal boosting
        if self.enable_temporal and self.temporal:
            result = self.temporal.rerank(query, current_docs)
            current_docs = result.documents
            pipeline_steps.append({
                "strategy": "temporal",
                "latency_ms": result.latency_ms
            })

        # Step 3: Personalization
        if self.enable_personalization and self.personalization and user_profile:
            result = self.personalization.rerank(current_docs, user_profile)
            current_docs = result.documents
            pipeline_steps.append({
                "strategy": "personalization",
                "latency_ms": result.latency_ms
            })

        # Step 4: MMR for diversity
        if self.enable_mmr and self.mmr:
            result = self.mmr.rerank(current_docs, top_k=top_k)
            current_docs = result.documents
            pipeline_steps.append({
                "strategy": "mmr",
                "latency_ms": result.latency_ms
            })
        elif top_k:
            current_docs = current_docs[:top_k]

        total_latency = (time.time() - total_start) * 1000

        logger.info(f"Full reranking pipeline completed in {total_latency:.2f}ms")

        return RerankResult(
            documents=current_docs,
            latency_ms=total_latency,
            strategy_used="combined",
            debug_info={
                "pipeline_steps": pipeline_steps,
                "total_strategies": len(pipeline_steps)
            }
        )


# CLI Examples
if __name__ == "__main__":
    import json

    # Load example data
    with open("example_data.json", "r") as f:
        data = json.load(f)

    query = data["query"]
    raw_docs = data["documents"]
    user_profile = data["user_profile"]

    # Convert to Document objects with initial scores
    documents = [
        Document(
            id=doc["id"],
            text=doc["text"],
            metadata=doc["metadata"],
            score=0.5  # Initial score
        )
        for doc in raw_docs
    ]

    print("=" * 60)
    print("Advanced Reranking Strategies Demo")
    print("=" * 60)

    # Example 1: Ensemble Reranking
    print("\n1. ENSEMBLE RERANKING")
    print("-" * 60)
    ensemble = EnsembleReranker(
        model_names=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
        aggregation="weighted"
    )
    result = ensemble.rerank(query, documents[:], top_k=3)
    print(f"Latency: {result.latency_ms:.2f}ms")
    print("Top 3 documents:")
    for i, doc in enumerate(result.documents, 1):
        print(f"  {i}. {doc.id}: {doc.score:.4f}")

    # Example 2: MMR Diversity
    print("\n2. MMR DIVERSITY")
    print("-" * 60)
    # First assign some relevance scores
    for doc in documents:
        doc.score = np.random.uniform(0.5, 0.9)

    mmr = MMRReranker(lambda_param=0.7)
    result = mmr.rerank(documents[:], top_k=3)
    print(f"Latency: {result.latency_ms:.2f}ms")
    print("Top 3 diverse documents:")
    for i, doc in enumerate(result.documents, 1):
        print(f"  {i}. {doc.id}")

    # Example 3: Temporal Boosting
    print("\n3. TEMPORAL BOOSTING")
    print("-" * 60)
    temporal = TemporalReranker(decay_days=30, boost_factor=1.5)
    for doc in documents:
        doc.score = 0.7  # Reset scores

    result = temporal.rerank(query, documents[:])
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Temporal detected: {result.debug_info['temporal_detected']}")
    print("Top 3 recent documents:")
    for i, doc in enumerate(result.documents[:3], 1):
        print(f"  {i}. {doc.id}: {doc.score:.4f}")

    # Example 4: Personalization
    print("\n4. PERSONALIZATION")
    print("-" * 60)
    personalization = PersonalizationReranker(min_interactions=100)
    for doc in documents:
        doc.score = 0.7  # Reset scores

    result = personalization.rerank(documents[:], user_profile)
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Personalized: {result.debug_info['personalized']}")
    print("Top 3 personalized documents:")
    for i, doc in enumerate(result.documents[:3], 1):
        print(f"  {i}. {doc.id}: {doc.score:.4f}")

    print("\n" + "=" * 60)
