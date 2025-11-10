"""
Module 9: Hypothetical Document Embeddings (HyDE)

HyDE addresses vocabulary mismatch by generating hypothetical document-style answers
before embedding, rather than directly embedding user queries. This improves retrieval
quality by 15-40% for conceptual queries from non-expert users.

Key limitations:
- Adds 500-1000ms latency overhead
- Costs $0.001-0.005 per query
- Only ~20% of queries benefit
- Can reduce precision on factoid queries
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import openai
from openai import AsyncOpenAI, OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyDEGenerator:
    """
    Generates hypothetical document answers for queries.

    Key insight: Transform questions into document-style answers
    before embedding for improved vocabulary matching.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 200
    ):
        """
        Initialize HyDE generator.

        Args:
            openai_api_key: OpenAI API key
            model: Model for hypothesis generation (default: gpt-4o-mini for cost)
            temperature: Low temp for consistency (default: 0.3)
            max_tokens: Concise hypotheses work better (default: 200)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_hypothesis(
        self,
        query: str,
        domain_context: Optional[str] = None,
        example_documents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a hypothetical answer to the query.

        Args:
            query: User's question
            domain_context: Optional context about document style/domain
            example_documents: Optional example documents showing style

        Returns:
            Dict with hypothesis, generation time, and metadata
        """
        start_time = time.time()

        # Craft prompt that encourages document-style answers
        system_prompt = """You are an expert writer creating hypothetical document excerpts.

Given a question, write a concise answer in the style of a formal document that would answer this question. Use professional, technical language that matches how official documents are written.

DO NOT:
- Start with "The answer is..."
- Use conversational language
- Include caveats like "typically" or "generally"

DO:
- Write in declarative, factual style
- Use precise terminology
- Write as if excerpted from an authoritative document
"""

        # Add few-shot examples if provided
        if example_documents:
            system_prompt += "\n\nExample document style:\n"
            for i, doc in enumerate(example_documents[:2], 1):
                system_prompt += f"{i}. {doc[:200]}...\n"

        if domain_context:
            system_prompt += f"\n\nDomain context: {domain_context}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            hypothesis = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time

            logger.info(f"Generated hypothesis in {generation_time:.2f}s")
            logger.debug(f"Query: {query[:100]}...")
            logger.debug(f"Hypothesis: {hypothesis[:150]}...")

            return {
                "hypothesis": hypothesis,
                "generation_time_ms": generation_time * 1000,
                "tokens_used": response.usage.total_tokens,
                "model": self.model,
                "success": True
            }

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {str(e)}")
            return {
                "hypothesis": query,  # Fallback to original query
                "generation_time_ms": 0,
                "tokens_used": 0,
                "model": self.model,
                "success": False,
                "error": str(e)
            }


class HyDERetriever:
    """
    Retrieval system using Hypothetical Document Embeddings.

    Instead of embedding the query, embeds a hypothetical answer
    and searches for documents similar to that answer.

    Note: Requires Pinecone for vector search. Gracefully handles missing service.
    """

    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize HyDE retriever.

        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key (optional, skips search if missing)
            pinecone_index_name: Pinecone index name
            embedding_model: OpenAI embedding model
        """
        # Initialize embedding model
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model

        # Initialize hypothesis generator
        self.hyde_gen = HyDEGenerator(openai_api_key=openai_api_key)

        # Initialize Pinecone (optional)
        self.pc = None
        self.index = None
        if pinecone_api_key and pinecone_index_name:
            try:
                from pinecone import Pinecone
                self.pc = Pinecone(api_key=pinecone_api_key)
                self.index = self.pc.Index(pinecone_index_name)
                logger.info(f"Connected to Pinecone index: {pinecone_index_name}")
            except ImportError:
                logger.warning("⚠️ Pinecone not installed. Vector search will be skipped.")
            except Exception as e:
                logger.warning(f"⚠️ Pinecone connection failed: {e}")
        else:
            logger.warning("⚠️ Pinecone credentials missing. Vector search will be skipped.")

    def embed_text(self, text: str) -> List[float]:
        """
        Embed text using OpenAI embeddings.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def retrieve_with_hyde(
        self,
        query: str,
        top_k: int = 10,
        domain_context: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve documents using HyDE.

        Process:
        1. Generate hypothetical answer
        2. Embed the hypothesis
        3. Search Pinecone with hypothesis embedding
        4. Return results with performance metadata

        Args:
            query: User question
            top_k: Number of results to retrieve
            domain_context: Optional domain context for hypothesis generation
            include_metadata: Include metadata in results

        Returns:
            Dict with results, hypothesis, and performance metrics
        """
        start_time = time.time()

        # Step 1: Generate hypothesis
        hypothesis_result = self.hyde_gen.generate_hypothesis(
            query=query,
            domain_context=domain_context
        )

        if not hypothesis_result['success']:
            logger.warning("Hypothesis generation failed, falling back to query")
            text_to_embed = query
        else:
            text_to_embed = hypothesis_result['hypothesis']

        # Step 2: Embed hypothesis
        embed_start = time.time()
        hypothesis_embedding = self.embed_text(text_to_embed)
        embed_time = (time.time() - embed_start) * 1000

        # Step 3: Search Pinecone (if available)
        search_results = []
        search_time = 0

        if self.index:
            try:
                search_start = time.time()
                search_response = self.index.query(
                    vector=hypothesis_embedding,
                    top_k=top_k,
                    include_metadata=include_metadata
                )
                search_results = search_response.matches
                search_time = (time.time() - search_start) * 1000
            except Exception as e:
                logger.error(f"⚠️ Pinecone search failed: {e}")
        else:
            logger.warning("⚠️ Skipping Pinecone search (no service)")

        total_time = (time.time() - start_time) * 1000

        return {
            "results": search_results,
            "hypothesis": hypothesis_result.get('hypothesis'),
            "performance": {
                "total_time_ms": total_time,
                "hypothesis_generation_ms": hypothesis_result['generation_time_ms'],
                "embedding_time_ms": embed_time,
                "search_time_ms": search_time
            },
            "metadata": {
                "used_hyde": hypothesis_result['success'],
                "tokens_used": hypothesis_result.get('tokens_used', 0),
                "num_results": len(search_results),
                "skipped_search": self.index is None
            }
        }


class HybridHyDERetriever:
    """
    Combines HyDE and traditional retrieval for best of both worlds.

    Strategy: Run both approaches, merge and rerank results using
    reciprocal rank fusion with configurable weights.
    """

    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
        hyde_weight: float = 0.6,
        traditional_weight: float = 0.4
    ):
        """
        Initialize hybrid retriever.

        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key (optional)
            pinecone_index_name: Pinecone index name
            hyde_weight: Weight for HyDE results (default: 0.6)
            traditional_weight: Weight for traditional results (default: 0.4)
        """
        self.hyde_retriever = HyDERetriever(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name
        )

        self.hyde_weight = hyde_weight
        self.traditional_weight = traditional_weight

    def retrieve_traditional(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Traditional retrieval: embed query directly.

        Args:
            query: User question
            top_k: Number of results

        Returns:
            Dict with results and performance metrics
        """
        start_time = time.time()

        # Embed the query (not a hypothesis)
        query_embedding = self.hyde_retriever.embed_text(query)

        # Search (if Pinecone available)
        results = []
        if self.hyde_retriever.index:
            try:
                search_response = self.hyde_retriever.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                results = search_response.matches
            except Exception as e:
                logger.error(f"⚠️ Traditional search failed: {e}")
        else:
            logger.warning("⚠️ Skipping traditional search (no Pinecone)")

        total_time = (time.time() - start_time) * 1000

        return {
            "results": results,
            "performance": {
                "total_time_ms": total_time
            },
            "metadata": {
                "skipped_search": self.hyde_retriever.index is None
            }
        }

    def merge_results(
        self,
        hyde_results: List,
        traditional_results: List,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Merge and rerank results using reciprocal rank fusion.

        Args:
            hyde_results: Results from HyDE
            traditional_results: Results from traditional retrieval
            top_k: Number of results to return

        Returns:
            Merged and ranked results
        """
        # Create score dictionary for all unique documents
        doc_scores = {}

        # Add HyDE results with weight
        for i, match in enumerate(hyde_results):
            doc_id = match.id if hasattr(match, 'id') else str(i)
            # Reciprocal rank scoring
            score = self.hyde_weight * (1.0 / (i + 1))
            doc_scores[doc_id] = {
                "score": score,
                "metadata": match.metadata if hasattr(match, 'metadata') else {},
                "source": "hyde"
            }

        # Add traditional results with weight
        for i, match in enumerate(traditional_results):
            doc_id = match.id if hasattr(match, 'id') else str(i)
            score = self.traditional_weight * (1.0 / (i + 1))

            if doc_id in doc_scores:
                # Document in both results - add scores
                doc_scores[doc_id]["score"] += score
                doc_scores[doc_id]["source"] = "both"
            else:
                doc_scores[doc_id] = {
                    "score": score,
                    "metadata": match.metadata if hasattr(match, 'metadata') else {},
                    "source": "traditional"
                }

        # Sort by combined score
        ranked_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        return [
            {
                "id": doc_id,
                "score": info["score"],
                "metadata": info["metadata"],
                "source": info["source"]
            }
            for doc_id, info in ranked_docs[:top_k]
        ]

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve using hybrid HyDE + traditional approach.

        Args:
            query: User question
            top_k: Number of results
            domain_context: Optional domain context

        Returns:
            Merged results with performance metrics
        """
        start_time = time.time()

        # Get HyDE results
        hyde_response = self.hyde_retriever.retrieve_with_hyde(
            query=query,
            top_k=top_k * 2,  # Get more for merging
            domain_context=domain_context
        )

        # Get traditional results
        traditional_response = self.retrieve_traditional(
            query=query,
            top_k=top_k * 2
        )

        # Merge results
        merged_results = self.merge_results(
            hyde_results=hyde_response["results"],
            traditional_results=traditional_response["results"],
            top_k=top_k
        )

        total_time = (time.time() - start_time) * 1000

        return {
            "results": merged_results,
            "hypothesis": hyde_response.get("hypothesis"),
            "performance": {
                "total_time_ms": total_time,
                "hyde_time_ms": hyde_response["performance"]["total_time_ms"],
                "traditional_time_ms": traditional_response["performance"]["total_time_ms"]
            },
            "metadata": {
                "hyde_count": sum(1 for r in merged_results if r["source"] in ["hyde", "both"]),
                "traditional_count": sum(1 for r in merged_results if r["source"] in ["traditional", "both"]),
                "both_count": sum(1 for r in merged_results if r["source"] == "both")
            }
        }


class QueryClassifier:
    """
    Classify queries to determine if HyDE is appropriate.

    HyDE helps with:
    - Vocabulary mismatch queries
    - Conceptual/explanatory questions
    - Domain-specific jargon queries

    HyDE hurts with:
    - Factoid queries (dates, numbers)
    - Well-phrased formal queries
    - Queries already using document terminology
    """

    def __init__(self):
        """Initialize query classifier with pattern matching rules."""
        # Patterns indicating HyDE is beneficial
        self.hyde_beneficial_patterns = [
            r"\bwhat (is|are)\b",
            r"\bhow (do|does|to)\b",
            r"\bwhy\b",
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bimplications?\b",
            r"\brelationship between\b"
        ]

        # Patterns indicating HyDE may hurt
        self.hyde_harmful_patterns = [
            r"\bwhen (was|did|will)\b",
            r"\bwho (is|was)\b",
            r"\bhow many\b",
            r"\bwhat (date|time|year|month)\b",
            r"\blist\b",
            r"\benumerate\b"
        ]

    def should_use_hyde(
        self,
        query: str,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Determine if HyDE should be used for this query.

        Args:
            query: User question
            threshold: Decision threshold

        Returns:
            Dict with decision and reasoning
        """
        query_lower = query.lower()

        # Count pattern matches
        beneficial_matches = sum(
            1 for pattern in self.hyde_beneficial_patterns
            if re.search(pattern, query_lower)
        )

        harmful_matches = sum(
            1 for pattern in self.hyde_harmful_patterns
            if re.search(pattern, query_lower)
        )

        # Calculate score
        total_patterns = len(self.hyde_beneficial_patterns) + len(self.hyde_harmful_patterns)
        hyde_score = (beneficial_matches - harmful_matches) / total_patterns

        # Decision
        use_hyde = hyde_score > threshold

        return {
            "use_hyde": use_hyde,
            "confidence": abs(hyde_score),
            "beneficial_signals": beneficial_matches,
            "harmful_signals": harmful_matches,
            "reasoning": self._explain_decision(
                use_hyde, beneficial_matches, harmful_matches
            )
        }

    def _explain_decision(
        self,
        use_hyde: bool,
        beneficial: int,
        harmful: int
    ) -> str:
        """Generate human-readable explanation."""
        if use_hyde:
            return f"HyDE recommended: Found {beneficial} conceptual question patterns, {harmful} factoid patterns"
        else:
            return f"Traditional retrieval recommended: Found {harmful} factoid patterns, {beneficial} conceptual patterns"


class AdaptiveHyDERetriever:
    """
    Intelligent retriever that automatically chooses HyDE vs traditional.

    Routes queries based on query type classification to optimize
    quality, latency, and cost.
    """

    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_index_name: Optional[str] = None
    ):
        """
        Initialize adaptive retriever.

        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key (optional)
            pinecone_index_name: Pinecone index name
        """
        self.hybrid_retriever = HybridHyDERetriever(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name
        )
        self.classifier = QueryClassifier()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        force_method: Optional[Literal["hyde", "traditional", "hybrid"]] = None
    ) -> Dict[str, Any]:
        """
        Adaptively retrieve using the best method for the query.

        Args:
            query: User question
            top_k: Number of results
            force_method: Override automatic routing (for testing)

        Returns:
            Retrieval results with routing decision metadata
        """
        # Classify query
        classification = self.classifier.should_use_hyde(query)

        # Determine method
        if force_method:
            method = force_method
            routing_reason = f"Forced to {force_method}"
        elif classification["use_hyde"]:
            method = "hybrid"  # Use hybrid for best of both
            routing_reason = classification["reasoning"]
        else:
            method = "traditional"
            routing_reason = classification["reasoning"]

        # Route to appropriate retriever
        if method == "hyde":
            results = self.hybrid_retriever.hyde_retriever.retrieve_with_hyde(
                query=query,
                top_k=top_k
            )
        elif method == "traditional":
            results = self.hybrid_retriever.retrieve_traditional(
                query=query,
                top_k=top_k
            )
        else:  # hybrid
            results = self.hybrid_retriever.retrieve_hybrid(
                query=query,
                top_k=top_k
            )

        # Add routing metadata
        results["routing"] = {
            "method_used": method,
            "classification": classification,
            "reasoning": routing_reason
        }

        return results


class HyDEEvaluator:
    """
    Compare HyDE vs traditional retrieval performance.

    Tracks retrieval quality, latency, and cost for decision-making.
    """

    def __init__(self, retriever: HybridHyDERetriever):
        """
        Initialize evaluator.

        Args:
            retriever: HybridHyDERetriever instance
        """
        self.retriever = retriever
        self.results_log = []

    def evaluate_query(
        self,
        query: str,
        relevant_doc_ids: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate a single query with both methods.

        Args:
            query: User question
            relevant_doc_ids: List of known relevant document IDs
            top_k: Number of results to retrieve

        Returns:
            Comparison metrics for both approaches
        """
        # HyDE retrieval
        hyde_results = self.retriever.hyde_retriever.retrieve_with_hyde(
            query=query,
            top_k=top_k
        )

        # Traditional retrieval
        traditional_results = self.retriever.retrieve_traditional(
            query=query,
            top_k=top_k
        )

        # Calculate metrics
        def calculate_metrics(results, method_name):
            retrieved_ids = [r.id if hasattr(r, 'id') else r['id']
                           for r in results["results"]]

            # Precision@K
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_doc_ids))
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0

            # Recall@K
            recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0

            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_doc_ids:
                    mrr = 1.0 / (i + 1)
                    break

            return {
                "method": method_name,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "mrr": mrr,
                "latency_ms": results["performance"]["total_time_ms"],
                "retrieved_count": len(retrieved_ids)
            }

        hyde_metrics = calculate_metrics(hyde_results, "hyde")
        traditional_metrics = calculate_metrics(traditional_results, "traditional")

        # Compare
        comparison = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "hyde": hyde_metrics,
            "traditional": traditional_metrics,
            "winner": {
                "precision": "hyde" if hyde_metrics["precision_at_k"] > traditional_metrics["precision_at_k"] else "traditional",
                "recall": "hyde" if hyde_metrics["recall_at_k"] > traditional_metrics["recall_at_k"] else "traditional",
                "latency": "traditional"  # Traditional always faster (no LLM call)
            },
            "hyde_hypothesis": hyde_results.get("hypothesis", "")
        }

        self.results_log.append(comparison)
        return comparison

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Aggregate statistics across all evaluated queries."""
        if not self.results_log:
            return {}

        hyde_wins = {
            "precision": 0,
            "recall": 0
        }

        hyde_total_latency = 0
        traditional_total_latency = 0

        for result in self.results_log:
            if result["winner"]["precision"] == "hyde":
                hyde_wins["precision"] += 1
            if result["winner"]["recall"] == "hyde":
                hyde_wins["recall"] += 1

            hyde_total_latency += result["hyde"]["latency_ms"]
            traditional_total_latency += result["traditional"]["latency_ms"]

        n = len(self.results_log)

        return {
            "total_queries": n,
            "hyde_precision_win_rate": hyde_wins["precision"] / n,
            "hyde_recall_win_rate": hyde_wins["recall"] / n,
            "avg_hyde_latency_ms": hyde_total_latency / n,
            "avg_traditional_latency_ms": traditional_total_latency / n,
            "latency_overhead_ms": (hyde_total_latency - traditional_total_latency) / n
        }

    def save_results(self, filepath: str):
        """Save evaluation results to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                "aggregate_stats": self.get_aggregate_stats(),
                "detailed_results": self.results_log
            }, f, indent=2)


