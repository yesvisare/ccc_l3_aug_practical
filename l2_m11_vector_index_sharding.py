"""
Module 11.4: Vector Index Sharding for Multi-Tenant SaaS

Implements production-grade vector database sharding using consistent hashing
to distribute tenants across multiple Pinecone indexes. Overcomes single-index
limitations (100 namespaces, 1M+ vectors) with deterministic tenant routing.

Key Components:
- ShardManager: Routes tenants using MurmurHash3 consistent hashing
- ShardedRAG: Coordinates queries across single or multiple shards
- Shard monitoring and rebalancing support

Trade-offs:
- Operational complexity vs scalability
- Single-tenant queries fast (~350ms), cross-shard queries slower
- Requires Redis for assignment tracking
"""

import logging
import mmh3
from typing import List, Dict, Optional, Any, Tuple
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShardManager:
    """
    Routes tenants to shards using consistent hashing.

    Uses MurmurHash3 for deterministic routing: shard_id = abs(mmh3.hash(tenant_id)) % num_shards
    Maintains explicit assignments in Redis to support rebalancing without disruption.
    """

    def __init__(self, redis_client: Optional[Any], num_shards: int = 4, shard_prefix: str = "tenant-shard"):
        """
        Initialize shard manager.

        Args:
            redis_client: Redis client for assignment tracking (optional for testing)
            num_shards: Number of shards to distribute across
            shard_prefix: Prefix for shard index names
        """
        self.redis_client = redis_client
        self.num_shards = num_shards
        self.shard_prefix = shard_prefix
        self._assignment_key = "tenant_shard_assignments"
        logger.info(f"ShardManager initialized with {num_shards} shards")

    def get_shard_for_tenant(self, tenant_id: str) -> int:
        """
        Deterministically route tenant to shard ID.

        First checks Redis for explicit assignment (supports rebalancing).
        Falls back to consistent hashing if no explicit assignment exists.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Shard ID (0 to num_shards-1)
        """
        # Check Redis for explicit assignment
        if self.redis_client:
            try:
                assignment = self.redis_client.hget(self._assignment_key, tenant_id)
                if assignment is not None:
                    shard_id = int(assignment)
                    logger.debug(f"Tenant {tenant_id} -> shard {shard_id} (cached)")
                    return shard_id
            except Exception as e:
                logger.warning(f"Redis lookup failed, falling back to hash: {e}")

        # Compute shard using consistent hashing
        shard_id = abs(mmh3.hash(tenant_id)) % self.num_shards

        # Cache assignment in Redis
        if self.redis_client:
            try:
                self.redis_client.hset(self._assignment_key, tenant_id, shard_id)
            except Exception as e:
                logger.error(f"Failed to cache assignment: {e}")

        logger.info(f"Tenant {tenant_id} -> shard {shard_id} (hashed)")
        return shard_id

    def get_shard_index_name(self, tenant_id: str) -> str:
        """
        Get Pinecone index name for tenant's shard.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Pinecone index name (e.g., 'tenant-shard-0')
        """
        shard_id = self.get_shard_for_tenant(tenant_id)
        return f"{self.shard_prefix}-{shard_id}"

    def assign_tenant_to_shard(self, tenant_id: str, shard_id: int) -> None:
        """
        Explicitly assign tenant to shard (for rebalancing).

        Args:
            tenant_id: Tenant to reassign
            shard_id: Target shard ID

        Raises:
            ValueError: If shard_id out of range
            RuntimeError: If Redis not available
        """
        if shard_id < 0 or shard_id >= self.num_shards:
            raise ValueError(f"Invalid shard_id {shard_id}, must be 0-{self.num_shards-1}")

        if not self.redis_client:
            raise RuntimeError("Redis required for explicit shard assignments")

        self.redis_client.hset(self._assignment_key, tenant_id, shard_id)
        logger.info(f"Explicitly assigned tenant {tenant_id} to shard {shard_id}")

    def get_shard_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get tenant distribution across shards.

        Returns:
            Dict mapping shard_id -> {"tenant_count": int, "tenants": List[str]}
        """
        stats = {i: {"tenant_count": 0, "tenants": []} for i in range(self.num_shards)}

        if not self.redis_client:
            logger.warning("Redis not available, returning empty stats")
            return stats

        try:
            assignments = self.redis_client.hgetall(self._assignment_key)
            for tenant_id, shard_id in assignments.items():
                shard_id = int(shard_id)
                stats[shard_id]["tenant_count"] += 1
                stats[shard_id]["tenants"].append(tenant_id)
        except Exception as e:
            logger.error(f"Failed to fetch shard stats: {e}")

        return stats


class ShardedRAG:
    """
    Sharded RAG system coordinating queries across multiple Pinecone indexes.

    Patterns:
    - Single-tenant query: Hits one shard only (~350ms P95)
    - Cross-shard query: Aggregates results from all shards (slower, admin use only)
    """

    def __init__(
        self,
        pinecone_client: Optional[Any],
        openai_client: Optional[Any],
        shard_manager: ShardManager,
        vector_dimension: int = 1536
    ):
        """
        Initialize sharded RAG system.

        Args:
            pinecone_client: Pinecone client instance
            openai_client: OpenAI client for embeddings
            shard_manager: ShardManager for routing
            vector_dimension: Embedding dimension
        """
        self.pinecone_client = pinecone_client
        self.openai_client = openai_client
        self.shard_manager = shard_manager
        self.vector_dimension = vector_dimension
        logger.info("ShardedRAG initialized")

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if API unavailable
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available, skipping embedding")
            return None

        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            logger.debug(f"Created embedding for text (dim={len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return None

    def upsert_documents(
        self,
        tenant_id: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest documents for tenant into their assigned shard.

        Each tenant gets a namespace within their shard index.

        Args:
            tenant_id: Tenant identifier
            documents: List of {"id": str, "text": str, "metadata": dict}

        Returns:
            {"success": bool, "shard": str, "count": int, "skipped": bool}
        """
        if not self.pinecone_client or not self.openai_client:
            logger.warning("Services unavailable, skipping upsert")
            return {"success": False, "skipped": True, "reason": "no services"}

        try:
            # Route to shard
            index_name = self.shard_manager.get_shard_index_name(tenant_id)

            # Get or create index
            index = self.pinecone_client.Index(index_name)

            # Prepare vectors
            vectors = []
            for doc in documents:
                embedding = self.create_embedding(doc["text"])
                if embedding:
                    vectors.append({
                        "id": doc["id"],
                        "values": embedding,
                        "metadata": {**doc.get("metadata", {}), "text": doc["text"]}
                    })

            # Upsert to tenant namespace
            if vectors:
                index.upsert(vectors=vectors, namespace=tenant_id)
                logger.info(f"Upserted {len(vectors)} docs for {tenant_id} to {index_name}")

            return {
                "success": True,
                "shard": index_name,
                "count": len(vectors),
                "skipped": False
            }

        except Exception as e:
            logger.error(f"Upsert failed for {tenant_id}: {e}")
            return {"success": False, "error": str(e), "skipped": False}

    def query_single_tenant(
        self,
        tenant_id: str,
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query single tenant's namespace (fast path).

        Hits only one shard, typical P95 latency ~350ms.

        Args:
            tenant_id: Tenant to query
            query_text: Search query
            top_k: Number of results

        Returns:
            {"results": List[dict], "latency_ms": float, "shard": str}
        """
        if not self.pinecone_client or not self.openai_client:
            logger.warning("Services unavailable, skipping query")
            return {"results": [], "skipped": True, "reason": "no services"}

        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.create_embedding(query_text)
            if not query_embedding:
                return {"results": [], "error": "embedding failed"}

            # Route to shard
            index_name = self.shard_manager.get_shard_index_name(tenant_id)
            index = self.pinecone_client.Index(index_name)

            # Query tenant namespace
            response = index.query(
                vector=query_embedding,
                namespace=tenant_id,
                top_k=top_k,
                include_metadata=True
            )

            latency_ms = (time.time() - start_time) * 1000
            results = [
                {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata
                }
                for match in response.matches
            ]

            logger.info(f"Single-tenant query: {len(results)} results, {latency_ms:.0f}ms, shard={index_name}")

            return {
                "results": results,
                "latency_ms": latency_ms,
                "shard": index_name,
                "skipped": False
            }

        except Exception as e:
            logger.error(f"Single-tenant query failed: {e}")
            return {"results": [], "error": str(e)}

    def query_cross_shard(
        self,
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query across all shards (admin use, slower).

        Aggregates results from all indexes. Use sparingly for admin/analytics.
        Higher latency due to multiple index queries + result merging.

        Args:
            query_text: Search query
            top_k: Results per shard (total may be num_shards * top_k)

        Returns:
            {"results": List[dict], "latency_ms": float, "shards_queried": int}
        """
        if not self.pinecone_client or not self.openai_client:
            logger.warning("Services unavailable, skipping cross-shard query")
            return {"results": [], "skipped": True, "reason": "no services"}

        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.create_embedding(query_text)
            if not query_embedding:
                return {"results": [], "error": "embedding failed"}

            all_results = []

            # Query each shard
            for shard_id in range(self.shard_manager.num_shards):
                index_name = f"{self.shard_manager.shard_prefix}-{shard_id}"
                try:
                    index = self.pinecone_client.Index(index_name)
                    response = index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True
                    )

                    for match in response.matches:
                        all_results.append({
                            "id": match.id,
                            "score": match.score,
                            "text": match.metadata.get("text", ""),
                            "metadata": match.metadata,
                            "shard": index_name
                        })

                except Exception as e:
                    logger.warning(f"Failed to query shard {index_name}: {e}")

            # Sort by score
            all_results.sort(key=lambda x: x["score"], reverse=True)

            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Cross-shard query: {len(all_results)} results, {latency_ms:.0f}ms")

            return {
                "results": all_results[:top_k * 2],  # Return top results across all shards
                "latency_ms": latency_ms,
                "shards_queried": self.shard_manager.num_shards,
                "skipped": False
            }

        except Exception as e:
            logger.error(f"Cross-shard query failed: {e}")
            return {"results": [], "error": str(e)}


def monitor_shard_health(
    shard_manager: ShardManager,
    pinecone_client: Optional[Any]
) -> Dict[str, Any]:
    """
    Check shard health against production thresholds.

    Thresholds:
    - MAX_VECTORS_PER_SHARD: 300K vectors
    - MAX_NAMESPACES_PER_SHARD: 20 namespaces
    - Rebalance trigger at 18/20 namespaces

    Args:
        shard_manager: ShardManager instance
        pinecone_client: Pinecone client

    Returns:
        {"shards": List[dict], "needs_rebalancing": bool, "alerts": List[str]}
    """
    if not pinecone_client:
        logger.warning("Pinecone unavailable, cannot monitor shard health")
        return {"shards": [], "needs_rebalancing": False, "alerts": ["Pinecone unavailable"]}

    shard_stats = []
    alerts = []
    needs_rebalancing = False

    for shard_id in range(shard_manager.num_shards):
        index_name = f"{shard_manager.shard_prefix}-{shard_id}"

        try:
            index = pinecone_client.Index(index_name)
            stats = index.describe_index_stats()

            namespace_count = len(stats.namespaces)
            total_vectors = stats.total_vector_count

            shard_info = {
                "shard_id": shard_id,
                "index_name": index_name,
                "namespace_count": namespace_count,
                "total_vectors": total_vectors,
                "status": "healthy"
            }

            # Check thresholds
            if total_vectors > 300000:
                alerts.append(f"{index_name}: {total_vectors} vectors (>300K limit)")
                shard_info["status"] = "warning"
                needs_rebalancing = True

            if namespace_count >= 18:
                alerts.append(f"{index_name}: {namespace_count}/20 namespaces (rebalance threshold)")
                shard_info["status"] = "warning"
                needs_rebalancing = True

            shard_stats.append(shard_info)

        except Exception as e:
            logger.error(f"Failed to check {index_name}: {e}")
            shard_stats.append({
                "shard_id": shard_id,
                "index_name": index_name,
                "status": "error",
                "error": str(e)
            })

    return {
        "shards": shard_stats,
        "needs_rebalancing": needs_rebalancing,
        "alerts": alerts
    }


if __name__ == "__main__":
    """CLI usage examples."""

    from config import Config, get_clients

    print("=== M11.4 Vector Index Sharding ===\n")

    # Initialize clients
    pinecone_client, redis_client, openai_client = get_clients()

    # Example 1: Shard routing
    print("Example 1: Tenant Shard Routing")
    shard_manager = ShardManager(redis_client, num_shards=Config.NUM_SHARDS)

    for tenant_id in ["tenant-001", "tenant-002", "tenant-003"]:
        shard_id = shard_manager.get_shard_for_tenant(tenant_id)
        index_name = shard_manager.get_shard_index_name(tenant_id)
        print(f"  {tenant_id} -> shard {shard_id} ({index_name})")

    # Example 2: Shard distribution stats
    print("\nExample 2: Shard Distribution")
    stats = shard_manager.get_shard_stats()
    for shard_id, info in stats.items():
        print(f"  Shard {shard_id}: {info['tenant_count']} tenants")

    # Example 3: Single-tenant query simulation
    print("\nExample 3: Query Patterns")
    rag = ShardedRAG(pinecone_client, openai_client, shard_manager)

    if pinecone_client and openai_client:
        result = rag.query_single_tenant("tenant-001", "test query", top_k=3)
        print(f"  Single-tenant: {result.get('latency_ms', 'N/A')}ms")
    else:
        print("  ⚠️ Skipping queries (no API keys)")

    print("\n✓ Module ready. Configure .env and run tests.")
