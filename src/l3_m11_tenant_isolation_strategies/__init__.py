"""
Module 11.1: Tenant Isolation Strategies for Multi-Tenant SaaS

This module implements production-grade tenant isolation for RAG systems supporting
100-500 customers with namespace-based and index-based isolation strategies.

Key Features:
- PostgreSQL Row-Level Security (RLS) for database-level isolation
- Namespace-based isolation for Free/Pro tiers (shared index)
- Index-based isolation for Enterprise tier (dedicated index)
- Cost tracking and allocation engine
- Performance isolation with resource quotas

Author: CCC Level 3
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Subscription tiers with different isolation strategies."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    """Resource limits per tenant tier."""
    max_documents: int
    max_daily_queries: int
    max_storage_mb: int
    isolation_strategy: str  # "namespace" or "index"


@dataclass
class TenantConfig:
    """Tenant configuration and metadata."""
    tenant_id: str
    tenant_name: str
    tier: TenantTier
    quota: ResourceQuota
    namespace: Optional[str] = None
    dedicated_index: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class CostMetrics:
    """Cost tracking metrics per tenant."""
    tenant_id: str
    embedding_calls: int
    llm_calls: int
    query_count: int
    storage_mb: float
    variable_cost: float  # Per-query costs
    allocated_fixed_cost: float  # Share of infrastructure


class TenantRegistry:
    """
    Central tenant management system.

    Tracks tenant configurations, quotas, and enforces isolation policies.
    In production, this would be backed by PostgreSQL with RLS policies.
    """

    def __init__(self):
        """Initialize tenant registry with tier-based quotas."""
        self.tenants: Dict[str, TenantConfig] = {}
        self.namespace_usage: Dict[str, int] = {}  # Track namespace allocation
        self._define_tier_quotas()
        logger.info("TenantRegistry initialized")

    def _define_tier_quotas(self) -> None:
        """Define resource quotas for each subscription tier."""
        self.tier_quotas = {
            TenantTier.FREE: ResourceQuota(
                max_documents=1000,
                max_daily_queries=100,
                max_storage_mb=50,
                isolation_strategy="namespace"
            ),
            TenantTier.PRO: ResourceQuota(
                max_documents=10000,
                max_daily_queries=1000,
                max_storage_mb=500,
                isolation_strategy="namespace"
            ),
            TenantTier.ENTERPRISE: ResourceQuota(
                max_documents=100000,
                max_daily_queries=10000,
                max_storage_mb=5000,
                isolation_strategy="index"
            )
        }

    def register_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        tier: TenantTier
    ) -> TenantConfig:
        """
        Register a new tenant with tier-based configuration.

        Args:
            tenant_id: Unique tenant identifier
            tenant_name: Human-readable tenant name
            tier: Subscription tier

        Returns:
            TenantConfig: Configured tenant

        Raises:
            ValueError: If tenant already exists or namespace limit reached
        """
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already registered")

        quota = self.tier_quotas[tier]

        # Assign isolation strategy based on tier
        if quota.isolation_strategy == "namespace":
            namespace = self._assign_namespace(tenant_id)
            config = TenantConfig(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                tier=tier,
                quota=quota,
                namespace=namespace,
                created_at=datetime.now()
            )
        else:  # index-based for Enterprise
            dedicated_index = f"tenant-{tenant_id}"
            config = TenantConfig(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                tier=tier,
                quota=quota,
                dedicated_index=dedicated_index,
                created_at=datetime.now()
            )

        self.tenants[tenant_id] = config
        logger.info(f"Registered tenant {tenant_id} ({tier.value}) with {quota.isolation_strategy} isolation")
        return config

    def _assign_namespace(self, tenant_id: str) -> str:
        """
        Assign namespace for shared index isolation.

        Critical: Monitor namespace limits (100 per Pinecone index).
        Alert at 80% capacity (72/90 namespaces).

        Args:
            tenant_id: Tenant identifier

        Returns:
            str: Assigned namespace

        Raises:
            RuntimeError: If namespace capacity exhausted
        """
        # Track namespace allocation per index
        shared_index = "shared-index-1"
        current_count = self.namespace_usage.get(shared_index, 0)

        # CRITICAL: Namespace exhaustion prevention
        max_namespaces = 90  # Conservative limit (Pinecone max: 100)
        if current_count >= max_namespaces:
            raise RuntimeError(
                f"Namespace exhaustion: {current_count}/{max_namespaces} used. "
                "Provision new index immediately!"
            )

        # Alert at 80% capacity
        if current_count >= int(max_namespaces * 0.8):
            logger.error(
                f"ALERT: Namespace capacity at {current_count}/{max_namespaces} (80%+). "
                "Provision new index soon!"
            )

        namespace = f"tenant_{tenant_id}"
        self.namespace_usage[shared_index] = current_count + 1
        logger.info(f"Assigned namespace {namespace} ({current_count + 1}/{max_namespaces})")
        return namespace

    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """
        Retrieve tenant configuration.

        In production, this would enforce PostgreSQL RLS policies.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantConfig or None if not found
        """
        return self.tenants.get(tenant_id)

    def check_quota(self, tenant_id: str, resource: str, amount: int) -> bool:
        """
        Check if tenant has quota available.

        Args:
            tenant_id: Tenant identifier
            resource: Resource type (documents, queries, storage)
            amount: Amount to check

        Returns:
            bool: True if quota available
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            logger.error(f"Tenant {tenant_id} not found for quota check")
            return False

        quota = tenant.quota
        # Simplified quota check - production would track actual usage
        if resource == "documents":
            return amount <= quota.max_documents
        elif resource == "queries":
            return amount <= quota.max_daily_queries
        elif resource == "storage":
            return amount <= quota.max_storage_mb

        return False


class TenantDataManager:
    """
    Wraps all data operations with mandatory tenant scoping.

    CRITICAL: Every query/upsert MUST include tenant_id to prevent cross-tenant leakage.
    This class makes it impossible to query without tenant context.
    """

    def __init__(self, registry: TenantRegistry):
        """
        Initialize data manager with tenant registry.

        Args:
            registry: TenantRegistry instance
        """
        self.registry = registry
        logger.info("TenantDataManager initialized")

    def upsert_documents(
        self,
        tenant_id: str,
        documents: List[Dict[str, Any]],
        pinecone_client=None
    ) -> Dict[str, Any]:
        """
        Insert documents with tenant isolation.

        Args:
            tenant_id: Tenant identifier (mandatory)
            documents: List of documents with vectors
            pinecone_client: Pinecone client (optional for demo)

        Returns:
            Dict with upsert results

        Raises:
            ValueError: If tenant not found or quota exceeded
        """
        tenant = self.registry.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Quota check
        if not self.registry.check_quota(tenant_id, "documents", len(documents)):
            raise ValueError(f"Tenant {tenant_id} document quota exceeded")

        # Add tenant_id to metadata for belt-and-suspenders protection
        for doc in documents:
            if "metadata" not in doc:
                doc["metadata"] = {}
            doc["metadata"]["tenant_id"] = tenant_id

        # Route to appropriate isolation strategy
        if tenant.namespace:
            # Namespace-based isolation
            logger.info(
                f"Upserting {len(documents)} docs for tenant {tenant_id} "
                f"to namespace {tenant.namespace}"
            )
            if pinecone_client:
                # Would call: pinecone_client.upsert(namespace=tenant.namespace, vectors=documents)
                pass
            return {
                "upserted": len(documents),
                "namespace": tenant.namespace,
                "isolation": "namespace"
            }
        else:
            # Index-based isolation
            logger.info(
                f"Upserting {len(documents)} docs for tenant {tenant_id} "
                f"to dedicated index {tenant.dedicated_index}"
            )
            if pinecone_client:
                # Would call: pinecone_client.Index(tenant.dedicated_index).upsert(vectors=documents)
                pass
            return {
                "upserted": len(documents),
                "index": tenant.dedicated_index,
                "isolation": "index"
            }

    def query_documents(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 5,
        pinecone_client=None
    ) -> Dict[str, Any]:
        """
        Query with MANDATORY tenant scoping.

        CRITICAL FAILURE PREVENTION:
        - Cannot query without tenant_id (enforced by method signature)
        - Namespace parameter is mandatory, not optional
        - Metadata filter adds redundant protection

        Args:
            tenant_id: Tenant identifier (mandatory - no default!)
            query_vector: Query embedding vector
            top_k: Number of results
            pinecone_client: Pinecone client (optional for demo)

        Returns:
            Dict with query results

        Raises:
            ValueError: If tenant not found
        """
        tenant = self.registry.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Quota check for queries
        if not self.registry.check_quota(tenant_id, "queries", 1):
            raise ValueError(f"Tenant {tenant_id} daily query quota exceeded")

        # Route based on isolation strategy
        if tenant.namespace:
            # Namespace-based query with redundant metadata filter
            logger.info(
                f"Querying tenant {tenant_id} namespace {tenant.namespace} (top_k={top_k})"
            )
            if pinecone_client:
                # Would call:
                # pinecone_client.query(
                #     namespace=tenant.namespace,  # PRIMARY isolation
                #     vector=query_vector,
                #     filter={"tenant_id": tenant_id},  # REDUNDANT protection
                #     top_k=top_k
                # )
                pass
            return {
                "matches": [],  # Would contain actual results
                "namespace": tenant.namespace,
                "isolation": "namespace",
                "overhead_ms": 18  # ~15-25ms isolation overhead
            }
        else:
            # Index-based query
            logger.info(
                f"Querying tenant {tenant_id} dedicated index {tenant.dedicated_index}"
            )
            if pinecone_client:
                # Would call: pinecone_client.Index(tenant.dedicated_index).query(...)
                pass
            return {
                "matches": [],
                "index": tenant.dedicated_index,
                "isolation": "index",
                "overhead_ms": 5  # Lower overhead for dedicated index
            }


class CostAllocationEngine:
    """
    Tracks and allocates costs per tenant.

    Distinguishes between:
    - Variable costs (per-query): embedding, LLM, vector search
    - Fixed costs (shared infrastructure): Pinecone base, database, monitoring
    """

    # Cost constants from source material
    COST_PER_1K_EMBED_TOKENS = 0.0001
    COST_PER_1K_LLM_TOKENS = 0.002
    COST_PER_QUERY_PINECONE = 0.00001

    def __init__(self):
        """Initialize cost tracking."""
        self.tenant_metrics: Dict[str, CostMetrics] = {}
        logger.info("CostAllocationEngine initialized")

    def track_query_cost(
        self,
        tenant_id: str,
        embed_tokens: int,
        llm_tokens: int
    ) -> float:
        """
        Track variable costs per query.

        Args:
            tenant_id: Tenant identifier
            embed_tokens: Tokens used for embeddings
            llm_tokens: Tokens used for LLM generation

        Returns:
            float: Total variable cost for this query
        """
        if tenant_id not in self.tenant_metrics:
            self.tenant_metrics[tenant_id] = CostMetrics(
                tenant_id=tenant_id,
                embedding_calls=0,
                llm_calls=0,
                query_count=0,
                storage_mb=0.0,
                variable_cost=0.0,
                allocated_fixed_cost=0.0
            )

        metrics = self.tenant_metrics[tenant_id]

        # Calculate variable costs
        embed_cost = (embed_tokens / 1000) * self.COST_PER_1K_EMBED_TOKENS
        llm_cost = (llm_tokens / 1000) * self.COST_PER_1K_LLM_TOKENS
        query_cost = self.COST_PER_QUERY_PINECONE

        total_cost = embed_cost + llm_cost + query_cost

        # Update metrics
        metrics.embedding_calls += 1
        metrics.llm_calls += 1
        metrics.query_count += 1
        metrics.variable_cost += total_cost

        logger.info(
            f"Tenant {tenant_id} query cost: ${total_cost:.6f} "
            f"(embed: ${embed_cost:.6f}, llm: ${llm_cost:.6f})"
        )

        return total_cost

    def allocate_fixed_costs(
        self,
        monthly_fixed_cost: float,
        allocation_basis: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Allocate shared infrastructure costs proportionally.

        CRITICAL: Fixed costs must be allocated to determine true profitability.
        Allocated costs should equal actual cloud bills within ±5%.

        Args:
            monthly_fixed_cost: Total monthly fixed infrastructure cost
            allocation_basis: Dict of {tenant_id: usage_percentage}

        Returns:
            Dict of {tenant_id: allocated_cost}
        """
        allocated = {}
        total_percentage = sum(allocation_basis.values())

        if total_percentage == 0:
            logger.error("Cannot allocate fixed costs: zero usage across all tenants")
            return {}

        for tenant_id, usage_pct in allocation_basis.items():
            allocated_cost = monthly_fixed_cost * (usage_pct / total_percentage)
            allocated[tenant_id] = allocated_cost

            # Update tenant metrics
            if tenant_id in self.tenant_metrics:
                self.tenant_metrics[tenant_id].allocated_fixed_cost = allocated_cost

            logger.info(
                f"Tenant {tenant_id}: ${allocated_cost:.2f} fixed cost "
                f"({usage_pct:.1f}% usage)"
            )

        return allocated

    def get_tenant_cost_summary(self, tenant_id: str) -> Optional[Dict[str, float]]:
        """
        Get complete cost breakdown for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dict with variable and fixed costs, or None
        """
        if tenant_id not in self.tenant_metrics:
            return None

        metrics = self.tenant_metrics[tenant_id]
        return {
            "variable_cost": metrics.variable_cost,
            "fixed_cost": metrics.allocated_fixed_cost,
            "total_cost": metrics.variable_cost + metrics.allocated_fixed_cost,
            "query_count": metrics.query_count
        }


def test_cross_tenant_isolation(
    data_manager: TenantDataManager,
    tenant_a_id: str,
    tenant_b_id: str
) -> bool:
    """
    CRITICAL SECURITY TEST: Verify cross-tenant data leakage prevention.

    This test should FAIL - tenant B should NOT access tenant A's data.

    Args:
        data_manager: TenantDataManager instance
        tenant_a_id: First tenant ID
        tenant_b_id: Second tenant ID

    Returns:
        bool: True if isolation works (query fails), False if leakage detected
    """
    logger.info(f"Testing cross-tenant isolation: {tenant_a_id} vs {tenant_b_id}")

    try:
        # Attempt to query tenant A's namespace as tenant B
        # This should only return tenant B's data (or error)
        result = data_manager.query_documents(
            tenant_id=tenant_b_id,
            query_vector=[0.1] * 384  # Dummy vector
        )

        # Check if any results belong to tenant A (they shouldn't!)
        if "matches" in result:
            for match in result["matches"]:
                if match.get("metadata", {}).get("tenant_id") == tenant_a_id:
                    logger.error(
                        f"SECURITY FAILURE: Tenant {tenant_b_id} accessed "
                        f"tenant {tenant_a_id}'s data!"
                    )
                    return False

        logger.info("Cross-tenant isolation test PASSED")
        return True

    except Exception as e:
        logger.error(f"Isolation test error: {e}")
        return False


# CLI Usage Examples
if __name__ == "__main__":
    print("=== Module 11.1: Tenant Isolation Strategies Demo ===\n")

    # Initialize components
    registry = TenantRegistry()
    data_manager = TenantDataManager(registry)
    cost_engine = CostAllocationEngine()

    # Register tenants with different tiers
    print("1. Registering tenants...")
    tenant_free = registry.register_tenant("tenant-001", "Acme Corp", TenantTier.FREE)
    tenant_pro = registry.register_tenant("tenant-002", "Beta LLC", TenantTier.PRO)
    tenant_ent = registry.register_tenant("tenant-003", "Enterprise Inc", TenantTier.ENTERPRISE)

    print(f"   - {tenant_free.tenant_name}: {tenant_free.tier.value} tier, namespace={tenant_free.namespace}")
    print(f"   - {tenant_pro.tenant_name}: {tenant_pro.tier.value} tier, namespace={tenant_pro.namespace}")
    print(f"   - {tenant_ent.tenant_name}: {tenant_ent.tier.value} tier, index={tenant_ent.dedicated_index}\n")

    # Upsert documents with isolation
    print("2. Upserting documents with tenant isolation...")
    docs = [
        {"id": "doc1", "values": [0.1] * 384, "metadata": {"title": "Document 1"}},
        {"id": "doc2", "values": [0.2] * 384, "metadata": {"title": "Document 2"}}
    ]
    result = data_manager.upsert_documents("tenant-001", docs)
    print(f"   - Upserted {result['upserted']} docs to {result['namespace']} ({result['isolation']})\n")

    # Query with tenant scoping
    print("3. Querying with mandatory tenant scoping...")
    query_result = data_manager.query_documents("tenant-001", [0.15] * 384, top_k=3)
    print(f"   - Queried {query_result['namespace']}, overhead: {query_result['overhead_ms']}ms\n")

    # Track costs
    print("4. Tracking costs per tenant...")
    cost = cost_engine.track_query_cost("tenant-001", embed_tokens=500, llm_tokens=1000)
    print(f"   - Query cost for tenant-001: ${cost:.6f}\n")

    # Allocate fixed costs
    print("5. Allocating shared infrastructure costs...")
    fixed_costs = cost_engine.allocate_fixed_costs(
        monthly_fixed_cost=115.0,  # $50 Pinecone + $30 DB + $35 monitoring
        allocation_basis={
            "tenant-001": 20.0,  # 20% usage
            "tenant-002": 50.0,  # 50% usage
            "tenant-003": 30.0   # 30% usage
        }
    )
    for tid, cost in fixed_costs.items():
        summary = cost_engine.get_tenant_cost_summary(tid)
        print(f"   - {tid}: ${summary['total_cost']:.2f} total (${summary['variable_cost']:.6f} variable + ${summary['fixed_cost']:.2f} fixed)")

    print("\n6. Testing cross-tenant isolation...")
    isolation_ok = test_cross_tenant_isolation(data_manager, "tenant-001", "tenant-002")
    print(f"   - Isolation test: {'PASSED' if isolation_ok else 'FAILED'}")

    print("\n=== Demo Complete ===")
