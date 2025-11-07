"""
Module 13: Enterprise RAG SaaS - Complete Multi-Tenant Integration

This module implements a production-ready, multi-tenant Compliance Copilot SaaS platform
that integrates configuration management, tenant context propagation, orchestration,
failure isolation, and resource attribution.

Architecture:
- Configuration Layer: Pydantic + Dynaconf for system/tenant settings
- Tenant Context: OpenTelemetry baggage + ContextVar for identity propagation
- Orchestration: ComplianceCopilotSaaS class coordinates workflows
- Resource Attribution: Track operations per tenant for billing

Critical Trade-offs:
- Works for 5-100 paying customers with >500ms P95 latency
- Breaks with <5 customers (overhead), <100 market (overengineered), <500ms latency needs
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from contextvars import ContextVar
from datetime import datetime
from enum import Enum

# Type hints for external dependencies (graceful degradation if not installed)
try:
    from pydantic import BaseModel, Field, validator
    from opentelemetry import trace, baggage
    from opentelemetry.context import Context
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    BaseModel = object
    Field = lambda *args, **kwargs: None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Tenant Context Management
# ============================================================================

# ContextVar for tenant propagation across async boundaries
_tenant_context: ContextVar[Optional[str]] = ContextVar('tenant_id', default=None)


class TenantContext:
    """
    Manages tenant identity propagation through async operations.
    Uses ContextVar for Python async and OpenTelemetry baggage for distributed tracing.
    """

    @staticmethod
    def set_tenant(tenant_id: str) -> None:
        """Set current tenant context."""
        _tenant_context.set(tenant_id)
        if TELEMETRY_AVAILABLE:
            try:
                ctx = baggage.set_baggage("tenant_id", tenant_id)
                logger.debug(f"Set tenant context: {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to set OpenTelemetry baggage: {e}")

    @staticmethod
    def get_tenant() -> Optional[str]:
        """Get current tenant ID from context."""
        tenant_id = _tenant_context.get()
        if not tenant_id and TELEMETRY_AVAILABLE:
            try:
                tenant_id = baggage.get_baggage("tenant_id")
            except Exception as e:
                logger.warning(f"Failed to get OpenTelemetry baggage: {e}")
        return tenant_id

    @staticmethod
    def clear_tenant() -> None:
        """Clear tenant context (for testing/cleanup)."""
        _tenant_context.set(None)


# ============================================================================
# Configuration Models
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval strategy options."""
    BASIC = "basic"
    HYBRID = "hybrid"
    AGENTIC = "agentic"


class ModelTier(str, Enum):
    """LLM model tier selection."""
    GPT35 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"


@dataclass
class ResourceLimits:
    """Per-tenant resource limits for cost control."""
    max_queries_per_hour: int = 100
    max_tokens_per_query: int = 4096
    max_concurrent_requests: int = 5
    max_documents_per_tenant: int = 10000
    timeout_seconds: float = 30.0


@dataclass
class TenantConfig:
    """
    Configuration cascade: System defaults → Tenant defaults → Query overrides.
    """
    tenant_id: str
    model_tier: ModelTier = ModelTier.GPT35
    retrieval_mode: RetrievalMode = RetrievalMode.BASIC
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    pinecone_namespace: Optional[str] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-generate Pinecone namespace from tenant_id if not provided."""
        if self.pinecone_namespace is None:
            self.pinecone_namespace = f"tenant_{self.tenant_id}"


# ============================================================================
# Usage Tracking & Billing
# ============================================================================

@dataclass
class UsageRecord:
    """Track operation for billing attribution."""
    tenant_id: str
    operation: str
    timestamp: datetime
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class UsageTracker:
    """
    Async usage tracking to avoid blocking request paths.

    Common Failure: Async billing lag - operations complete but billing delayed.
    Fix: Implement background worker with retry queue.
    """

    def __init__(self):
        self.records: List[UsageRecord] = []
        self._lock = asyncio.Lock()

    async def track(self, record: UsageRecord) -> None:
        """Record usage asynchronously."""
        async with self._lock:
            self.records.append(record)
            logger.info(f"[{record.tenant_id}] {record.operation}: {record.tokens_used} tokens, "
                       f"{record.latency_ms:.2f}ms, success={record.success}")

    def get_tenant_usage(self, tenant_id: str, hours: int = 1) -> List[UsageRecord]:
        """Retrieve usage records for a tenant within time window."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        return [
            r for r in self.records
            if r.tenant_id == tenant_id and r.timestamp.timestamp() > cutoff
        ]

    def calculate_cost(self, tenant_id: str, hours: int = 24) -> Dict[str, float]:
        """
        Calculate approximate costs for a tenant.

        Cost breakdown example (monthly):
        - Database: $50-200/month
        - Vector store: $70-500/month
        - LLM APIs: $100-2000/month
        - Observability: $50-300/month
        """
        records = self.get_tenant_usage(tenant_id, hours)
        total_tokens = sum(r.tokens_used for r in records if r.success)

        # Rough cost estimates (USD per 1K tokens)
        cost_per_1k = {
            ModelTier.GPT35: 0.002,
            ModelTier.GPT4: 0.03,
            ModelTier.GPT4_TURBO: 0.01
        }

        return {
            "total_tokens": total_tokens,
            "estimated_llm_cost": (total_tokens / 1000) * 0.01,  # Avg estimate
            "total_requests": len(records),
            "success_rate": sum(1 for r in records if r.success) / len(records) if records else 0
        }


# ============================================================================
# Configuration Management
# ============================================================================

class ConfigManager:
    """
    Unified configuration using Dynaconf-style cascade.
    Supports system defaults, tenant-specific overrides, and runtime config.

    Common Failure: Cache race conditions causing cross-tenant leakage.
    Fix: Use thread-safe caching with tenant-scoped locks.
    """

    def __init__(self):
        self._tenant_configs: Dict[str, TenantConfig] = {}
        self._system_defaults = TenantConfig(
            tenant_id="system",
            model_tier=ModelTier.GPT35,
            retrieval_mode=RetrievalMode.BASIC
        )
        self._lock = asyncio.Lock()

    async def load_tenant_config(self, tenant_id: str) -> TenantConfig:
        """
        Load tenant configuration with cascade.
        In production, this would query a database.
        """
        async with self._lock:
            if tenant_id not in self._tenant_configs:
                logger.info(f"Loading config for tenant: {tenant_id}")
                # Simulate DB lookup with system defaults
                self._tenant_configs[tenant_id] = TenantConfig(
                    tenant_id=tenant_id,
                    model_tier=self._system_defaults.model_tier,
                    retrieval_mode=self._system_defaults.retrieval_mode,
                    resource_limits=ResourceLimits()
                )
            return self._tenant_configs[tenant_id]

    def update_tenant_config(self, tenant_id: str, **kwargs) -> None:
        """Update tenant configuration at runtime."""
        if tenant_id in self._tenant_configs:
            config = self._tenant_configs[tenant_id]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logger.info(f"Updated config for tenant {tenant_id}: {kwargs}")
        else:
            logger.error(f"Tenant {tenant_id} not loaded")

    def set_system_defaults(self, **kwargs) -> None:
        """Update system-wide defaults."""
        for key, value in kwargs.items():
            if hasattr(self._system_defaults, key):
                setattr(self._system_defaults, key, value)
        logger.info(f"Updated system defaults: {kwargs}")


# ============================================================================
# RAG Components (Simulated)
# ============================================================================

class VectorStore:
    """
    Simulated Pinecone vector store with namespace isolation.

    Common Failure: Cascading rate limits when one tenant consumes quota.
    Fix: Per-tenant rate limiting + circuit breakers.
    """

    def __init__(self):
        self._namespaces: Dict[str, List[Dict[str, Any]]] = {}
        self._rate_limits: Dict[str, List[float]] = {}

    async def query(
        self,
        namespace: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query vectors in tenant namespace."""
        tenant_id = TenantContext.get_tenant()

        # Rate limit check (100 req/min per tenant)
        if tenant_id:
            now = time.time()
            self._rate_limits.setdefault(tenant_id, [])
            recent = [t for t in self._rate_limits[tenant_id] if now - t < 60]

            if len(recent) >= 100:
                logger.error(f"Rate limit exceeded for tenant {tenant_id}")
                raise Exception(f"Rate limit exceeded for tenant {tenant_id}")

            self._rate_limits[tenant_id] = recent + [now]

        # Simulate retrieval
        await asyncio.sleep(0.05)  # Simulate network latency

        docs = self._namespaces.get(namespace, [])
        logger.info(f"Retrieved {len(docs[:top_k])} docs from namespace: {namespace}")
        return docs[:top_k]

    async def upsert(
        self,
        namespace: str,
        vectors: List[Dict[str, Any]]
    ) -> None:
        """Insert/update vectors in tenant namespace."""
        self._namespaces.setdefault(namespace, [])
        self._namespaces[namespace].extend(vectors)
        logger.info(f"Upserted {len(vectors)} vectors to namespace: {namespace}")


class LLMClient:
    """
    Simulated LLM client with model tier support.

    Common Failure: Connection pool exhaustion during bulk operations.
    Fix: Connection pooling + request batching.
    """

    def __init__(self, model: ModelTier):
        self.model = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate completion from LLM."""
        # Simulate LLM latency based on model
        latency = {
            ModelTier.GPT35: 0.5,
            ModelTier.GPT4: 2.0,
            ModelTier.GPT4_TURBO: 1.0
        }
        await asyncio.sleep(latency.get(self.model, 1.0))

        # Simulate token usage
        tokens_used = min(max_tokens, len(prompt.split()) * 2)

        return {
            "model": self.model.value,
            "response": f"[Simulated response from {self.model.value}]",
            "tokens_used": tokens_used,
            "finish_reason": "stop"
        }


# ============================================================================
# Orchestration
# ============================================================================

class ComplianceCopilotSaaS:
    """
    Central orchestration class coordinating multi-tenant RAG workflows.

    Workflow:
    1. Authentication verification
    2. Config loading (cascade)
    3. Component initialization
    4. Execution with tenant context
    5. Post-processing (metrics/billing)

    Failure Isolation: Single tenant issues don't impact others.
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        usage_tracker: Optional[UsageTracker] = None
    ):
        self.config_manager = config_manager or ConfigManager()
        self.usage_tracker = usage_tracker or UsageTracker()
        self.vector_store = VectorStore()
        logger.info("ComplianceCopilotSaaS initialized")

    async def query(
        self,
        tenant_id: str,
        query_text: str,
        override_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute RAG query with full multi-tenant support.

        Args:
            tenant_id: Tenant identifier
            query_text: User query
            override_config: Query-level config overrides

        Returns:
            Response with answer, sources, metadata
        """
        start_time = time.time()

        # Set tenant context for entire request
        TenantContext.set_tenant(tenant_id)

        try:
            # 1. Load tenant configuration
            config = await self.config_manager.load_tenant_config(tenant_id)

            # 2. Apply query-level overrides (config cascade)
            if override_config:
                for key, value in override_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            logger.info(f"[{tenant_id}] Query: '{query_text[:50]}...' "
                       f"(model={config.model_tier.value}, mode={config.retrieval_mode.value})")

            # 3. Check resource limits
            recent_usage = self.usage_tracker.get_tenant_usage(tenant_id, hours=1)
            if len(recent_usage) >= config.resource_limits.max_queries_per_hour:
                raise Exception(f"Query quota exceeded: {len(recent_usage)}/hour")

            # 4. Initialize LLM client with tenant's model tier
            llm = LLMClient(config.model_tier)

            # 5. Retrieve context (namespace isolation)
            query_vector = [0.1] * 768  # Simulated embedding
            retrieved_docs = await self.vector_store.query(
                namespace=config.pinecone_namespace,
                query_vector=query_vector,
                top_k=5
            )

            # 6. Generate response
            context = "\n".join([doc.get("text", "") for doc in retrieved_docs])
            prompt = f"Context:\n{context}\n\nQuery: {query_text}\n\nAnswer:"

            llm_response = await llm.generate(
                prompt=prompt,
                max_tokens=config.resource_limits.max_tokens_per_query
            )

            # 7. Track usage for billing
            latency_ms = (time.time() - start_time) * 1000
            await self.usage_tracker.track(UsageRecord(
                tenant_id=tenant_id,
                operation="query",
                timestamp=datetime.now(),
                tokens_used=llm_response["tokens_used"],
                latency_ms=latency_ms,
                success=True
            ))

            return {
                "answer": llm_response["response"],
                "sources": [{"id": i, "score": 0.9-i*0.1} for i in range(len(retrieved_docs))],
                "metadata": {
                    "tenant_id": tenant_id,
                    "model": llm_response["model"],
                    "tokens_used": llm_response["tokens_used"],
                    "latency_ms": round(latency_ms, 2),
                    "retrieval_mode": config.retrieval_mode.value
                }
            }

        except Exception as e:
            logger.error(f"[{tenant_id}] Query failed: {str(e)}")

            # Track failure for billing/monitoring
            latency_ms = (time.time() - start_time) * 1000
            await self.usage_tracker.track(UsageRecord(
                tenant_id=tenant_id,
                operation="query",
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                success=False,
                error_message=str(e)
            ))

            raise

        finally:
            TenantContext.clear_tenant()

    async def ingest_documents(
        self,
        tenant_id: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest documents for a tenant with namespace isolation.

        Args:
            tenant_id: Tenant identifier
            documents: List of docs with 'text' and optional 'metadata'

        Returns:
            Ingestion summary
        """
        TenantContext.set_tenant(tenant_id)

        try:
            config = await self.config_manager.load_tenant_config(tenant_id)

            # Check document limit
            if len(documents) > config.resource_limits.max_documents_per_tenant:
                raise Exception(f"Document limit exceeded: {len(documents)} docs")

            # Simulate embedding + upsert
            vectors = [
                {
                    "id": f"{tenant_id}_{i}",
                    "vector": [0.1] * 768,  # Simulated
                    "metadata": doc.get("metadata", {}),
                    "text": doc["text"]
                }
                for i, doc in enumerate(documents)
            ]

            await self.vector_store.upsert(
                namespace=config.pinecone_namespace,
                vectors=vectors
            )

            logger.info(f"[{tenant_id}] Ingested {len(documents)} documents")

            return {
                "tenant_id": tenant_id,
                "documents_ingested": len(documents),
                "namespace": config.pinecone_namespace,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"[{tenant_id}] Ingestion failed: {str(e)}")
            raise

        finally:
            TenantContext.clear_tenant()

    def get_tenant_metrics(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """Retrieve metrics and costs for a tenant."""
        usage = self.usage_tracker.get_tenant_usage(tenant_id, hours)
        costs = self.usage_tracker.calculate_cost(tenant_id, hours)

        return {
            "tenant_id": tenant_id,
            "time_window_hours": hours,
            "total_queries": len([r for r in usage if r.operation == "query"]),
            "successful_queries": len([r for r in usage if r.operation == "query" and r.success]),
            "avg_latency_ms": sum(r.latency_ms for r in usage) / len(usage) if usage else 0,
            "costs": costs
        }


# ============================================================================
# CLI Usage Examples
# ============================================================================

async def main_demo():
    """Demonstrate multi-tenant RAG SaaS capabilities."""

    print("=" * 60)
    print("Module 13: Enterprise RAG SaaS Demo")
    print("=" * 60)

    # Initialize system
    copilot = ComplianceCopilotSaaS()

    # Configure system defaults
    copilot.config_manager.set_system_defaults(
        model_tier=ModelTier.GPT35,
        retrieval_mode=RetrievalMode.BASIC
    )

    # Simulate 3 tenants
    tenants = ["acme_corp", "beta_inc", "gamma_labs"]

    # Ingest documents for each tenant (namespace isolation)
    print("\n1. Ingesting documents per tenant...")
    for tenant in tenants:
        result = await copilot.ingest_documents(
            tenant_id=tenant,
            documents=[
                {"text": f"Document 1 for {tenant}"},
                {"text": f"Document 2 for {tenant}"},
                {"text": f"Document 3 for {tenant}"}
            ]
        )
        print(f"  ✓ {tenant}: {result['documents_ingested']} docs in ns={result['namespace']}")

    # Configure tenant-specific settings
    print("\n2. Configuring tenant tiers...")
    copilot.config_manager.update_tenant_config(
        "acme_corp",
        model_tier=ModelTier.GPT4,
        retrieval_mode=RetrievalMode.HYBRID
    )
    print("  ✓ acme_corp: Upgraded to GPT-4 + Hybrid retrieval")

    # Execute queries
    print("\n3. Executing queries...")
    for tenant in tenants:
        response = await copilot.query(
            tenant_id=tenant,
            query_text=f"What are the compliance requirements for {tenant}?"
        )
        print(f"  ✓ {tenant}: {response['metadata']['latency_ms']}ms, "
              f"model={response['metadata']['model']}")

    # Show metrics
    print("\n4. Tenant Metrics (Last Hour):")
    for tenant in tenants:
        metrics = copilot.get_tenant_metrics(tenant, hours=1)
        print(f"  {tenant}:")
        print(f"    - Queries: {metrics['successful_queries']}/{metrics['total_queries']}")
        print(f"    - Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"    - Est. Cost: ${metrics['costs']['estimated_llm_cost']:.4f}")

    print("\n✅ Demo completed successfully!")
    print("\nDecision Card:")
    print("  Use this when: 5-100 customers, >500ms latency OK, need multi-tenancy")
    print("  Avoid when: <5 customers, <100 market, <500ms latency, no DevOps team")


if __name__ == "__main__":
    asyncio.run(main_demo())
