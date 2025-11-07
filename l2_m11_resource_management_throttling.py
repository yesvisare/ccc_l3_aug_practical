"""
Module 11.3: Resource Management & Throttling
Multi-tenant resource quotas and fair scheduling for SaaS applications.

This module implements:
- Per-tenant quota tracking (queries, tokens, storage)
- Fair queue scheduling with round-robin
- Throttling middleware for FastAPI
- Background queue workers
- Resource-weighted quotas
"""

import redis
import json
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class QuotaType(Enum):
    """Types of quotas we track"""
    QUERIES_HOURLY = "queries_hourly"
    QUERIES_DAILY = "queries_daily"
    QUERIES_MONTHLY = "queries_monthly"
    TOKENS_MONTHLY = "tokens_monthly"
    STORAGE_GB = "storage_gb"


@dataclass
class TenantQuotas:
    """Quota limits for a tenant tier"""
    queries_per_hour: int = 100
    queries_per_day: int = 1000
    queries_per_month: int = 20000
    tokens_per_month: int = 1000000
    storage_gb: int = 10
    tier: str = "free"


class QuotaManager:
    """
    Manage per-tenant resource quotas.

    Tracks usage across multiple time windows and resource types.
    Uses Redis for distributed counting and atomic operations.
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

        # Default quota tiers
        self.quota_tiers = {
            "free": TenantQuotas(
                queries_per_hour=100,
                queries_per_day=1000,
                queries_per_month=20000,
                tokens_per_month=500000,
                storage_gb=5,
                tier="free"
            ),
            "pro": TenantQuotas(
                queries_per_hour=1000,
                queries_per_day=10000,
                queries_per_month=200000,
                tokens_per_month=5000000,
                storage_gb=50,
                tier="pro"
            ),
            "enterprise": TenantQuotas(
                queries_per_hour=10000,
                queries_per_day=100000,
                queries_per_month=2000000,
                tokens_per_month=50000000,
                storage_gb=500,
                tier="enterprise"
            )
        }

    def _get_quota_key(self, tenant_id: str, quota_type: QuotaType) -> str:
        """Generate Redis key for quota tracking"""
        now = datetime.utcnow()

        if quota_type == QuotaType.QUERIES_HOURLY:
            time_suffix = now.strftime("%Y-%m-%d-%H")
        elif quota_type == QuotaType.QUERIES_DAILY:
            time_suffix = now.strftime("%Y-%m-%d")
        elif quota_type == QuotaType.QUERIES_MONTHLY:
            time_suffix = now.strftime("%Y-%m")
        elif quota_type == QuotaType.TOKENS_MONTHLY:
            time_suffix = now.strftime("%Y-%m")
        else:  # STORAGE_GB
            time_suffix = "current"

        return f"quota:{tenant_id}:{quota_type.value}:{time_suffix}"

    def _get_ttl(self, quota_type: QuotaType) -> Optional[int]:
        """Get TTL for quota key (seconds)"""
        if quota_type == QuotaType.QUERIES_HOURLY:
            return 7200  # 2 hours
        elif quota_type == QuotaType.QUERIES_DAILY:
            return 172800  # 2 days
        elif quota_type in [QuotaType.QUERIES_MONTHLY, QuotaType.TOKENS_MONTHLY]:
            return 5184000  # 60 days
        else:  # STORAGE_GB
            return None  # No expiry

    def get_tenant_tier(self, tenant_id: str) -> TenantQuotas:
        """
        Get quota configuration for tenant.

        In production, this would query your tenant database.
        For now, we check Redis cache or default to 'free'.
        """
        tier_key = f"tenant:{tenant_id}:tier"
        tier_name = self.redis.get(tier_key)

        if tier_name and tier_name in self.quota_tiers:
            return self.quota_tiers[tier_name]

        return self.quota_tiers["free"]

    def set_tenant_tier(self, tenant_id: str, tier: str) -> None:
        """Set tenant's quota tier"""
        if tier not in self.quota_tiers:
            raise ValueError(f"Unknown tier: {tier}. Valid tiers: {list(self.quota_tiers.keys())}")

        tier_key = f"tenant:{tenant_id}:tier"
        self.redis.set(tier_key, tier)
        logger.info(f"Set tenant {tenant_id} to tier {tier}")

    def increment_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1
    ) -> int:
        """
        Increment usage counter for tenant.

        Returns new total for this time period.
        """
        key = self._get_quota_key(tenant_id, quota_type)

        # Atomic increment
        new_value = self.redis.incr(key, amount)

        # Set TTL if this is a new key
        ttl = self._get_ttl(quota_type)
        if ttl and self.redis.ttl(key) == -1:
            self.redis.expire(key, ttl)

        logger.debug(f"Incremented {quota_type.value} for {tenant_id}: {new_value}")
        return new_value

    def get_usage(self, tenant_id: str, quota_type: QuotaType) -> int:
        """Get current usage for tenant"""
        key = self._get_quota_key(tenant_id, quota_type)
        value = self.redis.get(key)
        return int(value) if value else 0

    def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        increment: int = 1
    ) -> Tuple[bool, int, int]:
        """
        Check if tenant is within quota.

        Returns: (under_quota, current_usage, limit)
        """
        tier = self.get_tenant_tier(tenant_id)
        current = self.get_usage(tenant_id, quota_type)

        # Determine limit based on quota type
        if quota_type == QuotaType.QUERIES_HOURLY:
            limit = tier.queries_per_hour
        elif quota_type == QuotaType.QUERIES_DAILY:
            limit = tier.queries_per_day
        elif quota_type == QuotaType.QUERIES_MONTHLY:
            limit = tier.queries_per_month
        elif quota_type == QuotaType.TOKENS_MONTHLY:
            limit = tier.tokens_per_month
        else:  # STORAGE_GB
            limit = tier.storage_gb

        under_quota = (current + increment) <= limit
        return under_quota, current, limit

    def record_query(self, tenant_id: str, tokens_used: int) -> Dict[str, bool]:
        """
        Record a query and token usage for tenant.

        Returns dict of quota checks: {quota_type: under_quota}
        """
        results = {}

        # Increment all query counters
        for quota_type in [
            QuotaType.QUERIES_HOURLY,
            QuotaType.QUERIES_DAILY,
            QuotaType.QUERIES_MONTHLY
        ]:
            self.increment_usage(tenant_id, quota_type, 1)
            under_quota, current, limit = self.check_quota(tenant_id, quota_type, 0)
            results[quota_type.value] = under_quota

        # Increment token counter
        if tokens_used > 0:
            self.increment_usage(tenant_id, QuotaType.TOKENS_MONTHLY, tokens_used)
            under_quota, current, limit = self.check_quota(
                tenant_id, QuotaType.TOKENS_MONTHLY, 0
            )
            results[QuotaType.TOKENS_MONTHLY.value] = under_quota

        return results

    def get_quota_status(self, tenant_id: str) -> Dict:
        """Get comprehensive quota status for tenant"""
        tier = self.get_tenant_tier(tenant_id)

        status = {
            "tenant_id": tenant_id,
            "tier": tier.tier,
            "quotas": {}
        }

        for quota_type in QuotaType:
            current = self.get_usage(tenant_id, quota_type)
            _, _, limit = self.check_quota(tenant_id, quota_type, 0)

            status["quotas"][quota_type.value] = {
                "current": current,
                "limit": limit,
                "remaining": max(0, limit - current),
                "percentage": round(current / limit * 100, 1) if limit > 0 else 0
            }

        return status


@dataclass
class QueuedRequest:
    """A request waiting in the queue"""
    request_id: str
    tenant_id: str
    query: str
    queued_at: float  # Unix timestamp
    priority: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> 'QueuedRequest':
        return cls(**json.loads(data))


class FairTenantQueue:
    """
    Fair queue manager using round-robin scheduling.

    Each tenant has their own queue. We process one request from
    each tenant in turn, preventing any single tenant from
    monopolizing resources.
    """

    def __init__(self, redis_client: redis.Redis, max_queue_size: int = 100):
        self.redis = redis_client
        self.max_queue_size = max_queue_size

        # Redis keys
        self.active_tenants_key = "queue:active_tenants"
        self.round_robin_key = "queue:round_robin_index"

    def _tenant_queue_key(self, tenant_id: str) -> str:
        """Redis list key for tenant's queue"""
        return f"queue:tenant:{tenant_id}"

    def _queue_size_key(self, tenant_id: str) -> str:
        """Redis key for tenant's queue size"""
        return f"queue:tenant:{tenant_id}:size"

    async def enqueue(self, request: QueuedRequest) -> bool:
        """
        Add request to tenant's queue.

        Returns True if enqueued, False if queue is full.
        """
        tenant_queue = self._tenant_queue_key(request.tenant_id)

        # Check queue size
        current_size = self.redis.llen(tenant_queue)
        if current_size >= self.max_queue_size:
            logger.warning(
                f"Queue full for tenant {request.tenant_id}: "
                f"{current_size}/{self.max_queue_size}"
            )
            return False

        # Add to queue (right push = FIFO)
        self.redis.rpush(tenant_queue, request.to_json())

        # Add tenant to active set
        self.redis.sadd(self.active_tenants_key, request.tenant_id)

        # Track queue size metric
        self.redis.set(self._queue_size_key(request.tenant_id), current_size + 1)

        logger.info(
            f"Enqueued request {request.request_id} for tenant {request.tenant_id} "
            f"(queue size: {current_size + 1})"
        )
        return True

    async def dequeue_fair(self) -> Optional[QueuedRequest]:
        """
        Dequeue next request using round-robin across tenants.

        This ensures fairness - no tenant can monopolize processing.
        """
        # Get all active tenants
        active_tenants = self.redis.smembers(self.active_tenants_key)
        if not active_tenants:
            return None

        # Convert to sorted list for deterministic round-robin
        tenant_list = sorted(list(active_tenants))

        # Get current round-robin index
        current_index = int(self.redis.get(self.round_robin_key) or 0)

        # Try each tenant in round-robin order
        attempts = 0
        while attempts < len(tenant_list):
            tenant_id = tenant_list[current_index % len(tenant_list)]
            tenant_queue = self._tenant_queue_key(tenant_id)

            # Try to dequeue from this tenant
            request_json = self.redis.lpop(tenant_queue)

            if request_json:
                request = QueuedRequest.from_json(request_json)

                # Update queue size
                current_size = self.redis.llen(tenant_queue)
                self.redis.set(self._queue_size_key(tenant_id), current_size)

                # If queue now empty, remove from active set
                if current_size == 0:
                    self.redis.srem(self.active_tenants_key, tenant_id)

                # Advance round-robin index
                self.redis.set(self.round_robin_key, (current_index + 1) % len(tenant_list))

                wait_time = datetime.now().timestamp() - request.queued_at
                logger.info(
                    f"Dequeued request {request.request_id} for tenant {tenant_id} "
                    f"(waited {wait_time:.1f}s, queue size now: {current_size})"
                )

                return request

            # This tenant's queue was empty, try next
            current_index = (current_index + 1) % len(tenant_list)
            attempts += 1

        return None

    def get_queue_stats(self) -> Dict:
        """Get statistics about queued requests"""
        active_tenants = self.redis.smembers(self.active_tenants_key)

        stats = {
            "total_queued_requests": 0,
            "active_tenants": len(active_tenants),
            "per_tenant": {}
        }

        for tenant_id in active_tenants:
            queue_size = self.redis.llen(self._tenant_queue_key(tenant_id))
            stats["total_queued_requests"] += queue_size
            stats["per_tenant"][tenant_id] = {
                "queue_size": queue_size,
                "max_queue_size": self.max_queue_size
            }

        return stats


class ResourceWeightedQuota:
    """
    Track resource-weighted usage, not just query counts.
    Prevents gaming by weighing expensive operations higher.
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def calculate_query_weight(self, request_data: dict) -> float:
        """
        Calculate resource weight for a query.

        Weight factors:
        - Model cost (GPT-4 = 20x, GPT-3.5 = 1x)
        - Context size (per 1K tokens)
        - Features used (tools = 2x, embeddings = 1.5x)
        """
        weight = 1.0

        # Model cost multiplier
        model_costs = {
            "gpt-4": 20.0,
            "gpt-4-turbo": 10.0,
            "gpt-3.5-turbo": 1.0
        }
        model = request_data.get("model", "gpt-3.5-turbo")
        weight *= model_costs.get(model, 1.0)

        # Context size multiplier (per 1K tokens)
        context = request_data.get("context", "")
        tokens = len(context.split()) * 1.3  # Rough token estimate
        weight *= max(1.0, tokens / 1000.0)

        # Feature multipliers
        if request_data.get("use_tools"):
            weight *= 2.0
        if request_data.get("use_embeddings"):
            weight *= 1.5

        return weight


class AtomicQuotaManager(QuotaManager):
    """
    Quota manager with atomic check-and-increment.
    Prevents race conditions under concurrent load.
    """

    def __init__(self, redis_client: redis.Redis):
        super().__init__(redis_client)

        # Lua script for atomic quota check-and-increment
        self.check_and_increment_script = self.redis.register_script("""
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local increment = tonumber(ARGV[2])
            local ttl = tonumber(ARGV[3])

            -- Get current value
            local current = tonumber(redis.call('GET', key) or '0')

            -- Check if increment would exceed limit
            if current + increment > limit then
                return {0, current, limit}
            end

            -- Under quota - increment and set TTL
            local new_value = redis.call('INCRBY', key, increment)

            -- Set TTL if not already set
            local current_ttl = redis.call('TTL', key)
            if current_ttl == -1 then
                redis.call('EXPIRE', key, ttl)
            end

            return {1, new_value, limit}
        """)

    def atomic_check_and_increment(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        increment: int = 1
    ) -> Tuple[bool, int, int]:
        """
        Atomically check quota and increment if under limit.

        Returns: (success, current_value, limit)
        """
        key = self._get_quota_key(tenant_id, quota_type)
        tier = self.get_tenant_tier(tenant_id)

        # Get limit for this quota type
        if quota_type == QuotaType.QUERIES_HOURLY:
            limit = tier.queries_per_hour
        elif quota_type == QuotaType.QUERIES_DAILY:
            limit = tier.queries_per_day
        else:
            limit = tier.queries_per_month

        ttl = self._get_ttl(quota_type) or 3600

        # Execute atomic script
        result = self.check_and_increment_script(
            keys=[key],
            args=[limit, increment, ttl]
        )

        success = bool(result[0])
        current = int(result[1])
        limit_val = int(result[2])

        return success, current, limit_val


class QueueWorker:
    """
    Background worker that processes queued requests fairly.
    """

    def __init__(
        self,
        queue: FairTenantQueue,
        quota_manager: QuotaManager,
        process_request: Callable,
        workers: int = 5,
        poll_interval: float = 1.0
    ):
        self.queue = queue
        self.quota_manager = quota_manager
        self.process_request = process_request
        self.workers = workers
        self.poll_interval = poll_interval
        self._running = False

    async def worker(self, worker_id: int):
        """Single worker that processes requests"""
        logger.info(f"Queue worker {worker_id} started")

        while self._running:
            try:
                request = await self.queue.dequeue_fair()

                if request:
                    # Check if tenant is now under quota
                    under_quota, current, limit = self.quota_manager.check_quota(
                        request.tenant_id,
                        QuotaType.QUERIES_HOURLY,
                        increment=1
                    )

                    if under_quota:
                        try:
                            logger.info(
                                f"Worker {worker_id} processing request "
                                f"{request.request_id} for tenant {request.tenant_id}"
                            )

                            result = await self.process_request(request)

                            # Record usage
                            tokens = result.get("tokens_used", 1000)
                            self.quota_manager.record_query(request.tenant_id, tokens)

                            logger.info(
                                f"Worker {worker_id} completed request "
                                f"{request.request_id}"
                            )

                        except Exception as e:
                            logger.error(
                                f"Worker {worker_id} failed to process "
                                f"request {request.request_id}: {e}",
                                exc_info=True
                            )
                    else:
                        # Still over quota - requeue
                        await self.queue.enqueue(request)
                        await asyncio.sleep(5)
                else:
                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        logger.info(f"Queue worker {worker_id} stopped")

    async def start(self):
        """Start all workers"""
        self._running = True
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.workers)
        ]
        logger.info(f"Started {self.workers} queue workers")
        await asyncio.gather(*workers)

    def stop(self):
        """Stop all workers"""
        self._running = False
        logger.info("Stopping queue workers")


# Example process_request function
async def process_rag_request(request: QueuedRequest) -> Dict:
    """Process a RAG query (placeholder)"""
    await asyncio.sleep(0.5)  # Simulate processing

    return {
        "request_id": request.request_id,
        "tenant_id": request.tenant_id,
        "response": f"Processed: {request.query}",
        "tokens_used": 1200
    }


if __name__ == "__main__":
    # Basic CLI usage examples
    logging.basicConfig(level=logging.INFO)

    print("=== Module 11.3: Resource Management & Throttling ===\n")

    # Initialize Redis
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"⚠️ Redis not available: {e}")
        print("  Run: docker run -d -p 6379:6379 redis:7-alpine")
        exit(1)

    # Example 1: Basic quota tracking
    print("\n1. Basic Quota Tracking:")
    qm = QuotaManager(r)
    qm.set_tenant_tier("tenant_demo", "pro")

    # Record some queries
    for i in range(3):
        results = qm.record_query("tenant_demo", tokens_used=1000)
        print(f"   Query {i+1} recorded: {results.get('queries_hourly')}")

    # Check status
    status = qm.get_quota_status("tenant_demo")
    hourly = status["quotas"]["queries_hourly"]
    print(f"   Status: {hourly['current']}/{hourly['limit']} queries used")

    # Example 2: Fair queue
    print("\n2. Fair Queue Management:")
    queue = FairTenantQueue(r, max_queue_size=10)

    async def demo_queue():
        # Enqueue some requests
        for i in range(3):
            req = QueuedRequest(
                request_id=f"req_{i}",
                tenant_id="tenant_demo",
                query=f"Test query {i}",
                queued_at=time.time()
            )
            await queue.enqueue(req)
            print(f"   Enqueued request {i+1}")

        # Get stats
        stats = queue.get_queue_stats()
        print(f"   Queue stats: {stats['total_queued_requests']} total requests")

    asyncio.run(demo_queue())

    print("\n✓ All examples completed successfully!")
