"""
Usage Metering & Analytics for SaaS (Module 12.1)

Production-grade usage metering system for multi-tenant SaaS applications.
Implements:
- Event capture with tenant attribution
- Async buffering (<5ms overhead per request)
- ClickHouse storage with materialized views
- Cost calculation and quota enforcement
- Billing export capabilities

Design Philosophy:
- Immutable event log for audit trail
- Non-blocking writes with fallback storage
- 10-100x better query performance vs PostgreSQL on aggregations
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import uuid

from config import Config, get_clickhouse_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class UsageEvent:
    """Single usage event - immutable for audit trail."""
    event_id: str
    tenant_id: str
    event_type: str  # 'query', 'token_input', 'token_output', 'storage'
    quantity: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "event_type": self.event_type,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata),
        }


@dataclass
class TenantQuota:
    """Per-tenant quota configuration."""
    tenant_id: str
    queries_per_day: int
    tokens_per_day: int
    storage_gb: float
    overage_allowed: bool = True


# ============================================================================
# ClickHouse Schema Management
# ============================================================================

class ClickHouseSchema:
    """Manages ClickHouse database schema setup and migrations."""

    @staticmethod
    def get_schema_sql() -> List[str]:
        """
        Returns SQL statements for complete schema setup.

        Schema design:
        - usage_events: Append-only event log, partitioned monthly
        - usage_hourly: Materialized view for real-time aggregations
        - usage_daily: Daily summaries per tenant
        - tenant_quotas: Quota limits and consumption tracking
        """
        return [
            # Main events table with 36-month TTL
            f"""
            CREATE TABLE IF NOT EXISTS usage_events (
                event_id String,
                tenant_id String,
                event_type String,
                quantity Float64,
                timestamp DateTime,
                metadata String,
                date Date DEFAULT toDate(timestamp)
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(date)
            ORDER BY (tenant_id, timestamp)
            TTL date + INTERVAL {Config.RETENTION_MONTHS} MONTH
            """,

            # Hourly aggregation materialized view
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS usage_hourly
            ENGINE = SummingMergeTree()
            PARTITION BY toYYYYMM(hour_start)
            ORDER BY (tenant_id, event_type, hour_start)
            AS SELECT
                tenant_id,
                event_type,
                toStartOfHour(timestamp) as hour_start,
                sum(quantity) as total_quantity,
                count() as event_count
            FROM usage_events
            GROUP BY tenant_id, event_type, hour_start
            """,

            # Daily aggregation materialized view
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS usage_daily
            ENGINE = SummingMergeTree()
            PARTITION BY toYYYYMM(day)
            ORDER BY (tenant_id, day)
            AS SELECT
                tenant_id,
                toDate(timestamp) as day,
                sumIf(quantity, event_type = 'query') as queries,
                sumIf(quantity, event_type = 'token_input') as tokens_input,
                sumIf(quantity, event_type = 'token_output') as tokens_output,
                sumIf(quantity, event_type = 'storage') as storage_gb
            FROM usage_events
            GROUP BY tenant_id, day
            """,

            # Tenant quotas table
            """
            CREATE TABLE IF NOT EXISTS tenant_quotas (
                tenant_id String,
                queries_per_day Int32,
                tokens_per_day Int64,
                storage_gb Float64,
                overage_allowed UInt8,
                updated_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY tenant_id
            """,
        ]

    @staticmethod
    def initialize_schema(client) -> bool:
        """
        Initialize ClickHouse schema.

        Returns:
            True if successful, False otherwise
        """
        if client is None:
            logger.error("Cannot initialize schema: ClickHouse client not available")
            return False

        try:
            for sql in ClickHouseSchema.get_schema_sql():
                client.execute(sql)
            logger.info("ClickHouse schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            return False


# ============================================================================
# Usage Tracker with Async Buffering
# ============================================================================

class UsageTracker:
    """
    Non-blocking usage event tracker with async buffering.

    Design:
    - Batches events (100 or 1-second intervals)
    - <5ms overhead per request
    - Fallback to JSONL files if ClickHouse unavailable
    """

    def __init__(self, client=None):
        self.client = client
        self.buffer: List[UsageEvent] = []
        self.buffer_lock = asyncio.Lock()
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start background flush task."""
        if self.running:
            return
        self.running = True
        self.flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Usage tracker started")

    async def stop(self):
        """Stop tracker and flush remaining events."""
        self.running = False
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        logger.info("Usage tracker stopped")

    async def track(self, event: UsageEvent):
        """
        Track a usage event (non-blocking).

        Args:
            event: UsageEvent to record
        """
        async with self.buffer_lock:
            self.buffer.append(event)

            # Flush if buffer full
            if len(self.buffer) >= Config.BUFFER_SIZE:
                await self._flush_locked()

    async def _periodic_flush(self):
        """Background task to flush buffer periodically."""
        while self.running:
            try:
                await asyncio.sleep(Config.FLUSH_INTERVAL_SECONDS)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def flush(self):
        """Flush buffered events to ClickHouse."""
        async with self.buffer_lock:
            await self._flush_locked()

    async def _flush_locked(self):
        """Internal flush implementation (assumes lock held)."""
        if not self.buffer:
            return

        events_to_flush = self.buffer[:]
        self.buffer.clear()

        # Try ClickHouse first
        if self.client:
            try:
                rows = [
                    (
                        e.event_id,
                        e.tenant_id,
                        e.event_type,
                        e.quantity,
                        e.timestamp,
                        json.dumps(e.metadata),
                    )
                    for e in events_to_flush
                ]
                self.client.execute(
                    "INSERT INTO usage_events (event_id, tenant_id, event_type, quantity, timestamp, metadata) VALUES",
                    rows
                )
                logger.info(f"Flushed {len(events_to_flush)} events to ClickHouse")
                return
            except Exception as e:
                logger.error(f"ClickHouse write failed: {e}, falling back to file")

        # Fallback to file storage
        try:
            with open(Config.FALLBACK_FILE_PATH, "a") as f:
                for event in events_to_flush:
                    f.write(json.dumps(event.to_dict()) + "\n")
            logger.warning(f"Wrote {len(events_to_flush)} events to fallback file")
        except Exception as e:
            logger.error(f"Fallback file write failed: {e}")


# ============================================================================
# Cost Calculation
# ============================================================================

class CostCalculator:
    """Maps usage quantities to billable amounts."""

    @staticmethod
    def calculate_event_cost(event: UsageEvent) -> float:
        """
        Calculate cost for a single event.

        Args:
            event: UsageEvent to price

        Returns:
            Cost in USD
        """
        pricing = Config.PRICING_CONFIG
        event_type = event.event_type
        quantity = event.quantity

        if event_type == "query":
            return quantity * pricing["query"]
        elif event_type == "token_input":
            return (quantity / 1000) * pricing["token_input"]
        elif event_type == "token_output":
            return (quantity / 1000) * pricing["token_output"]
        elif event_type == "storage":
            return quantity * pricing["storage_gb"]
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return 0.0

    @staticmethod
    def calculate_tenant_costs(client, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """
        Calculate total costs for a tenant over a date range.

        Args:
            client: ClickHouse client
            tenant_id: Tenant identifier
            start_date: Start of billing period
            end_date: End of billing period

        Returns:
            Dictionary with cost breakdown by event type
        """
        if client is None:
            return {}

        try:
            query = """
            SELECT
                event_type,
                sum(quantity) as total_quantity
            FROM usage_events
            WHERE tenant_id = %(tenant_id)s
              AND timestamp >= %(start_date)s
              AND timestamp < %(end_date)s
            GROUP BY event_type
            """

            results = client.execute(
                query,
                {
                    "tenant_id": tenant_id,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

            pricing = Config.PRICING_CONFIG
            costs = {}
            total = 0.0

            for event_type, quantity in results:
                if event_type == "query":
                    cost = quantity * pricing["query"]
                elif event_type == "token_input":
                    cost = (quantity / 1000) * pricing["token_input"]
                elif event_type == "token_output":
                    cost = (quantity / 1000) * pricing["token_output"]
                elif event_type == "storage":
                    cost = quantity * pricing["storage_gb"]
                else:
                    cost = 0.0

                costs[event_type] = cost
                total += cost

            costs["total"] = total
            return costs

        except Exception as e:
            logger.error(f"Cost calculation failed: {e}")
            return {}


# ============================================================================
# Quota Management
# ============================================================================

class QuotaManager:
    """Manages tenant quotas and overage detection."""

    def __init__(self, client):
        self.client = client

    def set_quota(self, quota: TenantQuota) -> bool:
        """
        Set or update tenant quota.

        Args:
            quota: TenantQuota configuration

        Returns:
            True if successful
        """
        if self.client is None:
            logger.error("Cannot set quota: ClickHouse client not available")
            return False

        try:
            self.client.execute(
                """
                INSERT INTO tenant_quotas
                (tenant_id, queries_per_day, tokens_per_day, storage_gb, overage_allowed)
                VALUES (%(tenant_id)s, %(queries_per_day)s, %(tokens_per_day)s, %(storage_gb)s, %(overage_allowed)s)
                """,
                {
                    "tenant_id": quota.tenant_id,
                    "queries_per_day": quota.queries_per_day,
                    "tokens_per_day": quota.tokens_per_day,
                    "storage_gb": quota.storage_gb,
                    "overage_allowed": 1 if quota.overage_allowed else 0,
                }
            )
            logger.info(f"Quota set for tenant {quota.tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set quota: {e}")
            return False

    def check_quota(self, tenant_id: str, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check if tenant is within quota limits.

        Args:
            tenant_id: Tenant identifier
            date: Date to check (defaults to today)

        Returns:
            Dictionary with quota status and usage details
        """
        if self.client is None:
            return {"error": "ClickHouse not available"}

        if date is None:
            date = datetime.now()

        try:
            # Get quota limits
            quota_result = self.client.execute(
                "SELECT queries_per_day, tokens_per_day, storage_gb, overage_allowed FROM tenant_quotas WHERE tenant_id = %(tenant_id)s",
                {"tenant_id": tenant_id}
            )

            if not quota_result:
                return {
                    "tenant_id": tenant_id,
                    "status": "no_quota_set",
                    "usage": {},
                }

            queries_limit, tokens_limit, storage_limit, overage_allowed = quota_result[0]

            # Get current usage
            usage_result = self.client.execute(
                """
                SELECT
                    sumIf(quantity, event_type = 'query') as queries,
                    sumIf(quantity, event_type = 'token_input') + sumIf(quantity, event_type = 'token_output') as tokens
                FROM usage_events
                WHERE tenant_id = %(tenant_id)s
                  AND toDate(timestamp) = %(date)s
                """,
                {"tenant_id": tenant_id, "date": date.date()}
            )

            queries_used, tokens_used = usage_result[0] if usage_result else (0, 0)
            queries_used = queries_used or 0
            tokens_used = tokens_used or 0

            # Check for overages
            over_quota = (queries_used > queries_limit) or (tokens_used > tokens_limit)

            return {
                "tenant_id": tenant_id,
                "date": date.date().isoformat(),
                "status": "over_quota" if over_quota else "within_quota",
                "overage_allowed": bool(overage_allowed),
                "limits": {
                    "queries_per_day": queries_limit,
                    "tokens_per_day": tokens_limit,
                },
                "usage": {
                    "queries": int(queries_used),
                    "tokens": int(tokens_used),
                },
                "remaining": {
                    "queries": max(0, queries_limit - int(queries_used)),
                    "tokens": max(0, tokens_limit - int(tokens_used)),
                },
            }

        except Exception as e:
            logger.error(f"Quota check failed: {e}")
            return {"error": str(e)}


# ============================================================================
# Billing Export
# ============================================================================

class BillingExporter:
    """Exports usage data for invoice generation."""

    def __init__(self, client):
        self.client = client

    def export_monthly_invoice(self, tenant_id: str, year: int, month: int) -> Dict[str, Any]:
        """
        Generate monthly invoice data for a tenant.

        Args:
            tenant_id: Tenant identifier
            year: Invoice year
            month: Invoice month (1-12)

        Returns:
            Invoice data with usage breakdown and costs
        """
        if self.client is None:
            return {"error": "ClickHouse not available"}

        try:
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)

            # Get usage summary
            usage_result = self.client.execute(
                """
                SELECT
                    event_type,
                    sum(quantity) as total_quantity,
                    count() as event_count
                FROM usage_events
                WHERE tenant_id = %(tenant_id)s
                  AND timestamp >= %(start_date)s
                  AND timestamp < %(end_date)s
                GROUP BY event_type
                """,
                {
                    "tenant_id": tenant_id,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

            # Calculate costs
            pricing = Config.PRICING_CONFIG
            line_items = []
            total_cost = 0.0

            for event_type, quantity, count in usage_result:
                if event_type == "query":
                    unit_price = pricing["query"]
                    cost = quantity * unit_price
                    unit = "query"
                elif event_type == "token_input":
                    unit_price = pricing["token_input"]
                    cost = (quantity / 1000) * unit_price
                    unit = "1K tokens"
                    quantity = quantity / 1000
                elif event_type == "token_output":
                    unit_price = pricing["token_output"]
                    cost = (quantity / 1000) * unit_price
                    unit = "1K tokens"
                    quantity = quantity / 1000
                elif event_type == "storage":
                    unit_price = pricing["storage_gb"]
                    cost = quantity * unit_price
                    unit = "GB"
                else:
                    continue

                line_items.append({
                    "description": f"{event_type.replace('_', ' ').title()}",
                    "quantity": round(quantity, 2),
                    "unit": unit,
                    "unit_price": unit_price,
                    "amount": round(cost, 2),
                })
                total_cost += cost

            return {
                "tenant_id": tenant_id,
                "period": f"{year}-{month:02d}",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "line_items": line_items,
                "total_amount": round(total_cost, 2),
                "currency": "USD",
            }

        except Exception as e:
            logger.error(f"Invoice export failed: {e}")
            return {"error": str(e)}


# ============================================================================
# Decorator for Automatic Instrumentation
# ============================================================================

def track_usage(event_type: str, quantity_func: Optional[Callable] = None):
    """
    Decorator to automatically track usage for API endpoints.

    Args:
        event_type: Type of event to track
        quantity_func: Optional function to extract quantity from result

    Usage:
        @track_usage("query")
        async def query_endpoint(tenant_id: str, query: str):
            # Implementation
            return result
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract tenant_id (assume first arg or kwarg)
            tenant_id = kwargs.get("tenant_id") or (args[0] if args else "unknown")

            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Determine quantity
            if quantity_func:
                quantity = quantity_func(result)
            else:
                quantity = 1.0

            # Track event (non-blocking)
            event = UsageEvent(
                event_id=str(uuid.uuid4()),
                tenant_id=str(tenant_id),
                event_type=event_type,
                quantity=quantity,
                timestamp=datetime.now(),
                metadata={"latency_ms": round(elapsed * 1000, 2)}
            )

            # Get global tracker if available
            if hasattr(wrapper, "_tracker"):
                asyncio.create_task(wrapper._tracker.track(event))

            return result
        return wrapper
    return decorator


# ============================================================================
# CLI Examples
# ============================================================================

if __name__ == "__main__":
    import sys

    # Initialize client
    client = get_clickhouse_client()

    if len(sys.argv) < 2:
        print("Usage: python l2_m12_usage_metering_analytics.py <command>")
        print("\nCommands:")
        print("  init-schema          Initialize ClickHouse schema")
        print("  set-quota <tenant>   Set default quota for tenant")
        print("  check-quota <tenant> Check quota status for tenant")
        print("  invoice <tenant> <year> <month>  Generate monthly invoice")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init-schema":
        success = ClickHouseSchema.initialize_schema(client)
        print("✓ Schema initialized" if success else "✗ Schema initialization failed")

    elif command == "set-quota" and len(sys.argv) == 3:
        tenant_id = sys.argv[2]
        quota = TenantQuota(
            tenant_id=tenant_id,
            queries_per_day=Config.DEFAULT_QUOTA_QUERIES_PER_DAY,
            tokens_per_day=Config.DEFAULT_QUOTA_TOKENS_PER_DAY,
            storage_gb=10.0,
        )
        manager = QuotaManager(client)
        success = manager.set_quota(quota)
        print(f"✓ Quota set for {tenant_id}" if success else "✗ Failed to set quota")

    elif command == "check-quota" and len(sys.argv) == 3:
        tenant_id = sys.argv[2]
        manager = QuotaManager(client)
        status = manager.check_quota(tenant_id)
        print(json.dumps(status, indent=2))

    elif command == "invoice" and len(sys.argv) == 5:
        tenant_id = sys.argv[2]
        year = int(sys.argv[3])
        month = int(sys.argv[4])
        exporter = BillingExporter(client)
        invoice = exporter.export_monthly_invoice(tenant_id, year, month)
        print(json.dumps(invoice, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
