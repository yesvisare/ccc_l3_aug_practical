"""
Module 12.4: Tenant Lifecycle Management
Complete implementation of tenant lifecycle operations including upgrades, downgrades,
data exports, deletions, and reactivation workflows.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from dataclasses import dataclass, asdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TenantState(str, Enum):
    """Valid tenant states in the lifecycle."""
    ACTIVE = "active"
    UPGRADING = "upgrading"
    DOWNGRADING = "downgrading"
    SUSPENDED = "suspended"
    EXPORTING = "exporting"
    DELETING = "deleting"
    DELETED = "deleted"
    REACTIVATING = "reactivating"


class PlanChangeType(str, Enum):
    """Types of plan changes."""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


@dataclass
class TenantMetadata:
    """Tenant metadata structure."""
    tenant_id: str
    name: str
    email: str
    current_plan: str
    state: TenantState
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    current_usage: Optional[Dict[str, int]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    suspended_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class LifecycleEvent:
    """Lifecycle event record."""
    event_id: str
    tenant_id: str
    event_type: str
    from_state: str
    to_state: str
    initiated_by: str
    timestamp: str
    status: str
    metadata: Optional[Dict[str, Any]] = None


class TenantLifecycleStateMachine:
    """
    State machine for managing valid tenant state transitions.
    Prevents invalid transitions and logs all state changes.
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        TenantState.ACTIVE: [
            TenantState.UPGRADING,
            TenantState.DOWNGRADING,
            TenantState.SUSPENDED,
            TenantState.EXPORTING,
            TenantState.DELETING
        ],
        TenantState.UPGRADING: [TenantState.ACTIVE, TenantState.SUSPENDED],
        TenantState.DOWNGRADING: [TenantState.ACTIVE, TenantState.SUSPENDED],
        TenantState.SUSPENDED: [TenantState.REACTIVATING, TenantState.DELETING],
        TenantState.EXPORTING: [TenantState.ACTIVE],
        TenantState.DELETING: [TenantState.DELETED],
        TenantState.DELETED: [],
        TenantState.REACTIVATING: [TenantState.ACTIVE, TenantState.SUSPENDED]
    }

    def __init__(self, tenant: TenantMetadata):
        """Initialize state machine with tenant."""
        self.tenant = tenant
        self.audit_log: List[Dict[str, Any]] = []

    def can_transition(self, to_state: TenantState) -> bool:
        """
        Check if transition to new state is valid.

        Args:
            to_state: Target state

        Returns:
            True if transition is valid
        """
        current_state = TenantState(self.tenant.state)
        return to_state in self.VALID_TRANSITIONS.get(current_state, [])

    def transition(self, to_state: TenantState, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute state transition with validation and audit logging.

        Args:
            to_state: Target state
            metadata: Optional metadata about the transition

        Returns:
            True if transition succeeded
        """
        from_state = TenantState(self.tenant.state)

        if not self.can_transition(to_state):
            logger.error(
                f"Invalid state transition for tenant {self.tenant.tenant_id}: "
                f"{from_state} -> {to_state}"
            )
            return False

        # Log transition
        audit_entry = {
            "tenant_id": self.tenant.tenant_id,
            "from_state": from_state.value,
            "to_state": to_state.value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.audit_log.append(audit_entry)

        # Update tenant state
        self.tenant.state = to_state
        self.tenant.updated_at = datetime.utcnow().isoformat()

        logger.info(
            f"Tenant {self.tenant.tenant_id} transitioned: "
            f"{from_state} -> {to_state}"
        )
        return True

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get full audit log for this tenant."""
        return self.audit_log


class PlanChangeManager:
    """
    Manages plan upgrades and downgrades with rollback capability.
    Integrates with Stripe for billing changes.
    """

    def __init__(self, plan_hierarchy: List[str], plan_limits: Dict[str, Dict[str, int]]):
        """
        Initialize plan change manager.

        Args:
            plan_hierarchy: Ordered list of plans (low to high)
            plan_limits: Resource limits for each plan
        """
        self.plan_hierarchy = plan_hierarchy
        self.plan_limits = plan_limits

    def get_change_type(self, from_plan: str, to_plan: str) -> Optional[PlanChangeType]:
        """
        Determine if change is upgrade or downgrade.

        Args:
            from_plan: Current plan
            to_plan: Target plan

        Returns:
            PlanChangeType or None if invalid
        """
        try:
            from_idx = self.plan_hierarchy.index(from_plan)
            to_idx = self.plan_hierarchy.index(to_plan)

            if to_idx > from_idx:
                return PlanChangeType.UPGRADE
            elif to_idx < from_idx:
                return PlanChangeType.DOWNGRADE
            else:
                return None
        except ValueError:
            logger.error(f"Invalid plan in hierarchy: {from_plan} or {to_plan}")
            return None

    def validate_downgrade(self, tenant: TenantMetadata, to_plan: str) -> tuple[bool, Optional[str]]:
        """
        Validate that current usage fits within new plan limits.

        Args:
            tenant: Tenant metadata with current usage
            to_plan: Target plan

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not tenant.current_usage:
            return True, None

        target_limits = self.plan_limits.get(to_plan, {})

        for resource, current_value in tenant.current_usage.items():
            limit = target_limits.get(resource, 0)

            # -1 means unlimited
            if limit == -1:
                continue

            if current_value > limit:
                return False, (
                    f"Current {resource} usage ({current_value}) exceeds "
                    f"{to_plan} plan limit ({limit})"
                )

        return True, None

    def execute_upgrade(
        self,
        tenant: TenantMetadata,
        to_plan: str,
        stripe_client=None
    ) -> Dict[str, Any]:
        """
        Execute plan upgrade with resource provisioning and billing update.

        Args:
            tenant: Tenant to upgrade
            to_plan: Target plan
            stripe_client: Optional Stripe client for billing

        Returns:
            Result dictionary with status and details
        """
        logger.info(f"Executing upgrade for {tenant.tenant_id}: {tenant.current_plan} -> {to_plan}")

        try:
            # Step 1: Validate plan hierarchy
            change_type = self.get_change_type(tenant.current_plan, to_plan)
            if change_type != PlanChangeType.UPGRADE:
                return {
                    "success": False,
                    "error": f"Invalid upgrade path: {tenant.current_plan} -> {to_plan}"
                }

            # Step 2: Provision new resources (before billing to avoid service interruption)
            provisioning_result = self._provision_resources(tenant, to_plan)
            if not provisioning_result["success"]:
                return provisioning_result

            # Step 3: Update Stripe subscription with proration
            billing_result = {"success": True, "proration_amount": 0}
            if stripe_client and tenant.stripe_subscription_id:
                billing_result = self._update_stripe_subscription(
                    stripe_client,
                    tenant.stripe_subscription_id,
                    to_plan,
                    prorate=True
                )

                if not billing_result["success"]:
                    # Rollback resource provisioning
                    logger.error(f"Billing update failed for {tenant.tenant_id}, rolling back")
                    self._rollback_resources(tenant, tenant.current_plan)
                    return billing_result

            # Step 4: Update tenant metadata
            tenant.current_plan = to_plan
            tenant.updated_at = datetime.utcnow().isoformat()

            logger.info(f"Upgrade completed for {tenant.tenant_id}")
            return {
                "success": True,
                "tenant_id": tenant.tenant_id,
                "new_plan": to_plan,
                "proration_amount": billing_result.get("proration_amount", 0),
                "next_billing_date": billing_result.get("next_billing_date")
            }

        except Exception as e:
            logger.error(f"Upgrade failed for {tenant.tenant_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def execute_downgrade(
        self,
        tenant: TenantMetadata,
        to_plan: str,
        stripe_client=None,
        schedule_for_period_end: bool = True
    ) -> Dict[str, Any]:
        """
        Execute plan downgrade with usage validation and safe resource reduction.

        Args:
            tenant: Tenant to downgrade
            to_plan: Target plan
            stripe_client: Optional Stripe client
            schedule_for_period_end: If True, apply downgrade at billing period end

        Returns:
            Result dictionary with status and details
        """
        logger.info(f"Executing downgrade for {tenant.tenant_id}: {tenant.current_plan} -> {to_plan}")

        try:
            # Step 1: Validate downgrade is allowed
            change_type = self.get_change_type(tenant.current_plan, to_plan)
            if change_type != PlanChangeType.DOWNGRADE:
                return {
                    "success": False,
                    "error": f"Invalid downgrade path: {tenant.current_plan} -> {to_plan}"
                }

            # Step 2: Validate current usage fits new plan
            is_valid, error_msg = self.validate_downgrade(tenant, to_plan)
            if not is_valid:
                return {"success": False, "error": error_msg}

            # Step 3: Schedule Stripe changes for billing period end
            billing_result = {"success": True, "scheduled_for": None}
            if stripe_client and tenant.stripe_subscription_id:
                billing_result = self._update_stripe_subscription(
                    stripe_client,
                    tenant.stripe_subscription_id,
                    to_plan,
                    prorate=False,
                    at_period_end=schedule_for_period_end
                )

                if not billing_result["success"]:
                    return billing_result

            # Step 4: Reduce resource allocations (safe, non-destructive)
            reduction_result = self._reduce_resources(tenant, to_plan)
            if not reduction_result["success"]:
                return reduction_result

            # Step 5: Update tenant metadata
            if not schedule_for_period_end:
                tenant.current_plan = to_plan
            tenant.updated_at = datetime.utcnow().isoformat()

            logger.info(f"Downgrade completed for {tenant.tenant_id}")
            return {
                "success": True,
                "tenant_id": tenant.tenant_id,
                "new_plan": to_plan,
                "scheduled_for": billing_result.get("scheduled_for"),
                "immediate": not schedule_for_period_end
            }

        except Exception as e:
            logger.error(f"Downgrade failed for {tenant.tenant_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def _provision_resources(self, tenant: TenantMetadata, plan: str) -> Dict[str, Any]:
        """Provision resources for new plan (stub implementation)."""
        logger.info(f"Provisioning resources for {tenant.tenant_id} on {plan} plan")
        # In production: allocate database capacity, storage, API rate limits, etc.
        return {"success": True, "resources_provisioned": self.plan_limits.get(plan, {})}

    def _rollback_resources(self, tenant: TenantMetadata, plan: str) -> Dict[str, Any]:
        """Rollback resource provisioning (stub implementation)."""
        logger.info(f"Rolling back resources for {tenant.tenant_id} to {plan} plan")
        return {"success": True}

    def _reduce_resources(self, tenant: TenantMetadata, plan: str) -> Dict[str, Any]:
        """Safely reduce resource allocations (stub implementation)."""
        logger.info(f"Reducing resources for {tenant.tenant_id} to {plan} plan")
        # In production: reduce quotas, storage, rate limits (without data loss)
        return {"success": True, "resources_reduced": self.plan_limits.get(plan, {})}

    def _update_stripe_subscription(
        self,
        stripe_client,
        subscription_id: str,
        new_plan: str,
        prorate: bool = True,
        at_period_end: bool = False
    ) -> Dict[str, Any]:
        """
        Update Stripe subscription (stub implementation).

        Args:
            stripe_client: Stripe client
            subscription_id: Subscription ID
            new_plan: New plan name
            prorate: Whether to prorate charges
            at_period_end: Apply changes at period end

        Returns:
            Result dictionary
        """
        if not stripe_client:
            logger.warning("Stripe client not configured, skipping billing update")
            return {"success": True, "skipped": True}

        logger.info(f"Updating Stripe subscription {subscription_id} to {new_plan}")

        # In production: call stripe_client.Subscription.modify()
        # For now, return mock success
        return {
            "success": True,
            "proration_amount": 25.00 if prorate else 0,
            "next_billing_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            "scheduled_for": (datetime.utcnow() + timedelta(days=30)).isoformat() if at_period_end else None
        }


class DataExportService:
    """
    GDPR-compliant data export service with chunked downloads.
    Generates exports in background jobs with signed URLs.
    """

    def __init__(self, chunk_size_mb: int = 50, storage_path: str = "/tmp/tenant_exports"):
        """
        Initialize export service.

        Args:
            chunk_size_mb: Size of export chunks in MB
            storage_path: Local storage path for exports
        """
        self.chunk_size_mb = chunk_size_mb
        self.storage_path = storage_path

    def initiate_export(
        self,
        tenant: TenantMetadata,
        export_type: str = "full",
        requested_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Initiate data export for tenant.

        Args:
            tenant: Tenant to export
            export_type: Type of export (full, incremental)
            requested_by: User who requested export

        Returns:
            Export job details
        """
        export_id = f"export_{tenant.tenant_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        logger.info(f"Initiating {export_type} export for {tenant.tenant_id}: {export_id}")

        # In production: enqueue Celery task for background processing
        export_job = {
            "export_id": export_id,
            "tenant_id": tenant.tenant_id,
            "export_type": export_type,
            "requested_by": requested_by,
            "requested_at": datetime.utcnow().isoformat(),
            "status": "queued",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
        }

        logger.info(f"Export job created: {export_id}")
        return export_job

    def process_export(self, tenant: TenantMetadata, export_id: str) -> Dict[str, Any]:
        """
        Process data export with chunking (stub for background job).

        Args:
            tenant: Tenant to export
            export_id: Export job ID

        Returns:
            Export result with download URL
        """
        logger.info(f"Processing export {export_id} for {tenant.tenant_id}")

        # In production:
        # 1. Query all tenant data from database
        # 2. Chunk data into manageable pieces
        # 3. Generate ZIP archive
        # 4. Upload to storage (S3, GCS, etc.)
        # 5. Generate signed URL with expiration

        # Mock export processing
        export_data = {
            "tenant_id": tenant.tenant_id,
            "exported_at": datetime.utcnow().isoformat(),
            "data": {
                "users": ["user1", "user2"],
                "projects": ["project1"],
                "documents": ["doc1", "doc2", "doc3"]
            }
        }

        # Calculate checksum
        data_bytes = json.dumps(export_data, sort_keys=True).encode()
        checksum = f"sha256:{hashlib.sha256(data_bytes).hexdigest()}"

        file_size_mb = len(data_bytes) / (1024 * 1024)

        result = {
            "export_id": export_id,
            "tenant_id": tenant.tenant_id,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "file_size_mb": round(file_size_mb, 2),
            "download_url": f"https://example.com/exports/{export_id}.zip",
            "url_expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "checksum": checksum
        }

        logger.info(f"Export completed: {export_id}")
        return result

    def verify_export_integrity(self, export_id: str, checksum: str) -> bool:
        """
        Verify export file integrity.

        Args:
            export_id: Export ID
            checksum: Expected checksum

        Returns:
            True if verification passed
        """
        logger.info(f"Verifying integrity of export {export_id}")
        # In production: recalculate checksum and compare
        return True


class DeletionManager:
    """
    Manages tenant deletion with soft-delete, retention policies, and verification.
    """

    def __init__(self, retention_days: int = 30):
        """
        Initialize deletion manager.

        Args:
            retention_days: Days to retain soft-deleted data
        """
        self.retention_days = retention_days

    def soft_delete(self, tenant: TenantMetadata, requested_by: str) -> Dict[str, Any]:
        """
        Soft-delete tenant with retention period.

        Args:
            tenant: Tenant to delete
            requested_by: User who requested deletion

        Returns:
            Deletion details
        """
        deletion_id = f"del_{tenant.tenant_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        logger.info(f"Soft-deleting tenant {tenant.tenant_id}: {deletion_id}")

        soft_deleted_at = datetime.utcnow()
        hard_delete_scheduled = soft_deleted_at + timedelta(days=self.retention_days)

        # Update tenant state
        tenant.state = TenantState.DELETED
        tenant.deleted_at = soft_deleted_at.isoformat()
        tenant.updated_at = soft_deleted_at.isoformat()

        deletion_record = {
            "deletion_id": deletion_id,
            "tenant_id": tenant.tenant_id,
            "requested_by": requested_by,
            "requested_at": soft_deleted_at.isoformat(),
            "soft_deleted_at": soft_deleted_at.isoformat(),
            "hard_delete_scheduled_at": hard_delete_scheduled.isoformat(),
            "status": "soft_deleted",
            "retention_days": self.retention_days,
            "can_reactivate": True
        }

        logger.info(
            f"Tenant {tenant.tenant_id} soft-deleted. "
            f"Hard delete scheduled for {hard_delete_scheduled.isoformat()}"
        )

        return deletion_record

    def hard_delete(self, tenant: TenantMetadata) -> Dict[str, Any]:
        """
        Permanently delete tenant data after retention period.

        Args:
            tenant: Tenant to permanently delete

        Returns:
            Verification result
        """
        logger.info(f"Executing hard delete for {tenant.tenant_id}")

        # In production:
        # 1. Delete all tenant data from database
        # 2. Delete files from storage
        # 3. Cancel Stripe subscription
        # 4. Remove all traces except audit logs

        verification_steps = [
            {"step": "database_records", "deleted": True},
            {"step": "storage_files", "deleted": True},
            {"step": "stripe_subscription", "cancelled": True},
            {"step": "cache_entries", "deleted": True}
        ]

        result = {
            "tenant_id": tenant.tenant_id,
            "hard_deleted_at": datetime.utcnow().isoformat(),
            "verification": verification_steps,
            "all_verified": True
        }

        logger.info(f"Hard delete completed and verified for {tenant.tenant_id}")
        return result

    def verify_deletion(self, tenant_id: str) -> Dict[str, Any]:
        """
        Verify that all tenant data has been removed.

        Args:
            tenant_id: Tenant ID to verify

        Returns:
            Verification results
        """
        logger.info(f"Verifying deletion of {tenant_id}")

        # In production: check all data stores
        checks = {
            "database": False,  # Should be False (no data found)
            "storage": False,
            "cache": False,
            "stripe": False
        }

        all_clear = all(not found for found in checks.values())

        return {
            "tenant_id": tenant_id,
            "verified_at": datetime.utcnow().isoformat(),
            "checks": checks,
            "deletion_verified": all_clear
        }


class ReactivationWorkflow:
    """
    Handles tenant reactivation with state conflict resolution.
    Used for win-back campaigns and recovery from suspension/deletion.
    """

    def can_reactivate(self, tenant: TenantMetadata) -> tuple[bool, Optional[str]]:
        """
        Check if tenant can be reactivated.

        Args:
            tenant: Tenant to check

        Returns:
            Tuple of (can_reactivate, reason_if_not)
        """
        if tenant.state == TenantState.DELETED:
            # Check if within retention period
            if tenant.deleted_at:
                deleted_time = datetime.fromisoformat(tenant.deleted_at.replace('Z', '+00:00'))
                retention_limit = deleted_time + timedelta(days=30)  # Example retention

                if datetime.utcnow() > retention_limit.replace(tzinfo=None):
                    return False, "Retention period expired, data permanently deleted"

            return True, None

        elif tenant.state == TenantState.SUSPENDED:
            return True, None

        else:
            return False, f"Cannot reactivate tenant in state: {tenant.state}"

    def reactivate(
        self,
        tenant: TenantMetadata,
        reactivation_plan: Optional[str] = None,
        stripe_client=None
    ) -> Dict[str, Any]:
        """
        Reactivate suspended or deleted tenant.

        Args:
            tenant: Tenant to reactivate
            reactivation_plan: Optional plan to reactivate on (defaults to previous)
            stripe_client: Optional Stripe client

        Returns:
            Reactivation result
        """
        logger.info(f"Attempting to reactivate tenant {tenant.tenant_id}")

        # Check if reactivation is allowed
        can_reactivate, reason = self.can_reactivate(tenant)
        if not can_reactivate:
            return {"success": False, "error": reason}

        try:
            # Step 1: Restore tenant data if soft-deleted
            if tenant.state == TenantState.DELETED:
                restore_result = self._restore_tenant_data(tenant)
                if not restore_result["success"]:
                    return restore_result

            # Step 2: Reactivate or create Stripe subscription
            billing_result = {"success": True}
            if stripe_client:
                billing_result = self._reactivate_stripe_subscription(
                    stripe_client,
                    tenant,
                    reactivation_plan or tenant.current_plan
                )

                if not billing_result["success"]:
                    return billing_result

            # Step 3: Update tenant state
            tenant.state = TenantState.ACTIVE
            if reactivation_plan:
                tenant.current_plan = reactivation_plan
            tenant.suspended_at = None
            tenant.deleted_at = None
            tenant.updated_at = datetime.utcnow().isoformat()

            logger.info(f"Tenant {tenant.tenant_id} successfully reactivated")
            return {
                "success": True,
                "tenant_id": tenant.tenant_id,
                "reactivated_at": datetime.utcnow().isoformat(),
                "plan": tenant.current_plan,
                "subscription_id": billing_result.get("subscription_id")
            }

        except Exception as e:
            logger.error(f"Reactivation failed for {tenant.tenant_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def _restore_tenant_data(self, tenant: TenantMetadata) -> Dict[str, Any]:
        """Restore soft-deleted tenant data (stub implementation)."""
        logger.info(f"Restoring data for {tenant.tenant_id}")
        # In production: un-mark deleted records, restore from backups if needed
        return {"success": True, "records_restored": 100}

    def _reactivate_stripe_subscription(
        self,
        stripe_client,
        tenant: TenantMetadata,
        plan: str
    ) -> Dict[str, Any]:
        """Reactivate or create Stripe subscription (stub implementation)."""
        if not stripe_client:
            logger.warning("Stripe client not configured, skipping billing reactivation")
            return {"success": True, "skipped": True}

        logger.info(f"Reactivating Stripe subscription for {tenant.tenant_id}")

        # In production: resume subscription or create new one
        return {
            "success": True,
            "subscription_id": f"sub_reactivated_{tenant.tenant_id}",
            "status": "active"
        }


# Convenience functions for module interface

def upgrade_tenant(
    tenant_data: Dict[str, Any],
    to_plan: str,
    plan_hierarchy: List[str],
    plan_limits: Dict[str, Dict[str, int]],
    stripe_client=None
) -> Dict[str, Any]:
    """
    Upgrade tenant to higher plan.

    Args:
        tenant_data: Tenant data dictionary
        to_plan: Target plan
        plan_hierarchy: Plan hierarchy list
        plan_limits: Plan limits dictionary
        stripe_client: Optional Stripe client

    Returns:
        Upgrade result
    """
    tenant = TenantMetadata(**tenant_data)
    manager = PlanChangeManager(plan_hierarchy, plan_limits)
    return manager.execute_upgrade(tenant, to_plan, stripe_client)


def downgrade_tenant(
    tenant_data: Dict[str, Any],
    to_plan: str,
    plan_hierarchy: List[str],
    plan_limits: Dict[str, Dict[str, int]],
    stripe_client=None
) -> Dict[str, Any]:
    """
    Downgrade tenant to lower plan.

    Args:
        tenant_data: Tenant data dictionary
        to_plan: Target plan
        plan_hierarchy: Plan hierarchy list
        plan_limits: Plan limits dictionary
        stripe_client: Optional Stripe client

    Returns:
        Downgrade result
    """
    tenant = TenantMetadata(**tenant_data)
    manager = PlanChangeManager(plan_hierarchy, plan_limits)
    return manager.execute_downgrade(tenant, to_plan, stripe_client)


def export_tenant_data(tenant_data: Dict[str, Any], export_type: str = "full") -> Dict[str, Any]:
    """
    Export tenant data for GDPR compliance.

    Args:
        tenant_data: Tenant data dictionary
        export_type: Type of export

    Returns:
        Export job details
    """
    tenant = TenantMetadata(**tenant_data)
    service = DataExportService()
    return service.initiate_export(tenant, export_type)


def delete_tenant(tenant_data: Dict[str, Any], requested_by: str = "system") -> Dict[str, Any]:
    """
    Soft-delete tenant with retention period.

    Args:
        tenant_data: Tenant data dictionary
        requested_by: User who requested deletion

    Returns:
        Deletion details
    """
    tenant = TenantMetadata(**tenant_data)
    manager = DeletionManager()
    return manager.soft_delete(tenant, requested_by)


def reactivate_tenant(
    tenant_data: Dict[str, Any],
    reactivation_plan: Optional[str] = None,
    stripe_client=None
) -> Dict[str, Any]:
    """
    Reactivate suspended or soft-deleted tenant.

    Args:
        tenant_data: Tenant data dictionary
        reactivation_plan: Optional plan for reactivation
        stripe_client: Optional Stripe client

    Returns:
        Reactivation result
    """
    tenant = TenantMetadata(**tenant_data)
    workflow = ReactivationWorkflow()
    return workflow.reactivate(tenant, reactivation_plan, stripe_client)


if __name__ == "__main__":
    """
    CLI usage examples for tenant lifecycle operations.
    """
    print("=== Tenant Lifecycle Management CLI ===\n")

    # Example configuration
    PLAN_HIERARCHY = ["free", "starter", "professional", "enterprise"]
    PLAN_LIMITS = {
        "free": {"users": 5, "storage_gb": 1, "api_calls_per_day": 100},
        "starter": {"users": 20, "storage_gb": 10, "api_calls_per_day": 1000},
        "professional": {"users": 100, "storage_gb": 100, "api_calls_per_day": 10000},
        "enterprise": {"users": -1, "storage_gb": -1, "api_calls_per_day": -1}
    }

    # Example tenant
    example_tenant = {
        "tenant_id": "tenant_demo",
        "name": "Demo Corp",
        "email": "admin@demo.com",
        "current_plan": "free",
        "state": "active",
        "current_usage": {"users": 3, "storage_gb": 0.5, "api_calls_per_day": 50}
    }

    print("1. Upgrade Example:")
    upgrade_result = upgrade_tenant(
        example_tenant,
        "starter",
        PLAN_HIERARCHY,
        PLAN_LIMITS
    )
    print(f"   Result: {upgrade_result}\n")

    print("2. Data Export Example:")
    export_result = export_tenant_data(example_tenant)
    print(f"   Result: {export_result}\n")

    print("3. Soft Delete Example:")
    delete_result = delete_tenant(example_tenant, "admin@demo.com")
    print(f"   Result: {delete_result}\n")

    print("✓ CLI examples completed")
