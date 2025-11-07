"""
Module 12.4: Tenant Lifecycle Management - Smoke Tests
Minimal tests to verify core functionality without external dependencies.
"""

import pytest
import json
from pathlib import Path

# Import modules to test
import config
import l2_m12_tenant_lifecycle_management as lifecycle


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        """Verify config loads without errors."""
        assert config.Config.PLAN_HIERARCHY is not None
        assert len(config.Config.PLAN_HIERARCHY) > 0

    def test_plan_hierarchy(self):
        """Verify plan hierarchy is correctly ordered."""
        expected = ["free", "starter", "professional", "enterprise"]
        assert config.Config.PLAN_HIERARCHY == expected

    def test_plan_limits_exist(self):
        """Verify all plans have defined limits."""
        for plan in config.Config.PLAN_HIERARCHY:
            assert plan in config.Config.PLAN_LIMITS
            limits = config.Config.PLAN_LIMITS[plan]
            assert "users" in limits
            assert "storage_gb" in limits
            assert "api_calls_per_day" in limits

    def test_get_clients_graceful_without_services(self):
        """Verify get_clients returns dict even without services."""
        clients = config.get_clients()
        assert isinstance(clients, dict)
        assert "stripe" in clients
        assert "redis" in clients
        assert "database" in clients


class TestTenantStateMachine:
    """Test tenant state machine."""

    def test_state_machine_initialization(self):
        """Verify state machine initializes correctly."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="free",
            state=lifecycle.TenantState.ACTIVE
        )
        sm = lifecycle.TenantLifecycleStateMachine(tenant)
        assert sm.tenant.tenant_id == "test_001"
        assert sm.tenant.state == lifecycle.TenantState.ACTIVE

    def test_valid_transition(self):
        """Test valid state transition."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="free",
            state=lifecycle.TenantState.ACTIVE
        )
        sm = lifecycle.TenantLifecycleStateMachine(tenant)

        # Active -> Upgrading is valid
        assert sm.can_transition(lifecycle.TenantState.UPGRADING)
        success = sm.transition(lifecycle.TenantState.UPGRADING)
        assert success
        assert sm.tenant.state == lifecycle.TenantState.UPGRADING

    def test_invalid_transition(self):
        """Test invalid state transition is blocked."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="free",
            state=lifecycle.TenantState.DELETED
        )
        sm = lifecycle.TenantLifecycleStateMachine(tenant)

        # Deleted -> Active is invalid (must use reactivation flow)
        assert not sm.can_transition(lifecycle.TenantState.ACTIVE)
        success = sm.transition(lifecycle.TenantState.ACTIVE)
        assert not success
        assert sm.tenant.state == lifecycle.TenantState.DELETED

    def test_audit_log(self):
        """Test audit log tracks transitions."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="free",
            state=lifecycle.TenantState.ACTIVE
        )
        sm = lifecycle.TenantLifecycleStateMachine(tenant)

        sm.transition(lifecycle.TenantState.UPGRADING)
        audit_log = sm.get_audit_log()

        assert len(audit_log) == 1
        assert audit_log[0]["from_state"] == "active"
        assert audit_log[0]["to_state"] == "upgrading"


class TestPlanChangeManager:
    """Test plan change operations."""

    def test_plan_manager_initialization(self):
        """Verify plan manager initializes correctly."""
        manager = lifecycle.PlanChangeManager(
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS
        )
        assert manager.plan_hierarchy == config.Config.PLAN_HIERARCHY
        assert manager.plan_limits == config.Config.PLAN_LIMITS

    def test_get_change_type_upgrade(self):
        """Test upgrade detection."""
        manager = lifecycle.PlanChangeManager(
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS
        )
        change_type = manager.get_change_type("free", "starter")
        assert change_type == lifecycle.PlanChangeType.UPGRADE

    def test_get_change_type_downgrade(self):
        """Test downgrade detection."""
        manager = lifecycle.PlanChangeManager(
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS
        )
        change_type = manager.get_change_type("professional", "starter")
        assert change_type == lifecycle.PlanChangeType.DOWNGRADE

    def test_validate_downgrade_success(self):
        """Test downgrade validation with valid usage."""
        manager = lifecycle.PlanChangeManager(
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS
        )
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="starter",
            state=lifecycle.TenantState.ACTIVE,
            current_usage={"users": 3, "storage_gb": 0.5, "api_calls_per_day": 50}
        )

        is_valid, error = manager.validate_downgrade(tenant, "free")
        assert is_valid
        assert error is None

    def test_validate_downgrade_failure(self):
        """Test downgrade validation with excessive usage."""
        manager = lifecycle.PlanChangeManager(
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS
        )
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="professional",
            state=lifecycle.TenantState.ACTIVE,
            current_usage={"users": 50, "storage_gb": 40, "api_calls_per_day": 5000}
        )

        is_valid, error = manager.validate_downgrade(tenant, "starter")
        assert not is_valid
        assert error is not None
        assert "users" in error

    def test_upgrade_without_stripe(self):
        """Test upgrade works without Stripe client."""
        tenant_data = {
            "tenant_id": "test_001",
            "name": "Test Corp",
            "email": "test@example.com",
            "current_plan": "free",
            "state": "active",
            "current_usage": {"users": 3, "storage_gb": 0.5, "api_calls_per_day": 50}
        }

        result = lifecycle.upgrade_tenant(
            tenant_data,
            "starter",
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS,
            stripe_client=None
        )

        assert result["success"]
        assert result["new_plan"] == "starter"


class TestDataExportService:
    """Test data export operations."""

    def test_export_service_initialization(self):
        """Verify export service initializes correctly."""
        service = lifecycle.DataExportService(chunk_size_mb=50)
        assert service.chunk_size_mb == 50

    def test_initiate_export(self):
        """Test export initiation."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="starter",
            state=lifecycle.TenantState.ACTIVE
        )
        service = lifecycle.DataExportService()
        result = service.initiate_export(tenant, export_type="full")

        assert "export_id" in result
        assert result["tenant_id"] == "test_001"
        assert result["status"] == "queued"
        assert "estimated_completion" in result

    def test_process_export(self):
        """Test export processing."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="starter",
            state=lifecycle.TenantState.ACTIVE
        )
        service = lifecycle.DataExportService()
        result = service.process_export(tenant, "export_test_001")

        assert result["status"] == "completed"
        assert "download_url" in result
        assert "checksum" in result
        assert "file_size_mb" in result


class TestDeletionManager:
    """Test deletion operations."""

    def test_deletion_manager_initialization(self):
        """Verify deletion manager initializes correctly."""
        manager = lifecycle.DeletionManager(retention_days=30)
        assert manager.retention_days == 30

    def test_soft_delete(self):
        """Test soft deletion."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="starter",
            state=lifecycle.TenantState.ACTIVE
        )
        manager = lifecycle.DeletionManager(retention_days=30)
        result = manager.soft_delete(tenant, "admin@test.com")

        assert "deletion_id" in result
        assert result["status"] == "soft_deleted"
        assert result["retention_days"] == 30
        assert result["can_reactivate"]
        assert tenant.state == lifecycle.TenantState.DELETED

    def test_hard_delete(self):
        """Test hard deletion."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="starter",
            state=lifecycle.TenantState.DELETED
        )
        manager = lifecycle.DeletionManager()
        result = manager.hard_delete(tenant)

        assert result["all_verified"]
        assert "verification" in result
        assert len(result["verification"]) > 0


class TestReactivationWorkflow:
    """Test reactivation operations."""

    def test_reactivation_workflow_initialization(self):
        """Verify reactivation workflow initializes correctly."""
        workflow = lifecycle.ReactivationWorkflow()
        assert workflow is not None

    def test_can_reactivate_suspended(self):
        """Test reactivation check for suspended tenant."""
        tenant = lifecycle.TenantMetadata(
            tenant_id="test_001",
            name="Test Corp",
            email="test@example.com",
            current_plan="starter",
            state=lifecycle.TenantState.SUSPENDED
        )
        workflow = lifecycle.ReactivationWorkflow()
        can_reactivate, reason = workflow.can_reactivate(tenant)

        assert can_reactivate
        assert reason is None

    def test_reactivate_suspended_tenant(self):
        """Test reactivation of suspended tenant."""
        tenant_data = {
            "tenant_id": "test_001",
            "name": "Test Corp",
            "email": "test@example.com",
            "current_plan": "starter",
            "state": "suspended"
        }

        result = lifecycle.reactivate_tenant(
            tenant_data,
            reactivation_plan=None,
            stripe_client=None
        )

        assert result["success"]
        assert result["tenant_id"] == "test_001"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_upgrade_tenant_function(self):
        """Test upgrade_tenant convenience function."""
        tenant_data = {
            "tenant_id": "test_001",
            "name": "Test Corp",
            "email": "test@example.com",
            "current_plan": "free",
            "state": "active"
        }

        result = lifecycle.upgrade_tenant(
            tenant_data,
            "starter",
            config.Config.PLAN_HIERARCHY,
            config.Config.PLAN_LIMITS
        )

        assert "success" in result
        assert result["success"]

    def test_export_tenant_data_function(self):
        """Test export_tenant_data convenience function."""
        tenant_data = {
            "tenant_id": "test_001",
            "name": "Test Corp",
            "email": "test@example.com",
            "current_plan": "starter",
            "state": "active"
        }

        result = lifecycle.export_tenant_data(tenant_data)

        assert "export_id" in result
        assert result["status"] == "queued"

    def test_delete_tenant_function(self):
        """Test delete_tenant convenience function."""
        tenant_data = {
            "tenant_id": "test_001",
            "name": "Test Corp",
            "email": "test@example.com",
            "current_plan": "starter",
            "state": "active"
        }

        result = lifecycle.delete_tenant(tenant_data, "admin@test.com")

        assert "deletion_id" in result
        assert result["status"] == "soft_deleted"


class TestExampleData:
    """Test example data file."""

    def test_example_data_loads(self):
        """Verify example_data.json loads correctly."""
        example_file = Path("example_data.json")
        assert example_file.exists()

        with open(example_file) as f:
            data = json.load(f)

        assert "tenants" in data
        assert "lifecycle_events" in data
        assert "export_requests" in data
        assert "deletion_requests" in data

    def test_example_tenants_valid(self):
        """Verify example tenants have required fields."""
        with open("example_data.json") as f:
            data = json.load(f)

        for tenant in data["tenants"]:
            assert "tenant_id" in tenant
            assert "name" in tenant
            assert "email" in tenant
            assert "current_plan" in tenant
            assert "state" in tenant


if __name__ == "__main__":
    """Run smoke tests."""
    pytest.main([__file__, "-v", "--tb=short"])
