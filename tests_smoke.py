"""
Smoke tests for M12.3 Self-Service Tenant Onboarding.

Tests core functionality without requiring external services.
Network paths gracefully skip when API keys are missing.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l2_m12_self_service_tenant_onboarding import (
    create_skeleton_tenant,
    generate_tenant_id,
    hash_password,
    generate_stripe_checkout_url,
    handle_checkout_completed,
    provision_tenant,
    generate_api_key,
    track_activation_event,
    calculate_activation_metrics,
    check_provisioning_timeout,
    TenantStatus,
    PlanType,
)


class TestSkeletonTenant:
    """Test tenant creation."""

    def test_generate_tenant_id(self):
        """Test tenant ID generation."""
        tenant_id = generate_tenant_id()
        assert tenant_id is not None
        assert len(tenant_id) == 32  # 16 bytes hex = 32 chars
        assert isinstance(tenant_id, str)

    def test_hash_password(self):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = hash_password(password)
        assert hashed is not None
        assert len(hashed) == 64  # SHA-256 hex digest
        assert hashed != password

    def test_create_skeleton_tenant_success(self):
        """Test successful tenant creation."""
        tenant = create_skeleton_tenant(
            email="test@example.com",
            company_name="Test Corp",
            password="ValidPass123!",
            plan="starter"
        )

        assert tenant['email'] == "test@example.com"
        assert tenant['company_name'] == "Test Corp"
        assert tenant['plan'] == "starter"
        assert tenant['status'] == TenantStatus.PENDING_PAYMENT.value
        assert 'tenant_id' in tenant
        assert 'password_hash' in tenant
        assert tenant['password_hash'] != "ValidPass123!"

    def test_create_skeleton_tenant_invalid_plan(self):
        """Test tenant creation with invalid plan."""
        with pytest.raises(ValueError, match="Invalid plan"):
            create_skeleton_tenant(
                email="test@example.com",
                company_name="Test Corp",
                password="ValidPass123!",
                plan="invalid_plan"
            )

    def test_create_skeleton_tenant_short_password(self):
        """Test tenant creation with short password."""
        with pytest.raises(ValueError, match="Password must be at least 8 characters"):
            create_skeleton_tenant(
                email="test@example.com",
                company_name="Test Corp",
                password="short",
                plan="starter"
            )

    def test_create_skeleton_tenant_missing_fields(self):
        """Test tenant creation with missing fields."""
        with pytest.raises(ValueError, match="Missing required fields"):
            create_skeleton_tenant(
                email="",
                company_name="Test Corp",
                password="ValidPass123!",
                plan="starter"
            )


class TestStripeIntegration:
    """Test Stripe checkout integration."""

    def test_generate_checkout_url_no_client(self):
        """Test checkout URL generation without Stripe client."""
        url = generate_stripe_checkout_url("test_tenant_123", "pro")
        assert url is not None
        assert "test_tenant_123" in url
        assert "pro" in url

    def test_handle_checkout_completed(self):
        """Test webhook handler."""
        tenant_store = {
            "tenant_123": {
                "tenant_id": "tenant_123",
                "status": TenantStatus.PENDING_PAYMENT.value,
            }
        }

        webhook_data = {
            'object': {
                'metadata': {'tenant_id': 'tenant_123'},
                'customer': 'cus_test123',
                'subscription': 'sub_test456',
            }
        }

        updated = handle_checkout_completed(webhook_data, tenant_store)

        assert updated['status'] == TenantStatus.PROVISIONING.value
        assert updated['stripe_customer_id'] == 'cus_test123'
        assert updated['stripe_subscription_id'] == 'sub_test456'


class TestProvisioning:
    """Test tenant provisioning."""

    def test_generate_api_key(self):
        """Test API key generation."""
        api_key = generate_api_key("tenant_123")
        assert api_key is not None
        assert isinstance(api_key, str)
        assert len(api_key) > 20

    def test_provision_tenant_basic(self):
        """Test basic tenant provisioning without external services."""
        tenant_store = {
            "tenant_123": {
                "tenant_id": "tenant_123",
                "email": "test@example.com",
                "status": TenantStatus.PROVISIONING.value,
            }
        }

        sample_docs = [
            {"title": "Doc 1", "content": "Content 1"},
        ]

        result = provision_tenant(
            "tenant_123",
            tenant_store,
            pinecone_client=None,  # No client
            stripe_client=None,
            sample_docs=sample_docs
        )

        assert result['status'] == TenantStatus.ACTIVE.value
        assert 'pinecone_namespace' in result
        assert 'api_key' in result
        assert 'activated_at' in result

    def test_provision_tenant_not_found(self):
        """Test provisioning non-existent tenant."""
        tenant_store = {}

        with pytest.raises(ValueError, match="not found"):
            provision_tenant(
                "nonexistent",
                tenant_store,
            )


class TestActivationTracking:
    """Test activation monitoring."""

    def test_track_activation_event(self):
        """Test event tracking."""
        event = track_activation_event(
            "tenant_123",
            "signup_completed",
            {"source": "landing_page"}
        )

        assert event['tenant_id'] == "tenant_123"
        assert event['event_type'] == "signup_completed"
        assert event['metadata']['source'] == "landing_page"
        assert 'timestamp' in event

    def test_calculate_activation_metrics_empty(self):
        """Test metrics calculation with no events."""
        metrics = calculate_activation_metrics([])

        assert metrics['total_signups'] == 0
        assert metrics['activation_rate'] == 0.0

    def test_calculate_activation_metrics_full_funnel(self):
        """Test metrics calculation with full funnel."""
        events = [
            {'event_type': 'signup_completed'},
            {'event_type': 'signup_completed'},
            {'event_type': 'payment_confirmed'},
            {'event_type': 'payment_confirmed'},
            {'event_type': 'first_login'},
            {'event_type': 'first_query_executed'},
        ]

        metrics = calculate_activation_metrics(events)

        assert metrics['total_signups'] == 2
        assert metrics['payment_confirmed'] == 2
        assert metrics['first_login'] == 1
        assert metrics['first_query_executed'] == 1
        assert metrics['login_rate'] == 50.0
        assert metrics['activation_rate'] == 50.0

    def test_check_provisioning_timeout_active(self):
        """Test timeout check for active tenant."""
        from datetime import datetime

        tenant = {
            'tenant_id': 'test',
            'status': TenantStatus.ACTIVE.value,
            'updated_at': datetime.utcnow().isoformat(),
        }

        assert check_provisioning_timeout(tenant) is False

    def test_check_provisioning_timeout_recent(self):
        """Test timeout check for recent provisioning."""
        from datetime import datetime

        tenant = {
            'tenant_id': 'test',
            'status': TenantStatus.PROVISIONING.value,
            'updated_at': datetime.utcnow().isoformat(),
        }

        assert check_provisioning_timeout(tenant, timeout_seconds=300) is False


class TestConfigLoading:
    """Test configuration loading."""

    def test_config_imports(self):
        """Test that config module can be imported."""
        import config
        assert hasattr(config, 'DATABASE_URL')
        assert hasattr(config, 'get_clients')
        assert hasattr(config, 'verify_config')

    def test_config_verify(self):
        """Test config verification."""
        import config
        status = config.verify_config()
        assert isinstance(status, dict)
        assert 'stripe' in status
        assert 'pinecone' in status
        assert 'redis' in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
