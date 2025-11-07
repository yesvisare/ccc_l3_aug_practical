"""
Smoke tests for M12.2 Billing Integration.
Tests basic functionality without requiring live Stripe keys.
"""

import pytest
import os
from unittest.mock import Mock, patch
from l2_m12_billing_integration import (
    StripeBillingManager,
    UsageSyncService,
    DunningManager,
    verify_webhook_signature
)
from config import Config


def test_config_loads():
    """Test that configuration loads without errors"""
    assert Config.DEFAULT_TRIAL_DAYS > 0
    assert Config.DEFAULT_PLAN in ["starter", "pro", "enterprise"]
    assert Config.MAX_PAYMENT_FAILURES > 0


def test_billing_manager_initialization_no_key():
    """Test billing manager handles missing API key gracefully"""
    # Clear any existing key
    original_key = os.getenv("STRIPE_SECRET_KEY")
    os.environ["STRIPE_SECRET_KEY"] = ""

    billing = StripeBillingManager()
    assert billing.stripe_api_key == ""

    # Restore original key
    if original_key:
        os.environ["STRIPE_SECRET_KEY"] = original_key


def test_billing_manager_initialization_with_key():
    """Test billing manager initialization with API key"""
    test_key = "sk_test_12345"
    billing = StripeBillingManager(stripe_api_key=test_key)
    assert billing.stripe_api_key == test_key


def test_create_customer_without_key():
    """Test customer creation fails gracefully without API key"""
    billing = StripeBillingManager(stripe_api_key="")

    result = billing.create_customer(
        tenant_id="test_001",
        email="test@example.com",
        name="Test Company"
    )

    assert result is None


def test_usage_sync_service_initialization():
    """Test usage sync service initializes correctly"""
    billing = StripeBillingManager(stripe_api_key="")
    sync = UsageSyncService(billing_manager=billing)

    assert sync.billing is not None
    assert isinstance(sync.billing, StripeBillingManager)


def test_usage_sync_returns_results():
    """Test usage sync returns expected result structure"""
    billing = StripeBillingManager(stripe_api_key="")
    sync = UsageSyncService(billing_manager=billing)

    usage_data = [
        {
            "tenant_id": "tenant_001",
            "query_count": 5000,
            "subscription_id": "sub_test123"
        }
    ]

    results = sync.sync_daily_usage(usage_data)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["tenant_id"] == "tenant_001"
    assert "success" in results[0]


def test_dunning_manager_initialization():
    """Test dunning manager initializes"""
    dunning = DunningManager()
    assert dunning is not None


def test_dunning_first_failure():
    """Test dunning logic for first payment failure"""
    dunning = DunningManager()

    result = dunning.process_failed_payment(
        tenant_id="test_001",
        failure_count=1,
        invoice_amount=99.00
    )

    assert result["action"] == "reminder_sent"
    assert result["failure_count"] == 1


def test_dunning_second_failure():
    """Test dunning logic for second payment failure"""
    dunning = DunningManager()

    result = dunning.process_failed_payment(
        tenant_id="test_001",
        failure_count=2,
        invoice_amount=99.00
    )

    assert result["action"] == "warning_sent"
    assert result["failure_count"] == 2


def test_dunning_third_failure():
    """Test dunning logic for third payment failure"""
    dunning = DunningManager()

    result = dunning.process_failed_payment(
        tenant_id="test_001",
        failure_count=3,
        invoice_amount=99.00
    )

    assert result["action"] == "rate_limit_reduced"
    assert result["failure_count"] == 3


def test_dunning_fourth_failure():
    """Test dunning logic for fourth payment failure (suspension)"""
    dunning = DunningManager()

    result = dunning.process_failed_payment(
        tenant_id="test_001",
        failure_count=4,
        invoice_amount=99.00
    )

    assert result["action"] == "service_suspended"
    assert result["failure_count"] == 4


def test_dunning_reactivation():
    """Test tenant reactivation after payment success"""
    dunning = DunningManager()

    result = dunning.reactivate_tenant(tenant_id="test_001")

    assert result["action"] == "reactivated"
    assert result["tenant_id"] == "test_001"
    assert "timestamp" in result


def test_webhook_signature_verification_invalid_payload():
    """Test webhook signature verification with invalid payload"""
    result = verify_webhook_signature(
        payload=b"invalid",
        signature="sig_123",
        webhook_secret="whsec_test"
    )

    # Should return None for invalid payload
    assert result is None


def test_config_validation_without_keys():
    """Test config validation detects missing keys"""
    original_key = Config.STRIPE_SECRET_KEY
    Config.STRIPE_SECRET_KEY = ""

    is_valid, errors = Config.validate()

    assert not is_valid
    assert len(errors) > 0
    assert any("STRIPE_SECRET_KEY" in error for error in errors)

    # Restore
    Config.STRIPE_SECRET_KEY = original_key


def test_config_validation_with_invalid_key_format():
    """Test config validation detects invalid key format"""
    original_key = Config.STRIPE_SECRET_KEY
    Config.STRIPE_SECRET_KEY = "invalid_key"

    is_valid, errors = Config.validate()

    assert not is_valid
    assert any("sk_" in error for error in errors)

    # Restore
    Config.STRIPE_SECRET_KEY = original_key


def test_subscription_creation_without_key():
    """Test subscription creation fails gracefully without API key"""
    billing = StripeBillingManager(stripe_api_key="")

    result = billing.create_subscription(
        customer_id="cus_test123",
        plan_type="pro",
        tenant_id="test_001",
        trial_days=14
    )

    assert result is None


def test_subscription_creation_invalid_plan():
    """Test subscription creation with invalid plan type"""
    billing = StripeBillingManager(stripe_api_key="sk_test_fake")

    result = billing.create_subscription(
        customer_id="cus_test123",
        plan_type="invalid_plan",
        tenant_id="test_001",
        trial_days=14
    )

    assert result is None


def test_cancel_subscription_without_key():
    """Test subscription cancellation fails gracefully without API key"""
    billing = StripeBillingManager(stripe_api_key="")

    result = billing.cancel_subscription(
        subscription_id="sub_test123",
        cancel_at_period_end=True
    )

    assert result is None


def test_report_usage_without_key():
    """Test usage reporting fails gracefully without API key"""
    billing = StripeBillingManager(stripe_api_key="")

    result = billing.report_usage(
        subscription_id="sub_test123",
        quantity=5000
    )

    assert result is False


if __name__ == "__main__":
    print("Running smoke tests...")
    print("=" * 50)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
