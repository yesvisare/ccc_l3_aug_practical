"""
Smoke tests for Module 13.3: Launch Preparation & Marketing

Minimal tests to verify:
- Config loads properly
- Core functions return expected shapes
- Calculations produce reasonable results
- Network paths gracefully skip without keys
"""

import pytest
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.l3_m13_launch_prep_marketing import (
    ValueProposition,
    PricingTierConfig,
    PricingCalculator,
    ConversionFunnelAnalyzer,
    CAC_LTV_Calculator,
    GTMStrategySelector,
    UTMTracker,
    GTMMotion,
    validate_launch_readiness
)
from config import Config, get_clients, validate_config


class TestConfig:
    """Test configuration loading"""

    def test_config_loads(self):
        """Config should load without errors"""
        assert Config is not None
        assert hasattr(Config, 'DEFAULT_TARGET_GROSS_MARGIN')

    def test_config_validation(self):
        """Config validation should return tuple"""
        is_valid, missing = validate_config()
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)

    def test_get_clients_without_keys(self):
        """get_clients() should handle missing API keys gracefully"""
        clients = get_clients()
        assert isinstance(clients, dict)
        assert 'google_analytics' in clients
        assert 'mixpanel' in clients
        # Clients may be None if keys not configured - this is expected


class TestValueProposition:
    """Test value proposition creation and validation"""

    def test_create_value_prop(self):
        """Should create value proposition"""
        vp = ValueProposition(
            target_customer="Financial compliance teams",
            pain_point="waste 15 hours per week on manual search",
            solution_outcome="Find documents in seconds",
            unique_differentiator="AI-powered semantic search"
        )
        assert vp is not None
        assert vp.target_customer == "Financial compliance teams"

    def test_value_prop_headline(self):
        """Should generate headline"""
        vp = ValueProposition(
            target_customer="Test customer",
            pain_point="test pain",
            solution_outcome="Test outcome",
            unique_differentiator="test diff"
        )
        headline = vp.to_headline()
        assert isinstance(headline, str)
        assert len(headline) > 0

    def test_value_prop_validation(self):
        """Should validate value proposition"""
        vp = ValueProposition(
            target_customer="Financial services compliance teams",
            pain_point="waste 15+ hours per week",
            solution_outcome="Find documents in seconds",
            unique_differentiator="Purpose-built for compliance"
        )
        is_valid, issues = vp.validate()
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


class TestPricingCalculator:
    """Test pricing calculation logic"""

    def test_calculate_minimum_price(self):
        """Should calculate minimum price from COGS"""
        calc = PricingCalculator()
        min_price = calc.calculate_minimum_price(
            cogs_per_customer=70,
            target_gross_margin=0.67
        )
        assert min_price > 70  # Must be higher than COGS
        assert min_price < 300  # Reasonable upper bound
        # Expected: 70 / 0.33 = ~212

    def test_calculate_value_based_price(self):
        """Should calculate price based on value delivered"""
        calc = PricingCalculator()
        value_price = calc.calculate_value_based_price(
            hours_saved_per_week=10,
            hourly_labor_cost=100,
            value_capture_rate=0.25
        )
        assert value_price > 0
        # 10 hours * 4 weeks * 100 * 0.25 = 1000
        assert 900 < value_price < 1100

    def test_design_tier_structure(self):
        """Should create 3-tier pricing structure"""
        calc = PricingCalculator()
        tiers = calc.design_tier_structure(
            min_price=200,
            value_price=250
        )
        assert len(tiers) == 3
        assert all(isinstance(t, PricingTierConfig) for t in tiers)
        assert tiers[0].name == "Starter"
        assert tiers[1].name == "Professional"
        assert tiers[2].name == "Enterprise"
        # Professional should be most popular
        assert tiers[1].is_most_popular

    def test_tier_roi_calculation(self):
        """Should calculate ROI for pricing tier"""
        tier = PricingTierConfig(
            name="Test",
            monthly_price=499,
            query_limit=2500,
            document_limit=25000,
            user_limit=10,
            support_level="Email"
        )
        roi = tier.calculate_roi(hours_saved_per_week=10, hourly_labor_cost=100)
        assert 'monthly_value_delivered' in roi
        assert 'roi_multiplier' in roi
        assert roi['roi_multiplier'] > 0


class TestConversionFunnel:
    """Test conversion funnel analysis"""

    def test_calculate_funnel_metrics(self):
        """Should analyze conversion funnel"""
        analyzer = ConversionFunnelAnalyzer()
        funnel = analyzer.calculate_funnel_metrics(
            visitors=1000,
            signups=50,
            activated=25,
            paid=5
        )
        assert 'metrics' in funnel
        assert 'benchmarks' in funnel
        assert 'issues' in funnel
        assert funnel['metrics']['signup_rate'] == 0.05
        assert funnel['metrics']['overall_conversion'] == 0.005

    def test_funnel_identifies_issues(self):
        """Should identify bottlenecks in funnel"""
        analyzer = ConversionFunnelAnalyzer()
        # Bad funnel: low activation
        funnel = analyzer.calculate_funnel_metrics(
            visitors=1000,
            signups=100,  # Good signup rate
            activated=10,  # Bad activation (10%)
            paid=2
        )
        assert len(funnel['issues']) > 0
        assert any('activation' in issue.lower() for issue in funnel['issues'])


class TestCACLTV:
    """Test CAC and LTV calculations"""

    def test_calculate_cac(self):
        """Should calculate customer acquisition cost"""
        calc = CAC_LTV_Calculator()
        cac = calc.calculate_cac(
            total_marketing_spend=1500,
            total_sales_spend=500,
            new_customers=5
        )
        assert cac == 400  # 2000 / 5

    def test_calculate_cac_zero_customers(self):
        """Should handle zero customers gracefully"""
        calc = CAC_LTV_Calculator()
        cac = calc.calculate_cac(
            total_marketing_spend=1000,
            total_sales_spend=0,
            new_customers=0
        )
        assert cac == float('inf')

    def test_calculate_ltv(self):
        """Should calculate lifetime value"""
        calc = CAC_LTV_Calculator()
        ltv = calc.calculate_ltv(
            monthly_recurring_revenue=499,
            gross_margin=0.67,
            churn_rate_monthly=0.05
        )
        assert ltv > 0
        # LTV = 499 * 0.67 * (1/0.05) = 499 * 0.67 * 20 = ~6,686
        assert 6000 < ltv < 7000

    def test_calculate_unit_economics(self):
        """Should analyze unit economics"""
        calc = CAC_LTV_Calculator()
        unit_econ = calc.calculate_unit_economics(
            cac=400,
            ltv=6686,
            target_ratio=3.0
        )
        assert 'ltv_cac_ratio' in unit_econ
        assert 'is_healthy' in unit_econ
        assert 'health_status' in unit_econ
        assert unit_econ['ltv_cac_ratio'] > 3.0  # Should be healthy
        assert unit_econ['is_healthy'] is True


class TestGTMStrategy:
    """Test GTM strategy selection"""

    def test_recommend_self_service(self):
        """Should recommend self-service for mid-market"""
        strategy = GTMStrategySelector.recommend_gtm_motion(
            annual_contract_value=5000,
            addressable_market_size=15000,
            product_complexity_minutes=15,
            has_partner_ecosystem=False
        )
        assert 'primary_recommendation' in strategy
        assert strategy['primary_recommendation']['motion'] == GTMMotion.SELF_SERVICE

    def test_recommend_direct_sales(self):
        """Should recommend direct sales for high ACV"""
        strategy = GTMStrategySelector.recommend_gtm_motion(
            annual_contract_value=50000,
            addressable_market_size=500,
            product_complexity_minutes=30,
            has_partner_ecosystem=False
        )
        assert strategy['primary_recommendation']['motion'] == GTMMotion.DIRECT_SALES

    def test_recommend_freemium(self):
        """Should recommend freemium for low ACV + quick demo"""
        strategy = GTMStrategySelector.recommend_gtm_motion(
            annual_contract_value=500,
            addressable_market_size=50000,
            product_complexity_minutes=3,
            has_partner_ecosystem=False
        )
        assert strategy['primary_recommendation']['motion'] == GTMMotion.FREEMIUM


class TestUTMTracking:
    """Test UTM parameter handling"""

    def test_generate_utm_url(self):
        """Should generate URL with UTM parameters"""
        tracker = UTMTracker()
        url = tracker.generate_utm_url(
            base_url="https://example.com",
            source="linkedin",
            medium="cpc",
            campaign="test_campaign"
        )
        assert "utm_source=linkedin" in url
        assert "utm_medium=cpc" in url
        assert "utm_campaign=test_campaign" in url

    def test_parse_attribution_data(self):
        """Should analyze attribution from customer records"""
        tracker = UTMTracker()
        customers = [
            {'utm_source': 'twitter', 'utm_campaign': 'launch', 'is_paying': True, 'mrr': 499},
            {'utm_source': 'twitter', 'utm_campaign': 'launch', 'is_paying': True, 'mrr': 199},
            {'utm_source': 'linkedin', 'utm_campaign': 'q4', 'is_paying': False, 'mrr': 0},
        ]
        attribution = tracker.parse_attribution_data(customers)
        assert 'twitter/launch' in attribution
        assert attribution['twitter/launch']['paid_customers'] == 2
        assert attribution['twitter/launch']['total_mrr'] == 698


class TestLaunchReadiness:
    """Test launch readiness validation"""

    def test_validate_launch_ready(self):
        """Should validate complete checklist"""
        checklist = {
            'landing_page_deployed': True,
            'analytics_configured': True,
            'signup_flow_tested': True,
            'pricing_finalized': True,
            'gtm_plan_created': True
        }
        is_ready, missing = validate_launch_readiness(checklist)
        assert is_ready is True
        assert len(missing) == 0

    def test_validate_launch_not_ready(self):
        """Should identify missing items"""
        checklist = {
            'landing_page_deployed': True,
            'analytics_configured': False,
            'signup_flow_tested': True,
            'pricing_finalized': False,
            'gtm_plan_created': True
        }
        is_ready, missing = validate_launch_readiness(checklist)
        assert is_ready is False
        assert 'analytics_configured' in missing
        assert 'pricing_finalized' in missing


if __name__ == "__main__":
    # Run with: python tests_smoke.py
    pytest.main([__file__, "-v"])
