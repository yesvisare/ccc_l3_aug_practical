"""
Module 13.3: Launch Preparation & Marketing
Enterprise RAG SaaS - Go-to-Market Strategy and Launch Execution

This module provides functionality for:
- Value proposition generation and validation
- Pricing strategy design based on value metrics
- Go-to-market (GTM) plan creation
- Analytics tracking and conversion funnel analysis
- Customer acquisition cost (CAC) and lifetime value (LTV) calculations
- Attribution tracking and reporting
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GTMMotion(Enum):
    """Go-to-market motion types"""
    SELF_SERVICE = "self_service"
    DIRECT_SALES = "direct_sales"
    PARTNER_LED = "partner_led"
    FREEMIUM = "freemium"


class PricingTier(Enum):
    """Pricing tier levels"""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class ValueProposition:
    """Structured value proposition following best practices"""
    target_customer: str
    pain_point: str
    solution_outcome: str
    unique_differentiator: str
    quantified_benefit: Optional[str] = None

    def to_headline(self) -> str:
        """Generate hero headline from value prop"""
        return f"{self.solution_outcome}"

    def to_subheadline(self) -> str:
        """Generate subheadline with pain and benefit"""
        benefit = f" {self.quantified_benefit}" if self.quantified_benefit else ""
        return f"{self.unique_differentiator}.{benefit}"

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate value proposition for specificity and clarity"""
        issues = []

        # Check for vague terms
        vague_terms = ['modern', 'innovative', 'revolutionary', 'cutting-edge', 'next-gen']
        text = f"{self.target_customer} {self.pain_point} {self.solution_outcome}".lower()

        for term in vague_terms:
            if term in text:
                issues.append(f"Avoid vague buzzword: '{term}'")

        # Check for specificity
        if len(self.target_customer.split()) < 3:
            issues.append("Target customer too generic - be more specific")

        if not any(char.isdigit() for char in self.pain_point):
            issues.append("Pain point lacks quantification - add numbers/metrics")

        return len(issues) == 0, issues


@dataclass
class PricingTierConfig:
    """Configuration for a single pricing tier"""
    name: str
    monthly_price: float
    query_limit: int
    document_limit: int
    user_limit: int
    support_level: str
    features: List[str] = field(default_factory=list)
    is_most_popular: bool = False

    def calculate_roi(self, hours_saved_per_week: float, hourly_labor_cost: float = 100.0) -> Dict[str, float]:
        """Calculate ROI metrics for this tier"""
        monthly_value = hours_saved_per_week * 4 * hourly_labor_cost
        roi_multiplier = monthly_value / self.monthly_price if self.monthly_price > 0 else 0
        annual_savings = (monthly_value - self.monthly_price) * 12

        return {
            'monthly_value_delivered': monthly_value,
            'monthly_price': self.monthly_price,
            'roi_multiplier': roi_multiplier,
            'annual_savings': annual_savings,
            'payback_period_months': 1.0  # Immediate payback if positive ROI
        }


@dataclass
class IdealCustomerProfile:
    """Ideal Customer Profile (ICP) definition"""
    industry: str
    company_size_min: int
    company_size_max: int
    geography: List[str]
    job_titles: List[str]
    department: str
    pain_intensity: str  # e.g., "15+ hours/week wasted"
    budget_range_annual: Tuple[int, int]
    tech_maturity: str
    red_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ICP to dictionary format"""
        return {
            'industry': self.industry,
            'company_size': f"{self.company_size_min}-{self.company_size_max} employees",
            'geography': self.geography,
            'job_titles': self.job_titles,
            'department': self.department,
            'pain_intensity': self.pain_intensity,
            'budget_range': f"${self.budget_range_annual[0]:,}-${self.budget_range_annual[1]:,}/year",
            'tech_maturity': self.tech_maturity,
            'red_flags': self.red_flags
        }


@dataclass
class AcquisitionChannel:
    """Customer acquisition channel configuration"""
    name: str
    channel_type: str  # e.g., 'organic', 'paid', 'outbound'
    time_per_week: float  # hours
    monthly_cost: float
    expected_leads_per_month: int
    conversion_rate: float  # decimal, e.g., 0.02 for 2%
    priority: int  # 1 = highest priority

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate channel performance metrics"""
        expected_conversions = self.expected_leads_per_month * self.conversion_rate
        cost_per_lead = self.monthly_cost / self.expected_leads_per_month if self.expected_leads_per_month > 0 else 0
        cost_per_customer = self.monthly_cost / expected_conversions if expected_conversions > 0 else float('inf')

        return {
            'expected_conversions': expected_conversions,
            'cost_per_lead': cost_per_lead,
            'cost_per_customer': cost_per_customer,
            'time_investment_hours': time_per_week * 4
        }


class PricingCalculator:
    """Calculate pricing based on COGS, value metrics, and market positioning"""

    @staticmethod
    def calculate_minimum_price(
        cogs_per_customer: float,
        target_gross_margin: float = 0.67
    ) -> float:
        """
        Calculate minimum viable price based on COGS and target margin

        Args:
            cogs_per_customer: Monthly cost of goods sold per customer
            target_gross_margin: Target gross margin (default 67% for healthy SaaS)

        Returns:
            Minimum monthly price
        """
        if target_gross_margin >= 1.0 or target_gross_margin < 0:
            logger.error(f"Invalid target_gross_margin: {target_gross_margin}")
            raise ValueError("Target gross margin must be between 0 and 1")

        min_price = cogs_per_customer / (1 - target_gross_margin)
        logger.info(f"Minimum price: ${min_price:.2f} (COGS: ${cogs_per_customer}, Margin: {target_gross_margin*100}%)")
        return min_price

    @staticmethod
    def calculate_value_based_price(
        hours_saved_per_week: float,
        hourly_labor_cost: float = 100.0,
        value_capture_rate: float = 0.25
    ) -> float:
        """
        Calculate price based on value delivered to customer

        Args:
            hours_saved_per_week: Time saved for customer
            hourly_labor_cost: Customer's internal labor cost
            value_capture_rate: % of value to capture (20-30% typical)

        Returns:
            Monthly price based on value
        """
        monthly_value = hours_saved_per_week * 4 * hourly_labor_cost
        price = monthly_value * value_capture_rate

        logger.info(f"Value-based price: ${price:.2f} (Value: ${monthly_value}, Capture: {value_capture_rate*100}%)")
        return price

    @staticmethod
    def design_tier_structure(
        min_price: float,
        value_price: float,
        query_tiers: List[int] = [500, 2500, 10000]
    ) -> List[PricingTierConfig]:
        """
        Design 3-tier pricing structure

        Args:
            min_price: Minimum viable price from COGS
            value_price: Value-based price calculation
            query_tiers: Query limits for each tier

        Returns:
            List of pricing tier configurations
        """
        # Use value-based pricing, but ensure above minimum
        base_price = max(min_price, value_price)

        tiers = [
            PricingTierConfig(
                name="Starter",
                monthly_price=round(base_price, -1),  # Round to nearest 10
                query_limit=query_tiers[0],
                document_limit=5000,
                user_limit=2,
                support_level="Email support",
                features=["Basic semantic search", "Email support", "Standard integrations"]
            ),
            PricingTierConfig(
                name="Professional",
                monthly_price=round(base_price * 2.5, -1),
                query_limit=query_tiers[1],
                document_limit=25000,
                user_limit=10,
                support_level="Priority email + chat",
                features=["Advanced retrieval", "Priority support", "Custom integrations", "API access"],
                is_most_popular=True
            ),
            PricingTierConfig(
                name="Enterprise",
                monthly_price=round(base_price * 7.5, -1),
                query_limit=query_tiers[2],
                document_limit=999999,
                user_limit=999,
                support_level="Dedicated account manager + SLA",
                features=["Unlimited queries", "Dedicated support", "SSO/SAML", "Custom compliance reports", "SLA guarantees"]
            )
        ]

        logger.info(f"Created {len(tiers)} pricing tiers: {[f'{t.name} ${t.monthly_price}' for t in tiers]}")
        return tiers


class ConversionFunnelAnalyzer:
    """Analyze conversion funnel metrics and identify optimization opportunities"""

    @staticmethod
    def calculate_funnel_metrics(
        visitors: int,
        signups: int,
        activated: int,
        paid: int
    ) -> Dict[str, Any]:
        """
        Calculate conversion funnel metrics

        Args:
            visitors: Landing page visitors
            signups: Trial signups
            activated: Users who ran first query
            paid: Converted to paying customers

        Returns:
            Dictionary of conversion metrics and diagnostics
        """
        # Avoid division by zero
        signup_rate = signups / visitors if visitors > 0 else 0
        activation_rate = activated / signups if signups > 0 else 0
        paid_conversion_rate = paid / activated if activated > 0 else 0
        overall_conversion = paid / visitors if visitors > 0 else 0

        # Industry benchmarks
        benchmarks = {
            'signup_rate': {'target': 0.05, 'good': 0.10},
            'activation_rate': {'target': 0.50, 'good': 0.70},
            'paid_conversion_rate': {'target': 0.20, 'good': 0.40}
        }

        # Identify bottlenecks
        issues = []
        if signup_rate < benchmarks['signup_rate']['target']:
            issues.append("Low signup rate: Check value proposition clarity and landing page design")
        if activation_rate < benchmarks['activation_rate']['target']:
            issues.append("Low activation rate: Onboarding is too complex or product UX is confusing")
        if paid_conversion_rate < benchmarks['paid_conversion_rate']['target']:
            issues.append("Low paid conversion: Pricing may be too high or trial doesn't demonstrate value")

        return {
            'metrics': {
                'visitors': visitors,
                'signups': signups,
                'activated': activated,
                'paid': paid,
                'signup_rate': signup_rate,
                'activation_rate': activation_rate,
                'paid_conversion_rate': paid_conversion_rate,
                'overall_conversion': overall_conversion
            },
            'benchmarks': benchmarks,
            'issues': issues,
            'strongest_step': max([
                ('signup', signup_rate),
                ('activation', activation_rate),
                ('paid_conversion', paid_conversion_rate)
            ], key=lambda x: x[1])[0],
            'weakest_step': min([
                ('signup', signup_rate),
                ('activation', activation_rate),
                ('paid_conversion', paid_conversion_rate)
            ], key=lambda x: x[1])[0]
        }


class CAC_LTV_Calculator:
    """Calculate Customer Acquisition Cost (CAC) and Lifetime Value (LTV) metrics"""

    @staticmethod
    def calculate_cac(
        total_marketing_spend: float,
        total_sales_spend: float,
        new_customers: int
    ) -> float:
        """
        Calculate Customer Acquisition Cost

        Args:
            total_marketing_spend: Total marketing expenses
            total_sales_spend: Total sales expenses
            new_customers: Number of new customers acquired

        Returns:
            CAC per customer
        """
        if new_customers == 0:
            logger.warning("No new customers acquired, CAC is undefined")
            return float('inf')

        cac = (total_marketing_spend + total_sales_spend) / new_customers
        logger.info(f"CAC: ${cac:.2f} ({new_customers} customers, ${total_marketing_spend + total_sales_spend:.2f} spend)")
        return cac

    @staticmethod
    def calculate_ltv(
        monthly_recurring_revenue: float,
        gross_margin: float,
        churn_rate_monthly: float
    ) -> float:
        """
        Calculate Customer Lifetime Value

        Args:
            monthly_recurring_revenue: MRR per customer
            gross_margin: Gross margin percentage (decimal)
            churn_rate_monthly: Monthly churn rate (decimal)

        Returns:
            LTV per customer
        """
        if churn_rate_monthly == 0:
            logger.warning("Zero churn rate, using default 5% monthly churn")
            churn_rate_monthly = 0.05

        avg_customer_lifetime_months = 1 / churn_rate_monthly
        ltv = monthly_recurring_revenue * gross_margin * avg_customer_lifetime_months

        logger.info(f"LTV: ${ltv:.2f} (MRR: ${monthly_recurring_revenue}, Margin: {gross_margin*100}%, Churn: {churn_rate_monthly*100}%)")
        return ltv

    @staticmethod
    def calculate_unit_economics(
        cac: float,
        ltv: float,
        target_ratio: float = 3.0
    ) -> Dict[str, Any]:
        """
        Calculate unit economics and health metrics

        Args:
            cac: Customer acquisition cost
            ltv: Customer lifetime value
            target_ratio: Target LTV:CAC ratio (3:1 is healthy)

        Returns:
            Unit economics analysis
        """
        ltv_cac_ratio = ltv / cac if cac > 0 else 0
        is_healthy = ltv_cac_ratio >= target_ratio

        if ltv_cac_ratio < 1.0:
            health_status = "CRITICAL: Losing money on every customer"
        elif ltv_cac_ratio < target_ratio:
            health_status = "WARNING: Below target ratio, not sustainable"
        else:
            health_status = "HEALTHY: Good unit economics"

        return {
            'cac': cac,
            'ltv': ltv,
            'ltv_cac_ratio': ltv_cac_ratio,
            'target_ratio': target_ratio,
            'is_healthy': is_healthy,
            'health_status': health_status,
            'payback_period_months': cac / (ltv / 18) if ltv > 0 else float('inf')  # Assuming 18-month avg lifetime
        }


class GTMStrategySelector:
    """Select appropriate go-to-market strategy based on product and market characteristics"""

    @staticmethod
    def recommend_gtm_motion(
        annual_contract_value: float,
        addressable_market_size: int,
        product_complexity_minutes: int,
        has_partner_ecosystem: bool = False
    ) -> Dict[str, Any]:
        """
        Recommend GTM motion based on product characteristics

        Args:
            annual_contract_value: Average annual contract value
            addressable_market_size: Number of potential customers
            product_complexity_minutes: Time to understand product value
            has_partner_ecosystem: Whether established partners exist

        Returns:
            Recommended GTM strategy with reasoning
        """
        recommendations = []

        # Decision tree based on ACV
        if annual_contract_value > 10000:
            if has_partner_ecosystem:
                recommendations.append({
                    'motion': GTMMotion.PARTNER_LED,
                    'confidence': 'high',
                    'reasoning': 'High ACV + partner ecosystem = partner-led growth ideal'
                })
            recommendations.append({
                'motion': GTMMotion.DIRECT_SALES,
                'confidence': 'high',
                'reasoning': 'ACV > $10K requires sales-assisted motion'
            })
        elif annual_contract_value < 1000:
            if product_complexity_minutes < 5:
                recommendations.append({
                    'motion': GTMMotion.FREEMIUM,
                    'confidence': 'high',
                    'reasoning': 'Low ACV + quick value demo = freemium ideal'
                })
            else:
                recommendations.append({
                    'motion': GTMMotion.SELF_SERVICE,
                    'confidence': 'medium',
                    'reasoning': 'Low ACV but complex = risky self-service'
                })
        else:  # $1K-10K sweet spot
            if addressable_market_size > 10000:
                recommendations.append({
                    'motion': GTMMotion.SELF_SERVICE,
                    'confidence': 'high',
                    'reasoning': 'Mid-market ACV + large TAM = self-service PLG'
                })
            else:
                recommendations.append({
                    'motion': GTMMotion.DIRECT_SALES,
                    'confidence': 'medium',
                    'reasoning': 'Mid-market ACV + niche market = direct sales safer'
                })

        # Check complexity flag
        if product_complexity_minutes > 30:
            for rec in recommendations:
                if rec['motion'] == GTMMotion.SELF_SERVICE:
                    rec['confidence'] = 'low'
                    rec['reasoning'] += ' (WARNING: Complex products struggle with self-service)'

        primary = recommendations[0] if recommendations else {
            'motion': GTMMotion.DIRECT_SALES,
            'confidence': 'low',
            'reasoning': 'Default to direct sales when uncertain'
        }

        logger.info(f"GTM recommendation: {primary['motion'].value} ({primary['confidence']} confidence)")
        return {
            'primary_recommendation': primary,
            'alternatives': recommendations[1:] if len(recommendations) > 1 else [],
            'inputs': {
                'acv': annual_contract_value,
                'market_size': addressable_market_size,
                'complexity_minutes': product_complexity_minutes
            }
        }


class UTMTracker:
    """Manage UTM parameters and attribution tracking"""

    @staticmethod
    def generate_utm_url(
        base_url: str,
        source: str,
        medium: str,
        campaign: str,
        term: Optional[str] = None,
        content: Optional[str] = None
    ) -> str:
        """
        Generate URL with UTM parameters

        Args:
            base_url: Base URL to append parameters to
            source: Traffic source (e.g., 'linkedin', 'twitter')
            medium: Medium type (e.g., 'cpc', 'social', 'email')
            campaign: Campaign name
            term: Optional campaign term (for paid search keywords)
            content: Optional content differentiator (for A/B testing)

        Returns:
            URL with UTM parameters
        """
        params = f"utm_source={source}&utm_medium={medium}&utm_campaign={campaign}"

        if term:
            params += f"&utm_term={term}"
        if content:
            params += f"&utm_content={content}"

        separator = '&' if '?' in base_url else '?'
        url = f"{base_url}{separator}{params}"

        logger.info(f"Generated UTM URL: {source}/{medium}/{campaign}")
        return url

    @staticmethod
    def parse_attribution_data(
        customer_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze attribution data to identify best-performing channels

        Args:
            customer_records: List of customer dictionaries with utm_* fields

        Returns:
            Attribution report by source/campaign
        """
        attribution = {}

        for record in customer_records:
            source = record.get('utm_source', 'unknown')
            campaign = record.get('utm_campaign', 'unknown')
            mrr = record.get('mrr', 0)

            key = f"{source}/{campaign}"

            if key not in attribution:
                attribution[key] = {
                    'signups': 0,
                    'paid_customers': 0,
                    'total_mrr': 0,
                    'conversion_rate': 0
                }

            attribution[key]['signups'] += 1
            if record.get('is_paying', False):
                attribution[key]['paid_customers'] += 1
                attribution[key]['total_mrr'] += mrr

        # Calculate conversion rates
        for key in attribution:
            signups = attribution[key]['signups']
            paid = attribution[key]['paid_customers']
            attribution[key]['conversion_rate'] = paid / signups if signups > 0 else 0

        # Sort by total MRR
        sorted_attribution = dict(sorted(
            attribution.items(),
            key=lambda x: x[1]['total_mrr'],
            reverse=True
        ))

        logger.info(f"Attribution analysis: {len(sorted_attribution)} channels analyzed")
        return sorted_attribution


def validate_launch_readiness(checklist: Dict[str, bool]) -> Tuple[bool, List[str]]:
    """
    Validate launch readiness based on checklist

    Args:
        checklist: Dictionary of requirement -> completion status

    Returns:
        Tuple of (is_ready, list of missing items)
    """
    required_items = [
        'landing_page_deployed',
        'analytics_configured',
        'signup_flow_tested',
        'pricing_finalized',
        'gtm_plan_created'
    ]

    missing = []
    for item in required_items:
        if not checklist.get(item, False):
            missing.append(item)

    is_ready = len(missing) == 0

    if is_ready:
        logger.info("✅ Launch readiness check PASSED - ready to drive traffic")
    else:
        logger.error(f"❌ Launch readiness check FAILED - missing: {missing}")

    return is_ready, missing


# CLI usage examples
if __name__ == "__main__":
    print("=== Module 13.3: Launch Preparation & Marketing ===\n")

    # Example 1: Create and validate value proposition
    print("1. Value Proposition Creation")
    vp = ValueProposition(
        target_customer="Financial services compliance teams",
        pain_point="waste 15+ hours per week manually searching regulatory documents",
        solution_outcome="Find any regulatory document in seconds",
        unique_differentiator="AI-powered semantic search with audit-ready citations",
        quantified_benefit="Save 12 hours/week on average"
    )
    print(f"   Headline: {vp.to_headline()}")
    print(f"   Subheadline: {vp.to_subheadline()}")
    is_valid, issues = vp.validate()
    print(f"   Valid: {is_valid}")
    if issues:
        print(f"   Issues: {issues}")
    print()

    # Example 2: Calculate pricing
    print("2. Pricing Calculation")
    calc = PricingCalculator()

    cogs = 70.0  # Pinecone + OpenAI + infra
    min_price = calc.calculate_minimum_price(cogs, target_gross_margin=0.67)
    print(f"   Minimum price (67% margin): ${min_price:.2f}")

    value_price = calc.calculate_value_based_price(
        hours_saved_per_week=10,
        hourly_labor_cost=100,
        value_capture_rate=0.25
    )
    print(f"   Value-based price (25% capture): ${value_price:.2f}")

    tiers = calc.design_tier_structure(min_price, value_price)
    for tier in tiers:
        roi = tier.calculate_roi(hours_saved_per_week=tier.query_limit / 500 * 2)
        print(f"   {tier.name}: ${tier.monthly_price}/mo - {roi['roi_multiplier']:.1f}x ROI")
    print()

    # Example 3: GTM strategy recommendation
    print("3. GTM Strategy Selection")
    strategy = GTMStrategySelector.recommend_gtm_motion(
        annual_contract_value=5988,  # $499/month
        addressable_market_size=15000,
        product_complexity_minutes=15,
        has_partner_ecosystem=False
    )
    print(f"   Recommended: {strategy['primary_recommendation']['motion'].value}")
    print(f"   Confidence: {strategy['primary_recommendation']['confidence']}")
    print(f"   Reasoning: {strategy['primary_recommendation']['reasoning']}")
    print()

    # Example 4: Analyze conversion funnel
    print("4. Conversion Funnel Analysis")
    analyzer = ConversionFunnelAnalyzer()
    funnel = analyzer.calculate_funnel_metrics(
        visitors=1000,
        signups=50,
        activated=25,
        paid=5
    )
    print(f"   Signup rate: {funnel['metrics']['signup_rate']*100:.1f}%")
    print(f"   Activation rate: {funnel['metrics']['activation_rate']*100:.1f}%")
    print(f"   Paid conversion: {funnel['metrics']['paid_conversion_rate']*100:.1f}%")
    print(f"   Weakest step: {funnel['weakest_step']}")
    if funnel['issues']:
        print(f"   Issues detected:")
        for issue in funnel['issues']:
            print(f"     - {issue}")
    print()

    # Example 5: Calculate unit economics
    print("5. Unit Economics (CAC/LTV)")
    calc_ltv = CAC_LTV_Calculator()

    cac = calc_ltv.calculate_cac(
        total_marketing_spend=1500,
        total_sales_spend=500,
        new_customers=5
    )

    ltv = calc_ltv.calculate_ltv(
        monthly_recurring_revenue=499,
        gross_margin=0.67,
        churn_rate_monthly=0.05
    )

    unit_econ = calc_ltv.calculate_unit_economics(cac, ltv, target_ratio=3.0)
    print(f"   CAC: ${unit_econ['cac']:.2f}")
    print(f"   LTV: ${unit_econ['ltv']:.2f}")
    print(f"   LTV:CAC Ratio: {unit_econ['ltv_cac_ratio']:.1f}x")
    print(f"   Status: {unit_econ['health_status']}")
    print()

    # Example 6: Generate UTM URLs
    print("6. UTM Tracking")
    tracker = UTMTracker()
    url = tracker.generate_utm_url(
        base_url="https://compliancecopilot.ai",
        source="linkedin",
        medium="cpc",
        campaign="compliance_q4_2024",
        content="variant_a"
    )
    print(f"   UTM URL: {url}")
    print()

    print("⚠️  Note: Skipping API calls (no external services required for core calculations)")
