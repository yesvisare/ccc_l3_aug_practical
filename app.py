"""
Module 13.3: Launch Preparation & Marketing - FastAPI Application

RESTful API for launch planning, pricing strategy, and marketing analytics.
No business logic in this file - imports and calls functions from l3_m13_launch_prep_marketing.py
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# Import core functionality
from l3_m13_launch_prep_marketing import (
    ValueProposition,
    PricingCalculator,
    ConversionFunnelAnalyzer,
    CAC_LTV_Calculator,
    GTMStrategySelector,
    UTMTracker,
    PricingTierConfig,
    GTMMotion,
    validate_launch_readiness
)
from config import Config, get_clients, validate_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Launch Preparation & Marketing API",
    description="Module 13.3: APIs for SaaS launch planning, pricing strategy, and GTM execution",
    version="1.0.0"
)


# Request/Response Models
class ValuePropRequest(BaseModel):
    target_customer: str = Field(..., description="Specific target customer segment")
    pain_point: str = Field(..., description="Quantified pain point")
    solution_outcome: str = Field(..., description="Specific solution outcome")
    unique_differentiator: str = Field(..., description="What makes you different")
    quantified_benefit: Optional[str] = Field(None, description="Optional quantified benefit")


class PricingRequest(BaseModel):
    cogs_per_customer: float = Field(..., description="Monthly COGS per customer", gt=0)
    hours_saved_per_week: float = Field(..., description="Hours saved for customer", gt=0)
    hourly_labor_cost: float = Field(100.0, description="Customer's hourly labor cost")
    target_gross_margin: float = Field(0.67, description="Target gross margin (0-1)", ge=0, lt=1)
    value_capture_rate: float = Field(0.25, description="% of value to capture (0-1)", ge=0, lt=1)


class FunnelMetricsRequest(BaseModel):
    visitors: int = Field(..., description="Landing page visitors", ge=0)
    signups: int = Field(..., description="Trial signups", ge=0)
    activated: int = Field(..., description="Activated users", ge=0)
    paid: int = Field(..., description="Paying customers", ge=0)


class CACLTVRequest(BaseModel):
    total_marketing_spend: float = Field(..., description="Total marketing spend", ge=0)
    total_sales_spend: float = Field(0.0, description="Total sales spend", ge=0)
    new_customers: int = Field(..., description="New customers acquired", ge=0)
    monthly_recurring_revenue: float = Field(..., description="MRR per customer", gt=0)
    gross_margin: float = Field(0.67, description="Gross margin", ge=0, lt=1)
    churn_rate_monthly: float = Field(0.05, description="Monthly churn rate", ge=0, lt=1)


class GTMStrategyRequest(BaseModel):
    annual_contract_value: float = Field(..., description="Annual contract value", gt=0)
    addressable_market_size: int = Field(..., description="Number of potential customers", gt=0)
    product_complexity_minutes: int = Field(..., description="Minutes to understand product", ge=0)
    has_partner_ecosystem: bool = Field(False, description="Whether partners exist")


class UTMGenerateRequest(BaseModel):
    base_url: str = Field(..., description="Base URL")
    source: str = Field(..., description="Traffic source (e.g., linkedin, twitter)")
    medium: str = Field(..., description="Medium type (e.g., cpc, social)")
    campaign: str = Field(..., description="Campaign name")
    term: Optional[str] = Field(None, description="Campaign term")
    content: Optional[str] = Field(None, description="Content variant")


class AttributionRecord(BaseModel):
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None
    is_paying: bool = False
    mrr: float = 0.0


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "launch_prep_marketing", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Module 13.3: Launch Preparation & Marketing",
        "endpoints": {
            "health": "GET /health",
            "config": "GET /config",
            "value_prop": "POST /value-proposition",
            "pricing": "POST /pricing/calculate",
            "funnel": "POST /analytics/funnel",
            "cac_ltv": "POST /analytics/cac-ltv",
            "gtm_strategy": "POST /strategy/gtm",
            "utm_generate": "POST /utm/generate",
            "utm_attribution": "POST /utm/attribution"
        }
    }


@app.get("/config")
async def get_config_status():
    """
    Get configuration status and check if external services are available
    """
    is_valid, missing = validate_config()
    clients = get_clients()

    # Check which clients are configured
    configured_services = {
        name: (client is not None)
        for name, client in clients.items()
    }

    return {
        "config_valid": is_valid,
        "missing_configs": missing,
        "services_configured": configured_services,
        "note": "This module can run without external services for core calculations"
    }


@app.post("/value-proposition")
async def create_value_proposition(request: ValuePropRequest):
    """
    Create and validate a value proposition

    Returns headline, subheadline, and validation results
    """
    try:
        vp = ValueProposition(
            target_customer=request.target_customer,
            pain_point=request.pain_point,
            solution_outcome=request.solution_outcome,
            unique_differentiator=request.unique_differentiator,
            quantified_benefit=request.quantified_benefit
        )

        is_valid, issues = vp.validate()

        return {
            "value_proposition": {
                "headline": vp.to_headline(),
                "subheadline": vp.to_subheadline(),
                "target_customer": vp.target_customer
            },
            "validation": {
                "is_valid": is_valid,
                "issues": issues
            }
        }
    except Exception as e:
        logger.error(f"Error creating value proposition: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pricing/calculate")
async def calculate_pricing(request: PricingRequest):
    """
    Calculate pricing tiers based on COGS and value metrics

    Returns 3-tier pricing structure with ROI calculations
    """
    try:
        calc = PricingCalculator()

        # Calculate minimum and value-based prices
        min_price = calc.calculate_minimum_price(
            request.cogs_per_customer,
            request.target_gross_margin
        )

        value_price = calc.calculate_value_based_price(
            request.hours_saved_per_week,
            request.hourly_labor_cost,
            request.value_capture_rate
        )

        # Design tier structure
        tiers = calc.design_tier_structure(min_price, value_price)

        # Calculate ROI for each tier
        tiers_with_roi = []
        for tier in tiers:
            hours_for_tier = request.hours_saved_per_week * (tier.query_limit / 2500)
            roi = tier.calculate_roi(hours_for_tier, request.hourly_labor_cost)

            tiers_with_roi.append({
                "name": tier.name,
                "monthly_price": tier.monthly_price,
                "limits": {
                    "queries": tier.query_limit,
                    "documents": tier.document_limit,
                    "users": tier.user_limit
                },
                "support": tier.support_level,
                "features": tier.features,
                "is_most_popular": tier.is_most_popular,
                "roi": roi
            })

        return {
            "pricing_analysis": {
                "minimum_price": round(min_price, 2),
                "value_based_price": round(value_price, 2),
                "recommended_base": round(max(min_price, value_price), 2)
            },
            "tiers": tiers_with_roi
        }
    except Exception as e:
        logger.error(f"Error calculating pricing: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/funnel")
async def analyze_funnel(request: FunnelMetricsRequest):
    """
    Analyze conversion funnel metrics and identify bottlenecks

    Returns conversion rates, benchmarks, and optimization recommendations
    """
    try:
        analyzer = ConversionFunnelAnalyzer()
        funnel = analyzer.calculate_funnel_metrics(
            request.visitors,
            request.signups,
            request.activated,
            request.paid
        )

        return {
            "funnel_analysis": funnel,
            "recommendations": [
                f"Focus on improving: {funnel['weakest_step']}"
            ] + funnel['issues']
        }
    except Exception as e:
        logger.error(f"Error analyzing funnel: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analytics/cac-ltv")
async def calculate_cac_ltv(request: CACLTVRequest):
    """
    Calculate Customer Acquisition Cost (CAC) and Lifetime Value (LTV)

    Returns unit economics analysis with health status
    """
    try:
        calc = CAC_LTV_Calculator()

        cac = calc.calculate_cac(
            request.total_marketing_spend,
            request.total_sales_spend,
            request.new_customers
        )

        ltv = calc.calculate_ltv(
            request.monthly_recurring_revenue,
            request.gross_margin,
            request.churn_rate_monthly
        )

        unit_econ = calc.calculate_unit_economics(cac, ltv)

        return {
            "unit_economics": unit_econ,
            "inputs": {
                "total_spend": request.total_marketing_spend + request.total_sales_spend,
                "new_customers": request.new_customers,
                "mrr": request.monthly_recurring_revenue,
                "churn_rate": request.churn_rate_monthly
            }
        }
    except Exception as e:
        logger.error(f"Error calculating CAC/LTV: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/strategy/gtm")
async def recommend_gtm_strategy(request: GTMStrategyRequest):
    """
    Recommend go-to-market strategy based on product characteristics

    Returns primary recommendation with alternatives and reasoning
    """
    try:
        strategy = GTMStrategySelector.recommend_gtm_motion(
            request.annual_contract_value,
            request.addressable_market_size,
            request.product_complexity_minutes,
            request.has_partner_ecosystem
        )

        return {
            "gtm_recommendation": strategy,
            "summary": f"Recommended motion: {strategy['primary_recommendation']['motion'].value} "
                      f"(confidence: {strategy['primary_recommendation']['confidence']})"
        }
    except Exception as e:
        logger.error(f"Error recommending GTM strategy: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/utm/generate")
async def generate_utm_url(request: UTMGenerateRequest):
    """
    Generate URL with UTM tracking parameters

    Returns fully-formed URL for marketing campaigns
    """
    try:
        tracker = UTMTracker()
        url = tracker.generate_utm_url(
            request.base_url,
            request.source,
            request.medium,
            request.campaign,
            request.term,
            request.content
        )

        return {
            "utm_url": url,
            "parameters": {
                "source": request.source,
                "medium": request.medium,
                "campaign": request.campaign,
                "term": request.term,
                "content": request.content
            }
        }
    except Exception as e:
        logger.error(f"Error generating UTM URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/utm/attribution")
async def analyze_attribution(customer_records: List[AttributionRecord]):
    """
    Analyze attribution data from customer records

    Returns performance by source/campaign with conversion metrics
    """
    try:
        tracker = UTMTracker()

        # Convert Pydantic models to dicts
        records = [record.dict() for record in customer_records]

        attribution = tracker.parse_attribution_data(records)

        # Format results
        formatted = []
        for key, data in attribution.items():
            formatted.append({
                "source_campaign": key,
                **data
            })

        return {
            "attribution_analysis": formatted,
            "total_customers": len(customer_records),
            "paying_customers": sum(1 for r in records if r.get('is_paying', False)),
            "total_mrr": sum(r.get('mrr', 0) for r in records if r.get('is_paying', False))
        }
    except Exception as e:
        logger.error(f"Error analyzing attribution: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Optional: Metrics endpoint (if prometheus-client installed)
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return JSONResponse(
            content=generate_latest().decode('utf-8'),
            media_type=CONTENT_TYPE_LATEST
        )
except ImportError:
    logger.info("Prometheus client not installed - /metrics endpoint disabled")


# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Module 13.3: Launch Preparation & Marketing API")
    logger.info("API documentation available at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
