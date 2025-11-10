"""
FastAPI Application for Module 13.4: Portfolio Showcase & Career Launch

Provides REST API endpoints for portfolio generation and career decision tools.
No external API keys required - all functionality runs locally.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from datetime import datetime

# Import core module
from src.l3_m13_portfolio_career import (
    SystemMetrics, TechDecision, ApplicationMetrics,
    generate_architecture_doc, create_demo_script, generate_case_study,
    prepare_interview_responses, evaluate_portfolio_decision,
    track_application_metrics, passes_30_second_test
)
import src.l3_m13_portfolio_career as portfolio
from config import PortfolioConfig, OutputPaths, validate_config

# Optional: Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    METRICS_AVAILABLE = True

    # Define metrics
    request_counter = Counter('portfolio_requests_total', 'Total requests', ['endpoint'])
    request_duration = Histogram('portfolio_request_duration_seconds', 'Request duration', ['endpoint'])
except ImportError:
    METRICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Portfolio Showcase & Career Launch API",
    description="Module 13.4: Tools for creating technical portfolios and career planning",
    version="1.0.0"
)


# Request/Response Models
class SystemMetricsRequest(BaseModel):
    tenants: int = Field(100, ge=1, description="Number of tenants")
    queries_per_day: int = Field(50000, ge=0, description="Daily query volume")
    p95_latency_seconds: float = Field(1.8, ge=0, description="P95 latency in seconds")
    uptime_percentage: float = Field(99.5, ge=0, le=100, description="System uptime percentage")
    time_reduction_percentage: int = Field(87, ge=0, le=100, description="Time reduction achieved")
    arr_usd: int = Field(50000, ge=0, description="Annual Recurring Revenue in USD")
    user_satisfaction: float = Field(4.8, ge=0, le=5, description="User satisfaction score")
    cost_per_query_usd: float = Field(0.008, ge=0, description="Cost per query in USD")
    cost_savings_percentage: int = Field(60, ge=0, le=100, description="Cost savings percentage")


class TechDecisionRequest(BaseModel):
    choice: str = Field(..., description="Technology choice made")
    alternatives_considered: List[str] = Field(..., description="Alternative options evaluated")
    rationale: str = Field(..., description="Reasoning for choice")
    cost_monthly_usd: float = Field(..., ge=0, description="Monthly cost in USD")
    when_to_choose_alternative: str = Field(..., description="When to choose alternative")


class PortfolioDecisionRequest(BaseModel):
    target_companies: int = Field(..., ge=1, description="Number of companies to apply to")
    current_callback_rate: float = Field(..., ge=0, le=100, description="Current callback rate percentage")
    job_search_timeline_months: int = Field(..., ge=1, description="Job search timeline in months")
    referral_strength: str = Field(..., description="Referral strength: strong, weak, or none")
    target_level: str = Field(..., description="Target level: senior, staff, or principal")


class ApplicationMetricsRequest(BaseModel):
    applications_sent: int = Field(0, ge=0, description="Number of applications sent")
    callbacks_received: int = Field(0, ge=0, description="Number of callbacks received")
    interviews_completed: int = Field(0, ge=0, description="Number of interviews completed")
    offers_received: int = Field(0, ge=0, description="Number of offers received")


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if METRICS_AVAILABLE:
        request_counter.labels(endpoint='health').inc()

    config_valid = validate_config()

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "module": "13.4 Portfolio Showcase & Career Launch",
        "config_valid": config_valid,
        "metrics_enabled": METRICS_AVAILABLE
    }


# Architecture Documentation Generation
@app.post("/generate/architecture")
async def generate_architecture(
    system_name: str = Field(..., description="System name"),
    metrics: SystemMetricsRequest = Field(..., description="System metrics"),
    tech_stack: List[str] = Field(..., description="Technology stack"),
    decisions: List[TechDecisionRequest] = Field(..., description="Technical decisions")
):
    """
    Generate comprehensive architecture documentation.

    Returns structured documentation with overview, architecture, decisions, and alternatives.
    """
    if METRICS_AVAILABLE:
        request_counter.labels(endpoint='generate_architecture').inc()
        with request_duration.labels(endpoint='generate_architecture').time():
            return await _generate_architecture_internal(system_name, metrics, tech_stack, decisions)
    else:
        return await _generate_architecture_internal(system_name, metrics, tech_stack, decisions)


async def _generate_architecture_internal(system_name, metrics, tech_stack, decisions):
    try:
        logger.info(f"Generating architecture doc for {system_name}")

        # Convert request models to domain models
        sys_metrics = portfolio.SystemMetrics(**metrics.dict())
        tech_decisions = [portfolio.TechDecision(**d.dict()) for d in decisions]

        # Generate documentation
        arch_doc = portfolio.generate_architecture_doc(
            system_name=system_name,
            metrics=sys_metrics,
            tech_stack=tech_stack,
            decisions=tech_decisions
        )

        return {
            "status": "success",
            "documentation": arch_doc,
            "word_count": sum(len(section.split()) for section in arch_doc.values())
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating architecture doc: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Portfolio Decision Evaluation
@app.post("/evaluate/portfolio-decision")
async def evaluate_portfolio_decision(request: PortfolioDecisionRequest):
    """
    Evaluate whether to invest in portfolio creation.

    Returns recommendation, reasoning, time investment, and expected callback rate.
    """
    if METRICS_AVAILABLE:
        request_counter.labels(endpoint='evaluate_portfolio').inc()

    try:
        logger.info(f"Evaluating portfolio decision for {request.target_companies} companies")

        decision = portfolio.evaluate_portfolio_decision(
            target_companies=request.target_companies,
            current_callback_rate=request.current_callback_rate,
            job_search_timeline_months=request.job_search_timeline_months,
            referral_strength=request.referral_strength,
            target_level=request.target_level
        )

        return {
            "status": "success",
            "decision": decision
        }

    except Exception as e:
        logger.error(f"Error evaluating portfolio decision: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Application Metrics Tracking
@app.post("/track/application-metrics")
async def track_application_metrics(request: ApplicationMetricsRequest):
    """
    Track job application performance and get alerts/recommendations.

    Returns callback rate, offer rate, alerts, and recommendations.
    """
    if METRICS_AVAILABLE:
        request_counter.labels(endpoint='track_metrics').inc()

    try:
        logger.info(f"Tracking metrics for {request.applications_sent} applications")

        app_metrics = portfolio.ApplicationMetrics(**request.dict())
        tracking = portfolio.track_application_metrics(app_metrics)

        return {
            "status": "success",
            "metrics": tracking
        }

    except Exception as e:
        logger.error(f"Error tracking application metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# README Validation
@app.post("/validate/readme")
async def validate_readme(readme_content: str = Field(..., description="First 100 words of README")):
    """
    Validate if README passes the 30-second test.

    Checks for problem statement, impact metric, and scale indicator.
    """
    if METRICS_AVAILABLE:
        request_counter.labels(endpoint='validate_readme').inc()

    try:
        passes, missing = portfolio.passes_30_second_test(readme_content)

        return {
            "status": "success",
            "passes_test": passes,
            "missing_elements": missing,
            "recommendation": "README passes 30-second test" if passes else f"Add: {', '.join(missing)}"
        }

    except Exception as e:
        logger.error(f"Error validating README: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Configuration Info
@app.get("/info/config")
async def get_config_info():
    """Get portfolio configuration constants"""
    return {
        "time_investment": {
            "architecture_doc_hours": PortfolioConfig.ARCHITECTURE_DOC_HOURS,
            "demo_video_hours": PortfolioConfig.DEMO_VIDEO_HOURS,
            "case_study_hours": PortfolioConfig.CASE_STUDY_HOURS,
            "interview_prep_hours": PortfolioConfig.INTERVIEW_PREP_HOURS,
            "total_minimum_hours": PortfolioConfig.TOTAL_MINIMUM_HOURS,
            "total_recommended_hours": PortfolioConfig.TOTAL_RECOMMENDED_HOURS
        },
        "target_metrics": {
            "callback_rate_percent": PortfolioConfig.TARGET_CALLBACK_RATE,
            "offer_rate_percent": PortfolioConfig.TARGET_OFFER_RATE
        },
        "application_strategies": {
            "high_touch": f"{PortfolioConfig.APPLICATIONS_HIGH_TOUCH} applications (30-40% callback)",
            "medium_touch": f"{PortfolioConfig.APPLICATIONS_MEDIUM_TOUCH} applications (15-20% callback)",
            "low_touch": f"{PortfolioConfig.APPLICATIONS_LOW_TOUCH}+ applications (5-10% callback)"
        },
        "monthly_cost_inr": {
            "with_portfolio": f"{PortfolioConfig.COST_WITH_PORTFOLIO_MIN}-{PortfolioConfig.COST_WITH_PORTFOLIO_MAX}",
            "without_portfolio": f"{PortfolioConfig.COST_WITHOUT_PORTFOLIO_MIN}-{PortfolioConfig.COST_WITHOUT_PORTFOLIO_MAX}"
        }
    }


# Prometheus Metrics Endpoint
if METRICS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "Endpoint does not exist"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Portfolio Showcase & Career Launch API")
    logger.info(f"Metrics enabled: {METRICS_AVAILABLE}")

    # Validate configuration
    if not validate_config():
        logger.warning("Configuration validation failed - check OutputPaths")


# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server on http://127.0.0.1:8000")
    logger.info("API docs: http://127.0.0.1:8000/docs")

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
