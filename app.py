"""
Module 13.2: Governance & Compliance Documentation
FastAPI Application

REST API wrapper for compliance documentation functions.
No business logic here - only imports and calls to l3_m13_gov_compliance_docu.py
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import ComplianceConfig
from l3_m13_gov_compliance_docu import (
    create_sample_privacy_policy,
    create_sample_soc2_controls,
    create_sample_incident_playbook,
    run_evidence_collection,
    PrivacyPolicyGenerator,
    SOC2Documentation,
    IncidentResponsePlan,
    Subprocessor,
    DataRetention
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, ComplianceConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Module 13.2: Governance & Compliance Documentation API",
    description="REST API for compliance documentation generation and evidence collection",
    version="1.0.0"
)


# Request/Response Models
class GeneratePrivacyPolicyRequest(BaseModel):
    """Request model for privacy policy generation."""
    company_name: Optional[str] = Field(None, description="Company name (defaults to config)")
    contact_email: Optional[str] = Field(None, description="Privacy contact email (defaults to config)")


class GenerateSOC2Request(BaseModel):
    """Request model for SOC 2 documentation generation."""
    company_name: Optional[str] = Field(None, description="Company name (defaults to config)")


class IncidentRequest(BaseModel):
    """Request model for creating incident."""
    description: str = Field(..., description="Incident description")
    severity: Optional[str] = Field(None, description="Severity override (P0, P1, P2, P3)")


class IncidentUpdateRequest(BaseModel):
    """Request model for updating incident."""
    incident_id: str = Field(..., description="Incident identifier")
    status: str = Field(..., description="New status (responding, contained, resolved)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    config: Dict[str, Any]


class SkippedResponse(BaseModel):
    """Response when operation skipped due to missing dependencies."""
    skipped: bool
    reason: str


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status with configuration info
    """
    logger.info("Health check requested")

    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        config={
            "company_name": ComplianceConfig.COMPANY_NAME,
            "has_aws_credentials": ComplianceConfig.has_aws_credentials(),
            "has_database": ComplianceConfig.has_database(),
            "evidence_retention_days": ComplianceConfig.EVIDENCE_RETENTION_DAYS
        }
    )


@app.post("/generate/privacy-policy")
async def generate_privacy_policy(request: GeneratePrivacyPolicyRequest = None):
    """
    Generate GDPR-compliant privacy policy.

    Args:
        request: Optional customization parameters

    Returns:
        Privacy policy markdown text
    """
    logger.info("Privacy policy generation requested")

    try:
        company_name = request.company_name if request and request.company_name else ComplianceConfig.COMPANY_NAME
        contact_email = request.contact_email if request and request.contact_email else ComplianceConfig.COMPANY_EMAIL

        generator = PrivacyPolicyGenerator(company_name, contact_email)

        # Use example subprocessors and retentions
        subprocessors = [
            Subprocessor("AWS", "Infrastructure", "All data", "US-East-1", True, "2025-01-15"),
            Subprocessor("Pinecone", "Vector storage", "Embeddings", "US-East-1", True, "2025-01-15"),
            Subprocessor("OpenAI", "Embeddings/LLM", "Text (no PII)", "US", True, "2025-01-15"),
        ]

        retentions = [
            DataRetention("Account data", "Account lifetime + 7 years", "Tax/legal"),
            DataRetention("User content", "Account lifetime + 30 days", "Service + deletion buffer"),
            DataRetention("Audit logs", "2 years", "Security/compliance"),
        ]

        policy = generator.generate_privacy_policy(subprocessors, retentions)

        logger.info("Privacy policy generated successfully")
        return JSONResponse(content={
            "success": True,
            "policy": policy,
            "length": len(policy),
            "company_name": company_name
        })

    except Exception as e:
        logger.error(f"Error generating privacy policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/soc2-controls")
async def generate_soc2_controls(request: GenerateSOC2Request = None):
    """
    Generate SOC 2 controls documentation.

    Args:
        request: Optional customization parameters

    Returns:
        SOC 2 documentation markdown text
    """
    logger.info("SOC 2 controls generation requested")

    try:
        doc = create_sample_soc2_controls()

        logger.info("SOC 2 controls generated successfully")
        return JSONResponse(content={
            "success": True,
            "documentation": doc,
            "length": len(doc)
        })

    except Exception as e:
        logger.error(f"Error generating SOC 2 controls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/incident-playbook")
async def generate_incident_playbook():
    """
    Generate incident response playbook.

    Returns:
        Incident response playbook markdown text
    """
    logger.info("Incident response playbook generation requested")

    try:
        playbook = create_sample_incident_playbook()

        logger.info("Incident response playbook generated successfully")
        return JSONResponse(content={
            "success": True,
            "playbook": playbook,
            "length": len(playbook)
        })

    except Exception as e:
        logger.error(f"Error generating incident playbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/incident/create")
async def create_incident(request: IncidentRequest):
    """
    Create new security incident record.

    Args:
        request: Incident details

    Returns:
        Incident record with ID and classification
    """
    logger.info(f"Incident creation requested: {request.description}")

    try:
        ir_plan = IncidentResponsePlan(ComplianceConfig.COMPANY_NAME)
        incident = ir_plan.create_incident(request.description, request.severity)

        logger.info(f"Incident created: {incident.incident_id}")
        return JSONResponse(content={
            "success": True,
            "incident_id": incident.incident_id,
            "severity": incident.severity,
            "description": incident.description,
            "detection_time": incident.detection_time,
            "status": incident.status
        })

    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/incident/update")
async def update_incident(request: IncidentUpdateRequest):
    """
    Update incident status.

    Args:
        request: Incident ID and new status

    Returns:
        Success status
    """
    logger.info(f"Incident update requested: {request.incident_id} -> {request.status}")

    try:
        ir_plan = IncidentResponsePlan(ComplianceConfig.COMPANY_NAME)
        # Note: In production, would load existing incidents
        success = ir_plan.update_incident_status(request.incident_id, request.status)

        if success:
            logger.info(f"Incident updated: {request.incident_id}")
            return JSONResponse(content={
                "success": True,
                "incident_id": request.incident_id,
                "status": request.status
            })
        else:
            raise HTTPException(status_code=404, detail="Incident not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidence/collect")
async def collect_evidence(period: Optional[str] = Query(None, description="Evidence period (YYYY-MM)")):
    """
    Collect compliance evidence for the specified period.

    Args:
        period: Evidence period (defaults to current month)

    Returns:
        Evidence summary or skipped response if AWS not configured
    """
    logger.info(f"Evidence collection requested for period: {period or 'current'}")

    # Check if AWS credentials available
    if not ComplianceConfig.has_aws_credentials():
        logger.warning("Evidence collection skipped: No AWS credentials")
        return JSONResponse(
            status_code=200,
            content={
                "skipped": True,
                "reason": "AWS credentials not configured. Using mock evidence data."
            }
        )

    try:
        evidence = run_evidence_collection()

        logger.info(f"Evidence collection complete: {len(evidence)} types")
        return JSONResponse(content={
            "success": True,
            "evidence_types": list(evidence.keys()),
            "total_types": len(evidence),
            "period": period or datetime.now().strftime("%Y-%m"),
            "collection_time": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error collecting evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/status")
async def compliance_status():
    """
    Get overall compliance status.

    Returns:
        Compliance readiness summary
    """
    logger.info("Compliance status requested")

    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "company": ComplianceConfig.COMPANY_NAME,
            "frameworks": {
                "gdpr": {
                    "requirements": 88,
                    "status": "self_assessment_required"
                },
                "soc2": {
                    "controls": 64,
                    "status": "documentation_ready",
                    "last_audit": ComplianceConfig.SOC2_LAST_AUDIT_DATE
                },
                "hipaa": {
                    "safeguards": 164,
                    "status": "not_applicable"
                }
            },
            "last_penetration_test": ComplianceConfig.PENETRATION_TEST_DATE,
            "evidence_retention_days": ComplianceConfig.EVIDENCE_RETENTION_DAYS,
            "aws_configured": ComplianceConfig.has_aws_credentials()
        }

        logger.info("Compliance status retrieved")
        return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"Error getting compliance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Prometheus metrics endpoint
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    # Metrics
    api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
    api_errors = Counter('api_errors_total', 'Total API errors', ['endpoint'])

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("Prometheus metrics enabled")

except ImportError:
    logger.info("Prometheus metrics not available (prometheus-client not installed)")


# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting FastAPI server on {ComplianceConfig.API_HOST}:{ComplianceConfig.API_PORT}")

    uvicorn.run(
        "app:app",
        host=ComplianceConfig.API_HOST,
        port=ComplianceConfig.API_PORT,
        reload=True,
        log_level=ComplianceConfig.LOG_LEVEL.lower()
    )
