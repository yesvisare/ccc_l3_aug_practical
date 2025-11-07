"""
Module 13.2: Governance & Compliance Documentation
Smoke Tests

Minimal tests to verify basic functionality.
Tests gracefully skip network calls without keys.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from config import ComplianceConfig, EVIDENCE_TYPES, SOC2_CONTROLS
from l3_m13_gov_compliance_docu import (
    PrivacyPolicyGenerator,
    SOC2Documentation,
    IncidentResponsePlan,
    EvidenceCollector,
    Subprocessor,
    DataRetention,
    SOC2Control,
    create_sample_privacy_policy,
    create_sample_soc2_controls,
    create_sample_incident_playbook
)


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        """Test that config loads without errors."""
        assert ComplianceConfig.COMPANY_NAME is not None
        assert ComplianceConfig.COMPANY_EMAIL is not None
        assert isinstance(ComplianceConfig.EVIDENCE_RETENTION_DAYS, int)

    def test_aws_check(self):
        """Test AWS credentials check."""
        result = ComplianceConfig.has_aws_credentials()
        assert isinstance(result, bool)

    def test_constants(self):
        """Test that constants are defined."""
        assert SOC2_CONTROLS == 64
        assert len(EVIDENCE_TYPES) > 0


class TestPrivacyPolicyGenerator:
    """Test privacy policy generation."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = PrivacyPolicyGenerator("Test Company", "test@test.com")
        assert generator.company_name == "Test Company"
        assert generator.contact_email == "test@test.com"

    def test_subprocessor_table(self):
        """Test subprocessor table generation."""
        generator = PrivacyPolicyGenerator("Test", "test@test.com")
        subprocessors = [
            Subprocessor("AWS", "Hosting", "All", "US", True, "2025-01-01"),
            Subprocessor("OpenAI", "AI", "Text", "US", True, "2025-01-01")
        ]

        table = generator.generate_subprocessor_table(subprocessors)

        assert "AWS" in table
        assert "OpenAI" in table
        assert "✅" in table
        assert "2025-01-01" in table

    def test_retention_table(self):
        """Test retention table generation."""
        generator = PrivacyPolicyGenerator("Test", "test@test.com")
        retentions = [
            DataRetention("Account data", "7 years", "Legal"),
            DataRetention("Logs", "2 years", "Compliance")
        ]

        table = generator.generate_retention_table(retentions)

        assert "Account data" in table
        assert "7 years" in table
        assert "Legal" in table

    def test_complete_policy(self):
        """Test complete privacy policy generation."""
        generator = PrivacyPolicyGenerator("Test Company", "privacy@test.com")
        subprocessors = [
            Subprocessor("AWS", "Hosting", "All", "US", True, "2025-01-01")
        ]
        retentions = [
            DataRetention("Account data", "7 years", "Legal")
        ]

        policy = generator.generate_privacy_policy(subprocessors, retentions)

        assert "Privacy Policy" in policy
        assert "Test Company" in policy
        assert "GDPR" in policy
        assert "AWS" in policy
        assert len(policy) > 1000  # Reasonable length check


class TestSOC2Documentation:
    """Test SOC 2 documentation generation."""

    def test_initialization(self):
        """Test SOC2Documentation initialization."""
        doc = SOC2Documentation("Test Company")
        assert doc.company_name == "Test Company"
        assert len(doc.controls) == 0

    def test_add_control(self):
        """Test adding controls."""
        doc = SOC2Documentation("Test")
        control = SOC2Control(
            control_id="CC5.1",
            control_name="Access Control",
            description="Test control",
            implementation="RBAC",
            evidence_location="/evidence",
            responsible_party="CTO"
        )

        doc.add_control(control)
        assert len(doc.controls) == 1
        assert doc.controls[0].control_id == "CC5.1"

    def test_control_summary(self):
        """Test control summary generation."""
        doc = SOC2Documentation("Test")
        doc.add_control(SOC2Control(
            "CC1", "Test1", "Desc", "Impl", "Loc", "Person", "implemented"
        ))
        doc.add_control(SOC2Control(
            "CC2", "Test2", "Desc", "Impl", "Loc", "Person", "planned"
        ))

        summary = doc.get_control_summary()

        assert summary["implemented"] == 1
        assert summary["planned"] == 1

    def test_documentation_generation(self):
        """Test complete documentation generation."""
        doc = SOC2Documentation("Test Company")
        doc.add_control(SOC2Control(
            "CC5.1", "Access", "Test", "RBAC", "/evidence", "CTO"
        ))

        documentation = doc.generate_soc2_documentation()

        assert "SOC 2 Controls" in documentation
        assert "Test Company" in documentation
        assert "CC5.1" in documentation


class TestIncidentResponsePlan:
    """Test incident response functionality."""

    def test_initialization(self):
        """Test IncidentResponsePlan initialization."""
        plan = IncidentResponsePlan("Test Company")
        assert plan.company_name == "Test Company"
        assert len(plan.incidents) == 0

    def test_classify_incident(self):
        """Test incident severity classification."""
        plan = IncidentResponsePlan("Test")

        assert plan.classify_incident("data breach detected") == "P0"
        assert plan.classify_incident("potential breach") == "P1"
        assert plan.classify_incident("suspicious activity") == "P2"
        assert plan.classify_incident("normal event") == "P3"

    def test_create_incident(self):
        """Test incident creation."""
        plan = IncidentResponsePlan("Test")
        incident = plan.create_incident("Test incident", "P2")

        assert incident.severity == "P2"
        assert incident.description == "Test incident"
        assert incident.status == "detected"
        assert incident.incident_id.startswith("INC-")

    def test_update_incident_status(self):
        """Test incident status updates."""
        plan = IncidentResponsePlan("Test")
        incident = plan.create_incident("Test", "P2")

        success = plan.update_incident_status(incident.incident_id, "responding")
        assert success is True
        assert incident.status == "responding"
        assert incident.response_time is not None

    def test_incident_metrics(self):
        """Test incident metrics calculation."""
        plan = IncidentResponsePlan("Test")
        plan.create_incident("Test 1", "P0")
        plan.create_incident("Test 2", "P1")

        metrics = plan.get_incident_metrics()

        assert metrics["total"] == 2
        assert metrics["by_severity"]["P0"] == 1
        assert metrics["by_severity"]["P1"] == 1

    def test_playbook_generation(self):
        """Test playbook generation."""
        plan = IncidentResponsePlan("Test Company")
        playbook = plan.generate_playbook()

        assert "Incident Response Playbook" in playbook
        assert "Test Company" in playbook
        assert "P0" in playbook
        assert "Detection" in playbook


class TestEvidenceCollector:
    """Test evidence collection."""

    def test_initialization(self):
        """Test EvidenceCollector initialization."""
        collector = EvidenceCollector("test_evidence")
        assert collector.evidence_dir == Path("test_evidence")
        assert collector.evidence_dir.exists()

    def test_collect_user_access_matrix(self):
        """Test user access matrix collection."""
        collector = EvidenceCollector()
        evidence = collector.collect_user_access_matrix()

        assert "collection_date" in evidence
        assert "total_users" in evidence
        assert "by_role" in evidence
        assert isinstance(evidence["total_users"], int)

    def test_collect_backup_evidence(self):
        """Test backup evidence collection."""
        collector = EvidenceCollector()
        evidence = collector.collect_backup_evidence()

        assert "collection_date" in evidence
        assert "backup_frequency" in evidence
        assert "last_backup_date" in evidence
        assert isinstance(evidence["backup_success_rate"], float)

    def test_collect_vulnerability_scan(self):
        """Test vulnerability scan collection."""
        collector = EvidenceCollector()
        evidence = collector.collect_vulnerability_scan()

        assert "collection_date" in evidence
        assert "vulnerabilities" in evidence
        assert "critical" in evidence["vulnerabilities"]

    def test_collect_all_evidence(self):
        """Test collecting all evidence types."""
        collector = EvidenceCollector("test_evidence")
        evidence = collector.collect_all_evidence("2025-11")

        assert len(evidence) >= 3  # At least 3 types
        assert "user_access_matrix" in evidence
        assert "backup_logs" in evidence

        # Verify files were created
        period_dir = Path("test_evidence") / "2025-11"
        assert period_dir.exists()
        assert (period_dir / "evidence_summary.json").exists()


class TestSampleFunctions:
    """Test sample generation functions."""

    def test_sample_privacy_policy(self):
        """Test sample privacy policy creation."""
        policy = create_sample_privacy_policy()

        assert len(policy) > 1000
        assert "Privacy Policy" in policy
        assert "GDPR" in policy

    def test_sample_soc2_controls(self):
        """Test sample SOC 2 controls creation."""
        doc = create_sample_soc2_controls()

        assert len(doc) > 500
        assert "SOC 2" in doc
        assert "CC5.1" in doc

    def test_sample_incident_playbook(self):
        """Test sample incident playbook creation."""
        playbook = create_sample_incident_playbook()

        assert len(playbook) > 500
        assert "Incident Response" in playbook
        assert "P0" in playbook


class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "config" in data

    def test_generate_privacy_policy_endpoint(self, client):
        """Test privacy policy generation endpoint."""
        response = client.post("/generate/privacy-policy")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "policy" in data
        assert len(data["policy"]) > 1000

    def test_generate_soc2_endpoint(self, client):
        """Test SOC 2 controls generation endpoint."""
        response = client.post("/generate/soc2-controls")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "documentation" in data

    def test_generate_incident_playbook_endpoint(self, client):
        """Test incident playbook generation endpoint."""
        response = client.post("/generate/incident-playbook")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "playbook" in data

    def test_create_incident_endpoint(self, client):
        """Test incident creation endpoint."""
        response = client.post("/incident/create", json={
            "description": "Test security incident",
            "severity": "P2"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "incident_id" in data
        assert data["severity"] == "P2"

    def test_compliance_status_endpoint(self, client):
        """Test compliance status endpoint."""
        response = client.get("/compliance/status")
        assert response.status_code == 200
        data = response.json()
        assert "frameworks" in data
        assert "gdpr" in data["frameworks"]
        assert "soc2" in data["frameworks"]

    def test_evidence_collection_skips_gracefully(self, client):
        """Test evidence collection skips without AWS credentials."""
        response = client.post("/evidence/collect")
        assert response.status_code == 200
        # Should return either success or skipped
        data = response.json()
        assert "success" in data or "skipped" in data


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
