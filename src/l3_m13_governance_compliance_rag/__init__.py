"""
Module 13.2: Governance & Compliance Documentation
Main implementation module

This module provides functionality for creating and managing compliance documentation
including privacy policies, SOC 2 controls, incident response plans, and evidence collection.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from config import ComplianceConfig, EVIDENCE_TYPES, INCIDENT_SEVERITIES, SOC2_CONTROLS

# Configure logging
logging.basicConfig(
    level=getattr(logging, ComplianceConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Subprocessor:
    """Represents a data subprocessor for GDPR compliance."""
    name: str
    purpose: str
    data_shared: str
    location: str
    dpa_status: bool
    added_date: str


@dataclass
class DataRetention:
    """Data retention policy for GDPR compliance."""
    data_type: str
    retention_period: str
    reason: str


@dataclass
class SOC2Control:
    """SOC 2 control documentation."""
    control_id: str
    control_name: str
    description: str
    implementation: str
    evidence_location: str
    responsible_party: str
    status: str = "implemented"  # implemented, planned, not_applicable


@dataclass
class IncidentResponse:
    """Incident response record."""
    incident_id: str
    severity: str  # P0, P1, P2, P3
    description: str
    detection_time: str
    response_time: Optional[str] = None
    containment_time: Optional[str] = None
    resolution_time: Optional[str] = None
    status: str = "detected"  # detected, responding, contained, resolved


@dataclass
class ComplianceEvidence:
    """Compliance evidence record."""
    evidence_type: str
    collection_date: str
    period: str  # e.g., "2025-11"
    data: Dict[str, Any]
    status: str = "collected"


class PrivacyPolicyGenerator:
    """Generate GDPR-compliant privacy policies."""

    def __init__(self, company_name: str, contact_email: str):
        """
        Initialize privacy policy generator.

        Args:
            company_name: Name of the company
            contact_email: Privacy contact email
        """
        self.company_name = company_name
        self.contact_email = contact_email
        logger.info(f"Initialized PrivacyPolicyGenerator for {company_name}")

    def generate_subprocessor_table(self, subprocessors: List[Subprocessor]) -> str:
        """
        Generate markdown table of subprocessors.

        Args:
            subprocessors: List of subprocessor records

        Returns:
            Markdown formatted table
        """
        logger.info(f"Generating subprocessor table with {len(subprocessors)} entries")

        header = "| Subprocessor | Purpose | Data Shared | Location | DPA | Added |\n"
        header += "|--------------|---------|-------------|----------|-----|-------|\n"

        rows = []
        for sp in subprocessors:
            dpa = "✅" if sp.dpa_status else "❌"
            row = f"| {sp.name} | {sp.purpose} | {sp.data_shared} | {sp.location} | {dpa} | {sp.added_date} |"
            rows.append(row)

        return header + "\n".join(rows)

    def generate_retention_table(self, retentions: List[DataRetention]) -> str:
        """
        Generate markdown table of data retention periods.

        Args:
            retentions: List of data retention policies

        Returns:
            Markdown formatted table
        """
        logger.info(f"Generating retention table with {len(retentions)} policies")

        header = "| Data Type | Retention | Reason |\n"
        header += "|-----------|-----------|--------|\n"

        rows = []
        for ret in retentions:
            row = f"| {ret.data_type} | {ret.retention_period} | {ret.reason} |"
            rows.append(row)

        return header + "\n".join(rows)

    def generate_privacy_policy(
        self,
        subprocessors: List[Subprocessor],
        retentions: List[DataRetention],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate complete GDPR-compliant privacy policy.

        Args:
            subprocessors: List of data subprocessors
            retentions: List of data retention policies
            output_path: Optional path to save policy

        Returns:
            Privacy policy markdown text
        """
        logger.info("Generating complete privacy policy")

        subprocessor_table = self.generate_subprocessor_table(subprocessors)
        retention_table = self.generate_retention_table(retentions)

        policy = f"""# Privacy Policy - {self.company_name}

**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}

## 1. Introduction

This Privacy Policy describes how {self.company_name} collects, uses, and protects your personal data in compliance with GDPR, CCPA, and other applicable regulations.

## 2. Data We Collect

### Account Information
- **Data:** Email address, name, company name
- **Legal Basis:** Contract performance (GDPR Art. 6(1)(b))
- **Retention:** Account lifetime + 7 years

### User Content
- **Data:** Documents uploaded for processing
- **Legal Basis:** Contract performance
- **Retention:** Account lifetime + 30 days

### Usage Data
- **Data:** Query counts, latency metrics (anonymized)
- **Legal Basis:** Legitimate interest (GDPR Art. 6(1)(f))
- **Retention:** 2 years

## 3. How We Use Your Data

### Service Delivery (Required)
Your documents are processed through our RAG pipeline:
- **Embedding generation:** Text sent to OpenAI API (zero-retention DPA)
- **Vector storage:** Embeddings in Pinecone (isolated tenant namespace)
- **Query processing:** Matched against YOUR embeddings ONLY

### Data Isolation
Your documents NEVER:
- Shared with other customers
- Used to answer other customers' queries
- Visible to other tenants

## 4. Data Sharing & Subprocessors

{subprocessor_table}

**Subprocessor Changes:** 30 days notice before adding new subprocessors.

## 5. Data Retention

{retention_table}

## 6. Your GDPR Rights

You have the right to:
- **Access:** Request copy of your data (within 30 days)
- **Rectification:** Correct inaccurate data (within 30 days)
- **Erasure:** Delete your data (within 30 days)
- **Portability:** Export your data in machine-readable format
- **Restriction:** Limit processing of your data
- **Objection:** Object to certain processing activities
- **Withdraw Consent:** Where processing based on consent
- **Lodge Complaint:** With supervisory authority

**How to Exercise Rights:** Email {self.contact_email}

## 7. Data Security Measures

- **Encryption:** TLS 1.3 (transit), AES-256 (at rest)
- **Access Control:** Role-Based Access Control (RBAC) via Casbin
- **Authentication:** Email/password + optional MFA
- **Audit Logging:** All access logged for 2 years
- **PII Detection:** Automated redaction via Presidio

## 8. International Transfers

- **Primary Location:** {ComplianceConfig.AWS_REGION}
- **EU Option:** Available (eu-west-1)
- **Transfer Mechanism:** Standard Contractual Clauses (SCCs)

## 9. Data Breach Notification

In the event of a data breach:
- **Notification to Authorities:** Within 72 hours (GDPR Art. 33)
- **Notification to You:** Within 24 hours if high risk
- **Response:** Follow incident response plan

## 10. Children's Privacy

We do not knowingly collect data from children under 16.

## 11. Changes to This Policy

We will notify you 30 days before material changes via email.

## 12. Contact

**Privacy Contact:** {self.contact_email}
**Company:** {self.company_name}
**Last Reviewed:** {datetime.now().strftime('%Y-%m-%d')}
"""

        if output_path:
            Path(output_path).write_text(policy)
            logger.info(f"Privacy policy saved to {output_path}")

        return policy


class SOC2Documentation:
    """Generate SOC 2 controls documentation."""

    def __init__(self, company_name: str):
        """
        Initialize SOC 2 documentation generator.

        Args:
            company_name: Name of the company
        """
        self.company_name = company_name
        self.controls: List[SOC2Control] = []
        logger.info(f"Initialized SOC2Documentation for {company_name}")

    def add_control(self, control: SOC2Control) -> None:
        """
        Add a SOC 2 control to documentation.

        Args:
            control: SOC2Control instance
        """
        self.controls.append(control)
        logger.info(f"Added control {control.control_id}: {control.control_name}")

    def get_control_summary(self) -> Dict[str, int]:
        """
        Get summary of control implementation status.

        Returns:
            Dictionary with counts by status
        """
        summary = {"implemented": 0, "planned": 0, "not_applicable": 0}
        for control in self.controls:
            summary[control.status] = summary.get(control.status, 0) + 1

        logger.info(f"Control summary: {summary}")
        return summary

    def generate_soc2_documentation(self, output_path: Optional[str] = None) -> str:
        """
        Generate complete SOC 2 controls documentation.

        Args:
            output_path: Optional path to save documentation

        Returns:
            SOC 2 documentation markdown text
        """
        logger.info(f"Generating SOC 2 documentation with {len(self.controls)} controls")

        summary = self.get_control_summary()

        doc = f"""# SOC 2 Controls Documentation - {self.company_name}

**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
**Total Controls:** {SOC2_CONTROLS}
**Implemented:** {summary['implemented']}
**Planned:** {summary['planned']}
**Not Applicable:** {summary['not_applicable']}

## Overview

This document maps our security controls to SOC 2 Trust Service Criteria requirements.

## Controls

"""

        for control in self.controls:
            doc += f"""### {control.control_id} - {control.control_name}

**Description:** {control.description}

**Implementation:**
{control.implementation}

**Evidence Location:** {control.evidence_location}

**Responsible Party:** {control.responsible_party}

**Status:** {control.status.upper()}

---

"""

        if output_path:
            Path(output_path).write_text(doc)
            logger.info(f"SOC 2 documentation saved to {output_path}")

        return doc


class IncidentResponsePlan:
    """Manage incident response planning and execution."""

    def __init__(self, company_name: str):
        """
        Initialize incident response manager.

        Args:
            company_name: Name of the company
        """
        self.company_name = company_name
        self.incidents: List[IncidentResponse] = []
        logger.info(f"Initialized IncidentResponsePlan for {company_name}")

    def classify_incident(self, description: str) -> str:
        """
        Classify incident severity based on description.

        Args:
            description: Incident description

        Returns:
            Severity level (P0, P1, P2, P3)
        """
        desc_lower = description.lower()

        # P0: Critical
        if any(word in desc_lower for word in ["breach", "ransomware", "complete outage", "data leak"]):
            return "P0"

        # P1: High
        if any(word in desc_lower for word in ["potential breach", "failed attack", "critical vulnerability"]):
            return "P1"

        # P2: Medium
        if any(word in desc_lower for word in ["security event", "investigation", "suspicious"]):
            return "P2"

        # P3: Low
        return "P3"

    def create_incident(self, description: str, severity: Optional[str] = None) -> IncidentResponse:
        """
        Create new incident record.

        Args:
            description: Incident description
            severity: Optional severity override

        Returns:
            IncidentResponse instance
        """
        if severity is None:
            severity = self.classify_incident(description)

        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        detection_time = datetime.now().isoformat()

        incident = IncidentResponse(
            incident_id=incident_id,
            severity=severity,
            description=description,
            detection_time=detection_time
        )

        self.incidents.append(incident)
        logger.error(f"Incident created: {incident_id} [{severity}] {description}")

        return incident

    def update_incident_status(
        self,
        incident_id: str,
        status: str,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Update incident status and timestamps.

        Args:
            incident_id: Incident identifier
            status: New status (responding, contained, resolved)
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if updated, False if incident not found
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.status = status

                if status == "responding" and not incident.response_time:
                    incident.response_time = timestamp
                elif status == "contained" and not incident.containment_time:
                    incident.containment_time = timestamp
                elif status == "resolved" and not incident.resolution_time:
                    incident.resolution_time = timestamp

                logger.info(f"Incident {incident_id} updated to status: {status}")
                return True

        logger.warning(f"Incident {incident_id} not found")
        return False

    def get_incident_metrics(self) -> Dict[str, Any]:
        """
        Calculate incident response metrics.

        Returns:
            Dictionary with metrics (MTTD, MTTC, counts)
        """
        if not self.incidents:
            return {"total": 0}

        response_times = []
        containment_times = []

        for incident in self.incidents:
            if incident.response_time:
                detection = datetime.fromisoformat(incident.detection_time)
                response = datetime.fromisoformat(incident.response_time)
                response_times.append((response - detection).total_seconds() / 60)

            if incident.containment_time:
                detection = datetime.fromisoformat(incident.detection_time)
                containment = datetime.fromisoformat(incident.containment_time)
                containment_times.append((containment - detection).total_seconds() / 60)

        metrics = {
            "total": len(self.incidents),
            "by_severity": {sev: sum(1 for i in self.incidents if i.severity == sev) for sev in ["P0", "P1", "P2", "P3"]},
            "by_status": {status: sum(1 for i in self.incidents if i.status == status) for status in ["detected", "responding", "contained", "resolved"]},
        }

        if response_times:
            metrics["mean_time_to_respond_minutes"] = sum(response_times) / len(response_times)

        if containment_times:
            metrics["mean_time_to_contain_minutes"] = sum(containment_times) / len(containment_times)

        logger.info(f"Incident metrics: {metrics}")
        return metrics

    def generate_playbook(self, output_path: Optional[str] = None) -> str:
        """
        Generate incident response playbook.

        Args:
            output_path: Optional path to save playbook

        Returns:
            Incident response playbook markdown
        """
        logger.info("Generating incident response playbook")

        playbook = f"""# Incident Response Playbook - {self.company_name}

**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
**Last Tested:** [Schedule tabletop exercise]

## 1. Incident Classification

"""

        for severity, info in INCIDENT_SEVERITIES.items():
            playbook += f"""### {severity} - {info['name']}
- **Response Time:** {info['response_time']}
- **Customer Notification:** {info['notification']}

"""

        playbook += f"""## 2. Incident Response Team

**Incident Commander:** [CTO Name]
- Phone: [Verified Number]
- Email: {ComplianceConfig.INCIDENT_RESPONSE_EMAIL}

**Technical Lead:** [DevOps Engineer]
- Phone: [Verified Number]

**Legal Counsel:** [Law Firm]
- 24/7 Hotline: [Verified Number]

## 3. Response Phases

### Phase 1: Detection
- Datadog alerts trigger automatically
- Customer reports via support
- Security scans detect anomalies

### Phase 2: Analysis
- Classify severity (P0-P3)
- Assess scope and impact
- Create incident channel: #incident-[ID]

### Phase 3: Containment
- Block attacker IPs via WAF
- Disable compromised accounts
- Isolate affected systems
- Preserve evidence

### Phase 4: Eradication
- Remove threat
- Patch vulnerabilities
- Verify systems clean

### Phase 5: Recovery
- Restore normal operations
- Monitor for recurrence
- Verify data integrity

### Phase 6: Post-Incident Review
- Document lessons learned
- Update playbook
- Implement improvements

## 4. Communication Templates

### Customer Notification (P0/P1)

Subject: Security Incident Notification

Dear [Customer],

We are writing to inform you of a security incident that may have affected your data.

**What Happened:** [Brief description]
**When:** [Date/time]
**Impact:** [Scope of data affected]
**Actions Taken:** [Containment and remediation]
**Your Actions:** [Recommendations for customer]

We take security seriously and have implemented additional measures to prevent recurrence.

Contact: {ComplianceConfig.INCIDENT_RESPONSE_EMAIL}

Sincerely,
{self.company_name} Security Team

## 5. Testing Requirements

- **Quarterly:** Procedure testing
- **Annually:** Full tabletop exercise
- **Contact Verification:** Quarterly

**Last Drill:** [Date]
**Next Scheduled:** [Date]
"""

        if output_path:
            Path(output_path).write_text(playbook)
            logger.info(f"Incident response playbook saved to {output_path}")

        return playbook


class EvidenceCollector:
    """Collect and organize compliance evidence."""

    def __init__(self, evidence_dir: str = "evidence"):
        """
        Initialize evidence collector.

        Args:
            evidence_dir: Directory to store evidence
        """
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized EvidenceCollector with dir: {evidence_dir}")

    def collect_user_access_matrix(self) -> Dict[str, Any]:
        """
        Collect user access control evidence (CC5.1).

        Returns:
            User access matrix data
        """
        logger.info("Collecting user access matrix")

        # Mock data - in production, query from auth system
        evidence = {
            "collection_date": datetime.now().isoformat(),
            "total_users": 12,
            "by_role": {
                "admin": 2,
                "support": 3,
                "engineer": 5,
                "tenant_user": 2
            },
            "mfa_enabled": 8,
            "mfa_adoption_rate": 0.67
        }

        return evidence

    def collect_backup_evidence(self) -> Dict[str, Any]:
        """
        Collect backup and recovery evidence (CC7.4).

        Returns:
            Backup evidence data
        """
        logger.info("Collecting backup evidence")

        # Mock data - in production, query from backup system
        evidence = {
            "collection_date": datetime.now().isoformat(),
            "backup_frequency": "daily",
            "last_backup_date": datetime.now().isoformat(),
            "backup_success_rate": 1.0,
            "total_backups_last_30_days": 30,
            "failed_backups": 0,
            "last_recovery_test": (datetime.now() - timedelta(days=45)).isoformat(),
            "recovery_test_status": "pass"
        }

        return evidence

    def collect_vulnerability_scan(self) -> Dict[str, Any]:
        """
        Collect vulnerability scan evidence (CC7.1).

        Returns:
            Vulnerability scan data
        """
        logger.info("Collecting vulnerability scan evidence")

        # Mock data - in production, query from Trivy/scanner
        evidence = {
            "collection_date": datetime.now().isoformat(),
            "scan_date": datetime.now().isoformat(),
            "scanner": "Trivy",
            "vulnerabilities": {
                "critical": 0,
                "high": 2,
                "medium": 5,
                "low": 12
            },
            "remediation_status": {
                "critical": "all_fixed",
                "high": "2_in_progress",
                "medium": "5_accepted_risk"
            }
        }

        return evidence

    def collect_all_evidence(self, period: Optional[str] = None) -> Dict[str, ComplianceEvidence]:
        """
        Collect all evidence types for the period.

        Args:
            period: Evidence period (e.g., "2025-11"), defaults to current month

        Returns:
            Dictionary of evidence by type
        """
        if period is None:
            period = datetime.now().strftime("%Y-%m")

        logger.info(f"Collecting all evidence for period: {period}")

        evidence_records = {}

        # Collect user access matrix
        user_access_data = self.collect_user_access_matrix()
        evidence_records["user_access_matrix"] = ComplianceEvidence(
            evidence_type="user_access_matrix",
            collection_date=datetime.now().isoformat(),
            period=period,
            data=user_access_data
        )

        # Collect backup evidence
        backup_data = self.collect_backup_evidence()
        evidence_records["backup_logs"] = ComplianceEvidence(
            evidence_type="backup_logs",
            collection_date=datetime.now().isoformat(),
            period=period,
            data=backup_data
        )

        # Collect vulnerability scans
        vuln_data = self.collect_vulnerability_scan()
        evidence_records["vulnerability_scans"] = ComplianceEvidence(
            evidence_type="vulnerability_scans",
            collection_date=datetime.now().isoformat(),
            period=period,
            data=vuln_data
        )

        # Save evidence to disk
        self.save_evidence(evidence_records, period)

        logger.info(f"Collected {len(evidence_records)} evidence types")
        return evidence_records

    def save_evidence(self, evidence: Dict[str, ComplianceEvidence], period: str) -> None:
        """
        Save evidence to disk organized by period.

        Args:
            evidence: Dictionary of evidence records
            period: Evidence period (e.g., "2025-11")
        """
        period_dir = self.evidence_dir / period
        period_dir.mkdir(exist_ok=True)

        for evidence_type, record in evidence.items():
            filename = f"{evidence_type}.json"
            filepath = period_dir / filename

            with open(filepath, 'w') as f:
                json.dump(asdict(record), f, indent=2)

            logger.info(f"Saved evidence: {filepath}")

        # Create summary
        summary = {
            "period": period,
            "collection_date": datetime.now().isoformat(),
            "evidence_types": list(evidence.keys()),
            "total_evidence": len(evidence)
        }

        summary_path = period_dir / "evidence_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved evidence summary: {summary_path}")


def create_sample_privacy_policy() -> str:
    """
    Create sample privacy policy with example data.

    Returns:
        Privacy policy markdown text
    """
    logger.info("Creating sample privacy policy")

    generator = PrivacyPolicyGenerator(
        company_name=ComplianceConfig.COMPANY_NAME,
        contact_email=ComplianceConfig.COMPANY_EMAIL
    )

    subprocessors = [
        Subprocessor("AWS", "Infrastructure", "All data", "US-East-1", True, "2025-01-15"),
        Subprocessor("Pinecone", "Vector storage", "Embeddings", "US-East-1", True, "2025-01-15"),
        Subprocessor("OpenAI", "Embeddings/LLM", "Text (no PII)", "US", True, "2025-01-15"),
        Subprocessor("Datadog", "Monitoring", "Logs (PII redacted)", "US/EU", True, "2025-02-01"),
    ]

    retentions = [
        DataRetention("Account data", "Account lifetime + 7 years", "Tax/legal"),
        DataRetention("User content", "Account lifetime + 30 days", "Service + deletion buffer"),
        DataRetention("Audit logs", "2 years", "Security/compliance"),
        DataRetention("Backups", "90 days", "Disaster recovery"),
    ]

    policy = generator.generate_privacy_policy(subprocessors, retentions)
    logger.info("Sample privacy policy created")

    return policy


def create_sample_soc2_controls() -> str:
    """
    Create sample SOC 2 controls documentation.

    Returns:
        SOC 2 documentation markdown text
    """
    logger.info("Creating sample SOC 2 controls")

    soc2 = SOC2Documentation(ComplianceConfig.COMPANY_NAME)

    # Add sample controls
    soc2.add_control(SOC2Control(
        control_id="CC5.1",
        control_name="Logical Access Controls",
        description="Access to customer data restricted to authorized personnel only.",
        implementation="""- Authentication: Email/password + MFA (optional but encouraged)
- Authorization: Role-Based Access Control (RBAC) via Casbin
- Roles Defined:
  - Admin: Full access (2 people only)
  - Support: Read-only for support tickets
  - Engineer: Dev/staging only (no production)
  - Tenant User: Own tenant data only""",
        evidence_location="evidence/user_access_matrix.csv, config/casbin_policy.conf",
        responsible_party="CTO"
    ))

    soc2.add_control(SOC2Control(
        control_id="CC6.7",
        control_name="Encryption",
        description="Data encrypted in transit and at rest.",
        implementation="""- Transit: TLS 1.3 enforced on all API endpoints
- At Rest: AES-256 encryption on RDS, S3, EBS volumes
- Key Management: AWS KMS with automatic key rotation""",
        evidence_location="evidence/encryption_status.json, AWS console screenshots",
        responsible_party="DevOps Lead"
    ))

    soc2.add_control(SOC2Control(
        control_id="CC7.4",
        control_name="Backup and Recovery",
        description="Regular backups with tested recovery procedures.",
        implementation="""- Frequency: Daily automated backups at 2 AM UTC
- Retention: 90 days
- Testing: Quarterly recovery tests
- Last Test: {date} - PASSED""".format(date=(datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')),
        evidence_location="evidence/backup_logs.json, recovery_tests/",
        responsible_party="DevOps Lead"
    ))

    doc = soc2.generate_soc2_documentation()
    logger.info("Sample SOC 2 controls created")

    return doc


def create_sample_incident_playbook() -> str:
    """
    Create sample incident response playbook.

    Returns:
        Incident response playbook markdown text
    """
    logger.info("Creating sample incident response playbook")

    ir_plan = IncidentResponsePlan(ComplianceConfig.COMPANY_NAME)
    playbook = ir_plan.generate_playbook()

    logger.info("Sample incident response playbook created")
    return playbook


def run_evidence_collection() -> Dict[str, ComplianceEvidence]:
    """
    Run monthly evidence collection.

    Returns:
        Dictionary of collected evidence
    """
    logger.info("Running evidence collection")

    collector = EvidenceCollector()
    evidence = collector.collect_all_evidence()

    logger.info(f"Evidence collection complete: {len(evidence)} types collected")
    return evidence


