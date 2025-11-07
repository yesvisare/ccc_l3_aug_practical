"""
Module 13.2: Governance & Compliance Documentation
Configuration Management

Loads environment variables and provides configuration access.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ComplianceConfig:
    """Configuration for compliance documentation system."""

    # Company Information
    COMPANY_NAME: str = os.getenv("COMPANY_NAME", "YourCompany Inc.")
    COMPANY_EMAIL: str = os.getenv("COMPANY_EMAIL", "privacy@yourcompany.com")
    COMPLIANCE_CONTACT: str = os.getenv("COMPLIANCE_CONTACT", "compliance@yourcompany.com")
    INCIDENT_RESPONSE_EMAIL: str = os.getenv("INCIDENT_RESPONSE_EMAIL", "incidents@yourcompany.com")

    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCOUNT_ID: Optional[str] = os.getenv("AWS_ACCOUNT_ID")

    # Database
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # Evidence Collection
    EVIDENCE_BUCKET: str = os.getenv("EVIDENCE_BUCKET", "compliance-evidence-bucket")
    EVIDENCE_RETENTION_DAYS: int = int(os.getenv("EVIDENCE_RETENTION_DAYS", "730"))

    # Monitoring
    DATADOG_API_KEY: Optional[str] = os.getenv("DATADOG_API_KEY")

    # SOC 2 Configuration
    SOC2_AUDIT_FIRM: str = os.getenv("SOC2_AUDIT_FIRM", "Audit Firm Name")
    SOC2_LAST_AUDIT_DATE: str = os.getenv("SOC2_LAST_AUDIT_DATE", "2025-01-15")
    PENETRATION_TEST_DATE: str = os.getenv("PENETRATION_TEST_DATE", "2025-01-15")

    # API Configuration
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production")

    @classmethod
    def has_aws_credentials(cls) -> bool:
        """Check if AWS credentials are configured."""
        return bool(cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY)

    @classmethod
    def has_database(cls) -> bool:
        """Check if database is configured."""
        return bool(cls.DATABASE_URL)


def get_clients():
    """
    Initialize and return AWS clients if credentials available.

    Returns:
        dict: Dictionary of initialized clients, empty if credentials missing
    """
    clients = {}

    if not ComplianceConfig.has_aws_credentials():
        return clients

    try:
        import boto3

        clients['s3'] = boto3.client(
            's3',
            aws_access_key_id=ComplianceConfig.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=ComplianceConfig.AWS_SECRET_ACCESS_KEY,
            region_name=ComplianceConfig.AWS_REGION
        )

        clients['logs'] = boto3.client(
            'logs',
            aws_access_key_id=ComplianceConfig.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=ComplianceConfig.AWS_SECRET_ACCESS_KEY,
            region_name=ComplianceConfig.AWS_REGION
        )

        clients['rds'] = boto3.client(
            'rds',
            aws_access_key_id=ComplianceConfig.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=ComplianceConfig.AWS_SECRET_ACCESS_KEY,
            region_name=ComplianceConfig.AWS_REGION
        )

    except ImportError:
        pass  # boto3 not installed
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize AWS clients: {e}")

    return clients


# Constants for compliance frameworks
GDPR_REQUIREMENTS = 88
SOC2_CONTROLS = 64
HIPAA_SAFEGUARDS = 164

# SOC 2 Trust Principles
SOC2_PRINCIPLES = [
    "CC1-CC5: Common Criteria (Governance & Risk)",
    "CC6: Logical & Physical Access",
    "CC7: System Operations",
    "CC8: Change Management",
    "CC9: Risk Mitigation"
]

# Evidence types to collect
EVIDENCE_TYPES = [
    "user_access_matrix",
    "mfa_adoption",
    "encryption_status",
    "backup_logs",
    "monitoring_uptime",
    "training_completion",
    "change_management",
    "vulnerability_scans"
]

# Incident severity levels
INCIDENT_SEVERITIES = {
    "P0": {"name": "Critical", "response_time": "15 minutes", "notification": "24 hours"},
    "P1": {"name": "High", "response_time": "1 hour", "notification": "72 hours"},
    "P2": {"name": "Medium", "response_time": "4 hours", "notification": "N/A"},
    "P3": {"name": "Low", "response_time": "24 hours", "notification": "N/A"}
}
