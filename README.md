# Module 13.2: Governance & Compliance Documentation

Enterprise-grade compliance documentation system for B2B SaaS. Generate GDPR-compliant privacy policies, SOC 2 controls documentation, incident response playbooks, and automate compliance evidence collection.

## Overview

This module provides tools for creating compliance documentation that unlocks enterprise sales ($50K+ annual contracts). When Fortune 500 companies request SOC 2 certification or send 47-page security questionnaires, you'll have everything ready.

**Key Features:**
- **Privacy Policy Generator:** GDPR/CCPA-compliant policies with subprocessor tracking
- **SOC 2 Documentation:** Map your security controls to 64 SOC 2 requirements
- **Incident Response:** Tested playbooks with severity classification (P0-P3)
- **Evidence Collection:** Automated monthly collection for audit readiness
- **REST API:** FastAPI endpoints for integration with existing systems

**Use When:**
- ✅ Closing $50K+ enterprise deals requiring SOC 2
- ✅ Annual revenue > $200K (can afford compliance costs)
- ✅ B2B SaaS targeting Fortune 500 customers
- ✅ Security controls already implemented (L2 M6 complete)

**Avoid When:**
- ❌ Pre-product-market-fit (<10 customers)
- ❌ B2C consumer apps (SOC 2 not required)
- ❌ Revenue < $200K ARR (can't justify costs)
- ❌ Security controls not yet built (documentation without controls is fraud)

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your company information and AWS credentials (optional)
```

### 3. Run REST API

**Windows (PowerShell):**
```powershell
.\scripts\run_api.ps1
# or
powershell -c "$env:PYTHONPATH='$PWD'; uvicorn app:app --reload"
```

**Linux/Mac (Bash):**
```bash
export PYTHONPATH=$PWD
uvicorn app:app --reload
```

API available at http://localhost:8000
Interactive docs at http://localhost:8000/docs

### 4. Run Tests

**Windows (PowerShell):**
```powershell
.\scripts\run_tests.ps1
# or
powershell -c "$env:PYTHONPATH='$PWD'; pytest -q"
```

**Linux/Mac (Bash):**
```bash
export PYTHONPATH=$PWD
pytest -q tests/
```

### 5. Explore the Notebook

```bash
jupyter lab notebooks/L3_M13_Governance_Compliance_Documentation.ipynb
```

## Environment Variables

Configure your `.env` file with the following variables (see `.env.example` for template):

**Required (Company Information):**
```bash
COMPANY_NAME="YourCompany Inc."
COMPANY_EMAIL="privacy@yourcompany.com"
COMPLIANCE_CONTACT="compliance@yourcompany.com"
INCIDENT_RESPONSE_EMAIL="incidents@yourcompany.com"
```

**Optional (AWS Evidence Collection):**
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
EVIDENCE_BUCKET=compliance-evidence-bucket
```

**Optional (Database):**
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/compliance_db
```

**Optional (SOC 2 Configuration):**
```bash
SOC2_AUDIT_FIRM="Audit Firm Name"
SOC2_LAST_AUDIT_DATE=2025-01-15
PENETRATION_TEST_DATE=2025-01-15
```

**Optional (API Configuration):**
```bash
API_PORT=8000
API_HOST=0.0.0.0
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-change-in-production
```

**Note:** The module runs in a limited "degraded" mode if AWS credentials or database URL are not provided. API endpoints will return `{"skipped": true, "reason": "Service not initialized"}` for operations requiring these services.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                   Compliance Documentation System            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐   ┌─────────────────┐
│ Privacy Policy│    │ SOC 2 Controls │   │ Incident Response│
│   Generator   │    │  Documentation │   │    Playbook     │
└───────────────┘    └────────────────┘   └─────────────────┘
        │                     │                     │
        │  GDPR-compliant     │  64 controls        │  P0-P3 severity
        │  + subprocessors    │  + evidence         │  + procedures
        │  + retentions       │  + responsible      │  + contacts
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Evidence Collector│
                    │  (Monthly Cron)  │
                    └──────────────────┘
                              │
                              │  Automated collection:
                              │  - User access matrix
                              │  - Backup logs
                              │  - Vulnerability scans
                              │  - Training records
                              │
                              ▼
                    ┌──────────────────┐
                    │ Audit-Ready      │
                    │ Evidence Package │
                    └──────────────────┘
```

### The Four Layers of Compliance

1. **Security Controls** (What You Actually Do)
   - Implement: PII detection, encryption, RBAC, audit logs
   - From: Level 2 M6 + M13.1

2. **Policies** (What You Say You Do)
   - Document: Privacy policy, security policies, procedures
   - This module: Generates policies matching your implementation

3. **Evidence** (Proof You Do What You Say)
   - Collect: Audit logs, access records, test results
   - This module: Automates monthly evidence collection

4. **Governance** (How You Maintain It)
   - Review: Quarterly policy reviews, annual drills
   - This module: Provides checklists and schedules

## Common Failures & Fixes

### Failure 1: Incomplete Privacy Policy (Missing GDPR Sections)

**Symptom:** Enterprise legal team rejects policy due to missing DPA terms, incomplete subprocessor list, or vague retention periods.

**Fix:**
```python
# Add all subprocessors with DPA status
subprocessors = [
    Subprocessor("AWS", "Infrastructure", "All data", "US-East-1", True, "2025-01-15"),
    Subprocessor("Pinecone", "Vector DB", "Embeddings", "US-East-1", True, "2025-01-15"),
    # ... complete list
]

# Specify exact retention periods
retentions = [
    DataRetention("Account data", "Account lifetime + 7 years", "Tax/legal"),
    # ... not "as long as necessary"
]
```

**Prevention:** Use B2B-specific templates, get legal review ($2K-5K), maintain subprocessor register.

### Failure 2: Outdated Security Assessments (Penetration Test >6 Months Old)

**Symptom:** Enterprise security team requires pen test within last 6 months. Your last test was 10 months ago.

**Fix:**
- Commission new penetration test ($5K-15K, 2-3 weeks)
- OR delta assessment on changes since last test ($2K-5K, 1 week)
- OR self-assessment with CTO attestation (minor changes only)

**Prevention:** Schedule annual pen tests same month each year. Track significant changes (new endpoints, auth changes).

### Failure 3: Unclear Data Handling Policies (Customer Confusion on AI/ML)

**Symptom:** Enterprise customer asks: "Do you train AI models on our documents?"

**Fix:** Replace vague language with specific policies:
```markdown
### Service Delivery (Required)
- OpenAI does NOT retain your data (zero-retention DPA)
- Your documents NEVER shared with other customers

### AI Model Training (OPT-IN ONLY - Disabled by Default)
- We do NOT use your data for training unless you explicitly opt in
```

**Prevention:** Never use vague language. Default to NO data usage beyond required service. Make all non-essential usage OPT-IN.

### Failure 4: Untested Incident Response Plan

**Symptom:** Real incident occurs. Contact numbers disconnected, procedures don't work, team unfamiliar with plan.

**Fix:**
1. Verify ALL contact information (test call every number)
2. Test procedures (can you actually block IP via WAF?)
3. Conduct tabletop exercise (simulate incident with team)

**Prevention:** Quarterly contact verification, quarterly procedure testing, annual full tabletop exercise.

### Failure 5: Missing Compliance Evidence

**Symptom:** SOC 2 auditor requests October evidence. Logs rotated. Can't prove controls worked.

**Fix:**
```python
# Extend CloudWatch log retention
logs.put_retention_policy(
    logGroupName='/aws/lambda/rag-api',
    retentionInDays=730  # 2 years
)

# Set up monthly evidence collection
# Cron: 0 2 1 * * (first of month at 2 AM)
python evidence_collector.py
```

**Prevention:** Implement evidence retention BEFORE audit. Automate collection. Extend log retention to 2 years.

## Decision Card

### ✅ BENEFIT: Enterprise Sales Unlock

Formal compliance documentation enables enterprise sales requiring security verification. Close deals 4-8 weeks faster. Average impact: $200K-500K additional annual revenue.

### ❌ LIMITATION: High Ongoing Cost Without Guaranteed ROI

Compliance costs $30K-120K annually (legal, audits, pen tests, time). Zero ROI if you don't close enterprise deals requiring SOC 2. Early-stage (<$200K ARR) often can't afford or see positive ROI.

### 💰 COST: $30K-120K Annually + 300-400 Hours

**First year:** $30K-50K (legal, pen test, SOC 2 audit) + 60-80 hours documentation
**Ongoing:** $30K-50K/year + 20-30 hours/month (evidence, updates, questionnaires)
**Most expensive at small scale:** High % of revenue; manageable at $500K+ ARR (5-10%)

### 🤔 USE WHEN: Closing $50K+ Enterprise Deals Requiring SOC 2

Create formal documentation when you have enterprise prospects ($50K+ ACV) requesting SOC 2 or sending security questionnaires. Also when raising Series A and investors require SOC 2. Minimum: $200K ARR.

### 🚫 AVOID WHEN: Pre-PMF, <$200K ARR, or B2C Focus

Skip if pre-product-market-fit (<10 customers), revenue <$200K, or targeting consumers. Also avoid if security controls not implemented yet (L2 M6) - documentation without controls is fraud. Use basic privacy policy until you meet criteria.

## Troubleshooting

### "Privacy policy rejected by customer legal team"
- **Check:** Missing DPA terms, incomplete subprocessor list, vague retention periods
- **Fix:** Review Section 8 Failure 1 in main script, add missing sections
- **Cost:** $2K-5K legal review

### "Can't provide evidence for audit"
- **Check:** Logs rotated, no systematic collection, missing evidence types
- **Fix:** Extend log retention to 2 years, set up monthly evidence collection
- **Prevention:** Automate BEFORE audit (6 months lead time needed)

### "Incident response plan doesn't work in production"
- **Check:** Contact info outdated, procedures untested, team unfamiliar
- **Fix:** Verify contacts, test procedures, run tabletop exercise
- **Schedule:** Quarterly verification, annual drill

### "Evidence collection fails"
- **Check:** AWS credentials not configured
- **Fix:** Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env
- **Graceful:** Module skips network calls if credentials missing

### "SOC 2 audit too expensive"
- **Check:** Budget constraints, not enough enterprise pipeline
- **Options:**
  - Wait until $200K+ ARR (better ROI)
  - Use Vanta/Drata for guided audit ($27K-55K all-in)
  - DIY minimal (basic privacy policy only, $2K-5K)

### **Offline/Limited Mode**

The module runs in a limited "degraded" mode if `DATABASE_URL` or `AWS_ACCESS_KEY_ID` are not set in `.env`.

**What happens:**
- The `config.py` file returns `None` for clients that require credentials
- The `app.py` logic detects missing services and returns: `{"skipped": true, "reason": "Service not initialized (missing credentials)"}`
- Evidence collection uses mock data instead of real AWS API calls
- All core functionality (policy generation, SOC 2 documentation, incident planning) works normally

**When to use:**
- ✅ Local development without cloud dependencies
- ✅ CI/CD environments running tests
- ✅ Demonstrations without exposing credentials
- ✅ Learning the module structure and API

**To enable full mode:**
1. Copy `.env.example` to `.env`
2. Add your AWS credentials and database URL
3. Restart the API server
4. Set `OFFLINE=false` when running notebooks

## API Reference

### Health Check
```bash
GET /health
```

### Generate Privacy Policy
```bash
POST /generate/privacy-policy
{
  "company_name": "Optional",
  "contact_email": "Optional"
}
```

### Generate SOC 2 Controls
```bash
POST /generate/soc2-controls
```

### Generate Incident Playbook
```bash
POST /generate/incident-playbook
```

### Create Incident
```bash
POST /incident/create
{
  "description": "Security event description",
  "severity": "P2"  # P0, P1, P2, or P3
}
```

### Collect Evidence
```bash
POST /evidence/collect?period=2025-11
```

### Compliance Status
```bash
GET /compliance/status
```

## File Structure

```
.
├── src/
│   └── l3_m13_governance_compliance_rag/
│       └── __init__.py              # Core business logic
├── tests/
│   └── test_m13_governance_compliance_rag.py  # Pytest tests
├── notebooks/
│   └── L3_M13_Governance_Compliance_Documentation.ipynb
├── configs/
│   └── example.json                 # Config templates
├── scripts/
│   ├── run_api.ps1                  # Windows API launcher
│   └── run_tests.ps1                # Windows test runner
├── app.py                           # FastAPI application (thin wrapper)
├── config.py                        # Configuration management
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore patterns
├── README.md                        # This file
├── example_data_subprocessors.json  # Sample data
├── example_data_user_access_matrix.csv
├── example_data_soc2_controls.json
└── evidence/                        # Auto-generated (gitignored)
    └── 2025-11/
        ├── evidence_summary.json
        ├── user_access_matrix.json
        ├── backup_logs.json
        └── vulnerability_scans.json
```

## Next Steps

### If You're Ready for SOC 2 ($200K+ ARR, Enterprise Pipeline)
1. Commission penetration test ($5K-15K)
2. Start evidence collection NOW (need 6 months)
3. Get legal review of policies ($2K-5K)
4. Schedule SOC 2 Type I audit ($15K-40K)
5. Timeline: 6-9 months to SOC 2 report

### If You're Early Stage (<$200K ARR)
1. Create basic privacy policy only (use DIY minimal)
2. Implement security controls (Level 2 M6)
3. Wait until first $50K+ enterprise prospect requires SOC 2
4. Then pursue formal compliance (3-6 month lead time acceptable)

### Alternative Approaches
- **Vanta/Drata:** Automated compliance ($30K-55K/year all-in)
- **Consultant:** Expert guidance ($40K-80K/year)
- **DIY Minimal:** Basic privacy policy only ($2K-5K one-time)

See Decision Framework in main script for detailed comparison.

## Next Module

**M13.3: Launch Preparation & Marketing**
- Landing page and positioning
- Pricing strategy (3 tiers)
- Customer acquisition funnel
- Marketing site with signup

This is where your SaaS becomes a real business.

## Support

- **GitHub Issues:** [Report issues](https://github.com/yesvisare/ccc_l3_aug_practical/issues)
- **Documentation:** See main script M13_2_MAIN_SCRIPT.md
- **Course Discord:** #module-13

## License

Educational purposes - Level 3 CCC August Practical

---

**Remember:** Documentation follows implementation. Never document controls you don't have. Compliance without actual security is fraud, not compliance.

**Cost Reality:** Budget $30K-120K annually. Only pursue if revenue supports it ($200K+ ARR minimum).

**Time Reality:** 300-400 hours/year ongoing. Assign 0.2 FTE or outsource.

**ROI Reality:** Only positive if closing $50K+ enterprise deals. Zero value for B2C or pre-revenue.
