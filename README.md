# L3 Module 12: Self-Service Tenant Onboarding

> **Automate SaaS customer onboarding using multi-tenant architecture and billing integration.** Enables customers to become productive within 5 minutes through automated provisioning.

## Overview

This Level 3 module teaches how to build a complete self-service onboarding system that eliminates the 10-15 hours weekly spent on manual customer setup. The solution automates:

- **Signup & Payment Capture** - Synchronous collection of user info and Stripe checkout
- **Automated Provisioning** - Background tenant setup via Celery (Pinecone namespace, DB tables, API keys)
- **Welcome & Activation** - Email delivery and interactive setup wizard
- **Activation Monitoring** - Analytics to track conversion funnel and identify drop-offs

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (or leave empty for offline/limited mode)
```

### 3. Run Tests

```powershell
# Windows PowerShell
$env:PYTHONPATH="$PWD"; pytest -q
# or use the script
.\scripts\run_tests.ps1
```

```bash
# Unix/Linux/Mac
export PYTHONPATH=$PWD && pytest -q
# or use the script
./scripts/run_tests.sh
```

### 4. Start API Server

```powershell
# Windows PowerShell
$env:PYTHONPATH="$PWD"; uvicorn app:app --reload
# or use the script
.\scripts\run_api.ps1
```

```bash
# Unix/Linux/Mac
export PYTHONPATH=$PWD && uvicorn app:app --reload
# or use the script
./scripts/run_api.sh
```

### 5. Explore the Notebook

```bash
jupyter lab notebooks/L3_M12_Self_Service_Tenant_Onboarding.ipynb
```

## How It Works

```
┌─────────────┐
│   Signup    │  User submits email, company, password, plan
│  (Public)   │  → Creates skeleton tenant (status: pending_payment)
└──────┬──────┘  → Returns Stripe Checkout URL
       │
       ▼
┌─────────────┐
│   Stripe    │  User completes payment
│  Checkout   │  → Stripe sends webhook (checkout.session.completed)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Webhook   │  Status → "provisioning"
│   Handler   │  Stores customer/subscription IDs
└──────┬──────┘  Triggers Celery background task
       │
       ▼
┌─────────────┐
│   Celery    │  1. Create Pinecone namespace
│ Provisioning│  2. Generate API keys (JWT)
│    Task     │  3. Load sample data
└──────┬──────┘  4. Status → "active"
       │
       ▼
┌─────────────┐
│   Welcome   │  Send email with login link + API key
│    Email    │  Guide user through setup wizard
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Activation  │  Track: login, first document, first query
│  Analytics  │  Calculate conversion rates
└─────────────┘  Identify drop-off points
```

## Common Failures & Fixes

### Failure 1: Provisioning Job Crashes

**Symptom:** Tenant stuck in "provisioning" status indefinitely. User sees loading screen forever.

**Fix:**
- Implement timeout logic (default: 5 minutes)
- Transition to "failed" status with retry button
- See `check_provisioning_timeout()` function

```python
if check_provisioning_timeout(tenant):
    tenant['status'] = 'failed'
    # Show retry option to user
```

### Failure 2: Wizard Complexity

**Symptom:** Users drop off at document upload step (30%+ abandon rate).

**Fix:**
- A/B test simpler 2-step wizard vs. detailed configuration
- Track dropout rates per step via activation events
- Provide skip option with pre-loaded sample data

### Failure 3: Sample Data Loading Fails

**Symptom:** New tenant sees empty/broken account on first login.

**Fix:**
- Pre-generate sample data during provisioning
- Include error message with fallback to empty account
- Set `SAMPLE_DATA_ENABLED=false` to disable

### Failure 4: Inaccurate Activation Tracking

**Symptom:** Client-side tracking doesn't match reality.

**Fix:**
- Use server-side event logging (`/activation/track` endpoint)
- Validate client events against backend logs
- Monitor discrepancies in dashboard

### Failure 5: Welcome Email to Spam

**Symptom:** Customers never receive login credentials.

**Fix:**
- Configure authenticated domain (SPF/DKIM/DMARC)
- Include preview text
- Avoid spam trigger words ("free", "click here", etc.)
- Test with mail-tester.com before production

## Decision Card

### ✅ Use Self-Service When:

- **Target market:** SMB/mid-market (not enterprise)
- **Product simplicity:** Intuitive core workflows
- **Sales cycle:** <1 week
- **Unit economics:** Sustainable at $100+ MRR

### ❌ Avoid When:

- **Enterprise-first strategy:** Custom contracts required
- **Significant implementation:** Expert configuration needed
- **Complex integrations:** Prerequisites for setup
- **Compliance/security reviews:** Manual vetting necessary

### 🔀 Hybrid Approach (Recommended):

- **Self-service:** Starter/Pro plans
- **Sales-assisted:** Enterprise tier
- **Monitor activation rates:** Escalate at-risk customers to support

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Signup
```bash
curl -X POST http://localhost:8000/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "company_name": "Example Corp",
    "password": "SecurePass123!",
    "plan": "pro"
  }'
```

### Get Tenant Status
```bash
curl http://localhost:8000/tenant/{tenant_id}
```

### Track Activation Event
```bash
curl -X POST http://localhost:8000/activation/track \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "abc123",
    "event_type": "first_query_executed",
    "metadata": {"query": "test query"}
  }'
```

### Get Activation Metrics
```bash
curl http://localhost:8000/activation/metrics
```

## Production Considerations

### Scaling to 100+ Daily Signups

- **Rate limiting:** Implement on `/signup` endpoint (default: 10/hour/IP)
- **Redis caching:** Cache tenant lookups to reduce DB load
- **Celery workers:** Scale horizontally based on queue depth
- **Idempotent tasks:** Ensure provisioning can be safely retried

### Cost Breakdown

- **Stripe:** 2.9% + $0.30 per transaction
- **Pinecone:** $0.04-0.40 per 1M vectors (pod-dependent)
- **Celery/Redis:** Negligible with proper resource pooling
- **Email:** $0.0001-0.001 per message (SendGrid)

### Monitoring

Track these metrics:

1. **Signup funnel completion rate** - % who complete payment
2. **Provisioning job success rate** - % successful vs. failed
3. **Mean time-to-activation** - Signup → first query
4. **Webhook delivery latency** - Stripe → provisioning trigger

**Alert when:**
- Task queue backlog >100 jobs
- Provisioning success rate <95%
- Webhook latency >30 seconds

### Security

- ✓ Webhook signature verification prevents unauthorized provisioning
- ✓ API keys use tenant-scoped JWT claims
- ✓ Sample data loads with temporary, restricted credentials
- ✓ Rate limiting on signup endpoint prevents abuse

### Compliance

- **GDPR:** Store consent for marketing emails; implement unsubscribe
- **SOC2:** Audit webhook processing; maintain event logs (90 days)
- **PCI:** Never store raw card data; rely on Stripe tokenization

## Troubleshooting

### "Stripe client not available"

Check `.env` has valid `STRIPE_SECRET_KEY`. The module will gracefully skip Stripe calls and return mock URLs in development.

### "Pinecone namespace creation failed"

Verify:
1. `PINECONE_API_KEY` is set
2. Index exists (create via Pinecone console)
3. API key has write permissions

### "Celery task not triggered"

Ensure:
1. Redis is running (`redis-cli ping`)
2. Celery worker is started (`celery -A tasks worker`)
3. `CELERY_BROKER_URL` points to Redis

### "Welcome email not sent"

Check:
1. `SENDGRID_API_KEY` is valid
2. `SENDGRID_FROM_EMAIL` is verified sender
3. Email not in spam (check SPF/DKIM/DMARC)

## Environment Variables

Copy `.env.example` to `.env` and configure the following keys:

```bash
# Database
DATABASE_URL=sqlite:///./tenants.db

# Stripe (payment processing)
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
STRIPE_PRICE_ID_STARTER=price_starter_monthly
STRIPE_PRICE_ID_PRO=price_pro_monthly
STRIPE_PRICE_ID_ENTERPRISE=price_enterprise_monthly

# Pinecone (vector database)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=saas-tenants

# Redis/Celery (background jobs)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# SendGrid (email delivery)
SENDGRID_API_KEY=SG.your_sendgrid_api_key
SENDGRID_FROM_EMAIL=onboarding@yourdomain.com
SENDGRID_FROM_NAME=Your SaaS Platform

# Application
APP_URL=http://localhost:8000
JWT_SECRET=your-secret-jwt-key-min-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=720

# Provisioning
PROVISIONING_TIMEOUT_SECONDS=300
MAX_PROVISIONING_RETRIES=3
SAMPLE_DATA_ENABLED=true

# Rate Limiting
SIGNUP_RATE_LIMIT_PER_HOUR=10
```

## Offline / Limited Mode

**The module runs in a limited mode if API keys are not configured.** This allows you to explore the codebase, run tests, and understand the architecture without requiring live service accounts.

When API keys (`STRIPE_SECRET_KEY`, `PINECONE_API_KEY`, `SENDGRID_API_KEY`) are not set in `.env`, the `config.py` file returns `None` for these clients. The core logic gracefully handles this by:

- **Stripe:** Returning mock checkout URLs instead of creating real sessions
- **Pinecone:** Skipping namespace creation and vector upserts (logs warning)
- **SendGrid:** Skipping email delivery (logs warning)

All tests pass in offline mode. The API returns `200` responses with `{"skipped": true, "reason": "..."}` for operations requiring external services.

To enable **OFFLINE mode** in the notebook, set:
```bash
export OFFLINE=true
```

## File Structure

```
.
├── app.py                                    # FastAPI server (thin wrapper)
├── config.py                                 # Configuration & client accessors
├── src/
│   └── l3_m12_self_service_tenant_onboarding/
│       └── __init__.py                       # Core business logic
├── notebooks/
│   └── L3_M12_Self_Service_Tenant_Onboarding.ipynb  # Interactive tutorial
├── tests/
│   └── test_m12_self_service_tenant_onboarding.py   # Test suite
├── configs/
│   └── example.json                          # Plan configurations
├── scripts/
│   ├── run_api.ps1                          # Windows API launcher
│   ├── run_api.sh                           # Unix API launcher
│   ├── run_tests.ps1                        # Windows test runner
│   └── run_tests.sh                         # Unix test runner
├── requirements.txt                          # Dependencies
├── .env.example                              # Environment template
├── example_data.json                         # Sample data
├── .gitignore                                # Git ignore rules
├── LICENSE                                   # License file
└── README.md                                 # This file
```

## Next Steps

After completing this module, proceed to:

- **Module 12.4:** Usage-based billing and metering
- **Module 13.1:** Multi-region deployment strategies
- **Module 13.2:** Disaster recovery and backup automation

## Resources

- [Stripe Checkout Docs](https://stripe.com/docs/payments/checkout)
- [Pinecone Namespaces](https://docs.pinecone.io/docs/namespaces)
- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html)
- [SendGrid Authentication](https://docs.sendgrid.com/ui/account-and-settings/how-to-set-up-domain-authentication)

---

**Key Takeaway:** *Self-service onboarding doesn't mean abandoned customers—it means automated onboarding with monitored activation. Build the automation pipeline, track where customers stick, and intervene when they struggle.*
