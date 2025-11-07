# Module 12.2: Billing Integration

**Automate subscription and usage-based billing with Stripe**

---

## Overview

This module implements automated billing integration with Stripe for a multi-tenant SaaS system. It transforms usage data from M12.1 (Usage Metering) into actual payment collection with subscription management, usage-based charges, payment retries, and dunning logic.

### What This Module Does

- **Customer Management**: Sync tenants to Stripe customers (idempotent)
- **Subscription Lifecycle**: Create, cancel, and manage subscriptions with trials
- **Usage Reporting**: Daily sync from ClickHouse to Stripe for accurate billing
- **Webhook Handling**: Process payment events (success, failure, cancellation)
- **Dunning Logic**: Escalating retry strategy for failed payments

### Prerequisites

- M12.1 (Usage Metering & Analytics) completed
- Level 2 complete (multi-tenant architecture)
- Stripe account (https://stripe.com)
- Python 3.11+

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your Stripe keys:

```bash
cp .env.example .env
# Edit .env with your Stripe keys from https://dashboard.stripe.com/apikeys
```

### 3. Verify Configuration

```bash
python config.py
```

### 4. Run Smoke Tests

```bash
pytest tests_smoke.py -v
```

### 5. Start the API

```bash
python app.py
# API runs on http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 6. Explore the Notebook

```bash
jupyter notebook L2_M12_Billing_Integration.ipynb
```

---

## How It Works

### Architecture Diagram

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  ClickHouse │────────▶│  Your App   │────────▶│   Stripe    │
│   (Usage)   │         │  (Billing)  │         │  (Payment)  │
└─────────────┘         └─────────────┘         └─────────────┘
                               │
                               │ Webhooks
                               ▼
                        ┌─────────────┐
                        │  Dunning    │
                        │   Logic     │
                        └─────────────┘
```

### Key Components

1. **StripeBillingManager** (`l2_m12_billing_integration.py`)
   - `create_customer()`: Create Stripe customer for tenant
   - `create_subscription()`: Set up billing with base + usage pricing
   - `report_usage()`: Send usage data to Stripe
   - `cancel_subscription()`: Handle cancellations

2. **UsageSyncService** (`l2_m12_billing_integration.py`)
   - `sync_daily_usage()`: Batch sync usage for all tenants
   - Idempotent with `action="set"` (safe to run multiple times)

3. **DunningManager** (`l2_m12_billing_integration.py`)
   - `process_failed_payment()`: Escalating retry strategy
   - Day 1: Reminder → Day 4: Warning → Day 7: Soft suspension → Day 8+: Full suspension

4. **Webhook Handlers** (`app.py`)
   - `invoice.payment_succeeded`: Mark tenant active
   - `invoice.payment_failed`: Trigger dunning
   - `customer.subscription.deleted`: Disable access
   - Background processing to respond <5 seconds

### Pricing Plans

| Plan       | Base Price | Included Queries | Overage Rate |
|------------|-----------|------------------|--------------|
| Starter    | $29/mo    | 10,000           | $0.001/query |
| Pro        | $99/mo    | 100,000          | $0.0008/query|
| Enterprise | $499/mo   | 1,000,000        | $0.0005/query|

---

## Common Failures & Fixes

### Failure 1: Webhook Timeout (Payment Not Recorded)

**Symptom**: Customer paid but service shows suspended.

**Root Cause**: Webhook endpoint took >5 seconds, Stripe gave up.

**Fix**: Process webhooks asynchronously
```python
# Use background_tasks.add_task() or Celery
background_tasks.add_task(process_event, event)
return {"status": "success"}  # <100ms response
```

**Prevention**: Monitor webhook response time (<500ms target).

---

### Failure 2: Double-Counting Usage

**Symptom**: Customer charged 2x actual usage.

**Root Cause**: Counting both cached and database queries.

**Fix**: Single source of truth + idempotent reporting
```python
# Use action="set" not "increment"
stripe.UsageRecord.create(
    subscription_item=item_id,
    quantity=query_count,
    action="set"  # Replaces previous value
)
```

**Prevention**: Daily comparison Stripe vs ClickHouse.

---

### Failure 3: Missing Payment Method

**Symptom**: Trial ends, customer uses service free.

**Root Cause**: No payment method on file, invoice stuck in "draft".

**Fix**: Require payment method at signup
```python
# Require payment method before creating subscription
stripe.Customer.create(
    email=email,
    payment_method=payment_method_id
)
```

**Prevention**: Monitor subscriptions in "incomplete" status.

---

### Failure 4: Out-of-Order Webhooks

**Symptom**: Customer payment succeeds but gets suspended anyway.

**Root Cause**: `payment_failed` webhook processed after `payment_succeeded`.

**Fix**: Always check current state
```python
# Fetch latest status before taking action
latest_invoice = stripe.Invoice.retrieve(invoice_id)
if latest_invoice.status == "paid":
    logger.info("Ignoring old webhook - already paid")
    return
```

**Prevention**: Make all webhook handlers idempotent.

---

### Failure 5: State Mismatch (Stripe vs Database)

**Symptom**: Customer cancelled in Stripe but still has access.

**Root Cause**: Webhook missed or manual change in Stripe Dashboard.

**Fix**: Daily reconciliation job
```python
# Run daily: sync Stripe → your database
def sync_subscription_states():
    for tenant in get_active_tenants():
        stripe_sub = stripe.Subscription.retrieve(sub_id)
        if stripe_sub.status != tenant.billing_status:
            update_database(tenant, stripe_sub.status)
```

**Prevention**: Alert on any discrepancies.

---

## Decision Card

### ✅ BENEFIT: Automated Revenue Collection & Scaling

Eliminates manual invoicing (saves 8-10 hours/month), automatically retries failed payments, scales to 500+ customers without additional work. **ROI: 40-hour setup paid back in 4-6 months.**

### ❌ LIMITATION: Complex Edge Cases Require Manual Intervention

Prorations on mid-month plan changes are messy, international tax needs manual config, billing disputes from confused customers. **Expect 2-5% of invoices to need manual review in first 6 months.**

### 💰 COST: Stripe Fees + Infrastructure + Time

- Stripe fees: 2.9% + $0.30 per transaction (~3-4% of revenue)
- Infrastructure: $100-200/month (workers, monitoring)
- Engineering: 40 hours setup + 5 hours/month maintenance
- **Total: 4-5% of revenue at 100 customers**

### 🤔 USE WHEN: 10+ Customers with Monthly Subscriptions

Use if you have (or will reach in 6 months):
- >10 paying customers
- Monthly/annual billing cycles
- Validated pricing model
- SMB/mid-market customers

### 🚫 AVOID WHEN: <10 Customers or Enterprise-Only

Skip if:
- <10 customers (manual invoicing faster)
- Enterprise-only with net-30 terms
- Pricing model experimental (changes monthly)

### Alternatives Comparison

| Your Situation | Best Choice | Why |
|----------------|-------------|-----|
| <10 customers, B2B | Manual (QuickBooks) | Efficient at small scale |
| 10-500 customers | **Stripe** ⭐ | Best features/cost balance |
| >500, complex pricing | Chargebee/Recurly | Handles complexity |
| International >50% | PayPal/Braintree | Better global coverage |
| >$1M ARR | Kill Bill | Own infrastructure |

---

## Troubleshooting

### "No Stripe API key configured"

**Solution**: Set `STRIPE_SECRET_KEY` in `.env` file.

```bash
# Get keys from https://dashboard.stripe.com/apikeys
STRIPE_SECRET_KEY=sk_test_your_key_here
```

### "Invalid webhook signature"

**Solution**: Update `STRIPE_WEBHOOK_SECRET` in `.env`.

```bash
# Get from https://dashboard.stripe.com/webhooks
STRIPE_WEBHOOK_SECRET=whsec_your_secret_here
```

### "Unknown plan type: xyz"

**Solution**: Use valid plan types: `starter`, `pro`, or `enterprise`.

In production, create these prices in Stripe Dashboard first.

### Webhook Failures

**Check Stripe Dashboard** → Developers → Webhooks → View logs

Common issues:
- Endpoint URL incorrect
- Signature verification failing
- Response taking >5 seconds
- Server down during webhook delivery

---

## Production Deployment Checklist

- [ ] Switch to live Stripe keys (`sk_live_`, not `sk_test_`)
- [ ] Configure webhook endpoint in Stripe Dashboard (production URL)
- [ ] Enable webhook signature verification
- [ ] Set up Celery/SQS for async webhook processing
- [ ] Schedule daily usage sync cron job
- [ ] Configure monitoring and alerts:
  - Webhook success rate >99%
  - Payment success rate >85%
  - Usage sync lag <6 hours
- [ ] Test with Stripe test cards:
  - `4242424242424242` (success)
  - `4000000000000002` (declined)
- [ ] Set up daily reconciliation job (Stripe ↔ DB sync)
- [ ] Configure rate limiting on webhook endpoint
- [ ] Review first month of invoices manually

---

## API Endpoints

### Health Check
```bash
GET /health
# Returns: {"status": "ok", "stripe_configured": true}
```

### Create Customer
```bash
POST /customers
{
  "tenant_id": "tenant_001",
  "email": "customer@example.com",
  "name": "Company Name"
}
```

### Create Subscription
```bash
POST /subscriptions
{
  "customer_id": "cus_xxx",
  "plan_type": "pro",
  "tenant_id": "tenant_001",
  "trial_days": 14
}
```

### Report Usage
```bash
POST /usage/report
{
  "tenant_id": "tenant_001",
  "subscription_id": "sub_xxx",
  "query_count": 5000
}
```

### Process Payment Failure
```bash
POST /dunning/process
{
  "tenant_id": "tenant_001",
  "failure_count": 1,
  "invoice_amount": 99.00
}
```

### Stripe Webhook
```bash
POST /webhooks/stripe
# Automatically called by Stripe
# Handles: payment_succeeded, payment_failed, subscription_deleted, etc.
```

---

## Testing

### Run All Tests
```bash
pytest tests_smoke.py -v
```

### Test Specific Function
```bash
pytest tests_smoke.py::test_dunning_first_failure -v
```

### Test with Stripe CLI
```bash
# Install Stripe CLI: https://stripe.com/docs/stripe-cli
stripe login
stripe listen --forward-to localhost:8000/webhooks/stripe

# In another terminal, trigger events:
stripe trigger invoice.payment_succeeded
stripe trigger invoice.payment_failed
```

---

## File Structure

```
.
├── l2_m12_billing_integration.py  # Core billing logic
├── config.py                       # Configuration management
├── app.py                          # FastAPI entrypoint
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── example_data.json               # Sample data
├── tests_smoke.py                  # Smoke tests
├── L2_M12_Billing_Integration.ipynb # Jupyter notebook
└── README.md                       # This file
```

---

## Cost Analysis

### At 100 Customers × $99/month

**Revenue**: $9,900/month

**Costs**:
- Stripe fees (2.9% + $0.30): ~$315/month
- Infrastructure (Celery, monitoring): ~$100/month
- Engineering time: 40 hours setup + 5 hours/month
- **Total**: ~$415/month (4.2% of revenue)

### Cost Savings vs Manual

- Manual invoicing: 8-10 hours/month
- Automated billing: 5 hours/month (mostly support)
- **Time saved**: 3-5 hours/month = 36-60 hours/year
- At $100/hour: **$3,600-$6,000/year saved**

---

## Next Steps

1. **Complete M12.3**: Self-Service Tenant Onboarding
2. **Integrate with M12.1**: Connect ClickHouse usage to Stripe
3. **Set up monitoring**: Webhook success rate, payment metrics
4. **Test failure scenarios**: Use Stripe test cards
5. **Review first month invoices**: Catch any calculation errors early

---

## Resources

- [Stripe API Documentation](https://stripe.com/docs/api)
- [Stripe Webhooks Guide](https://stripe.com/docs/webhooks)
- [Stripe Test Cards](https://stripe.com/docs/testing)
- [Stripe CLI](https://stripe.com/docs/stripe-cli)
- [Usage-Based Billing Guide](https://stripe.com/docs/billing/subscriptions/usage-based)

---

## Support

If you get stuck:
1. Check Stripe Dashboard → Developers → Logs
2. Review the Decision Card (when NOT to use)
3. Review the 5 Common Failures
4. Check webhook logs in Stripe Dashboard

**Remember**: Expect 2-5% of invoices to need manual review. This is normal.

---

**Module Complete** ✓

Next: **M12.3 - Self-Service Tenant Onboarding**
