# Module 13.3: Launch Preparation & Marketing

**Enterprise RAG SaaS - Go-to-Market Strategy and Customer Acquisition**

Transform your production-ready multi-tenant RAG SaaS into a market-ready business offering through strategic launch preparation, pricing strategy, and customer acquisition planning.

## Overview

This module provides the tools and frameworks to:
- Create high-converting value propositions tailored to specific customer segments
- Design 3-tier pricing strategies based on value metrics (not guesswork)
- Build comprehensive go-to-market (GTM) plans with customer acquisition channels
- Analyze conversion funnels and calculate unit economics (CAC/LTV)
- Track attribution across marketing channels with UTM parameters

**Key Learning:** Technical excellence alone doesn't create revenue. This module bridges the gap between "I have a working product" and "I have a sustainable business."

---

## Quickstart

### 1. Installation

```bash
# Clone repository
cd ccc_l3_aug_practical

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env with your analytics keys (Google Analytics, Mixpanel)
```

### 2. Run the API

```powershell
# Windows (PowerShell) - recommended
.\scripts\run_api.ps1

# Or manually
$env:PYTHONPATH = $PWD
uvicorn app:app --reload

# API available at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

```bash
# Linux/Mac
export PYTHONPATH=$PWD
uvicorn app:app --reload
```

### 3. Explore the Notebook

```bash
# Launch Jupyter Lab (recommended for L3)
jupyter lab notebooks/L3_M13_Launch_Preparation_Marketing.ipynb

# Or Jupyter Notebook
jupyter notebook notebooks/L3_M13_Launch_Preparation_Marketing.ipynb
```

### 4. Run Tests

```powershell
# Windows (PowerShell) - recommended
.\scripts\run_tests.ps1

# Or manually
$env:PYTHONPATH = $PWD
pytest -q tests/
```

```bash
# Linux/Mac
export PYTHONPATH=$PWD
pytest -q tests/
```

---

## File Structure

```
.
├── src/
│   └── l3_m13_launch_prep_marketing/
│       └── __init__.py              # Core business logic (all classes)
├── notebooks/
│   └── L3_M13_Launch_Preparation_Marketing.ipynb
├── tests/
│   └── test_m13_launch_prep_marketing.py
├── configs/
│   └── example.json                 # Sample ICP/pricing config
├── scripts/
│   ├── run_api.ps1                  # Windows: Start API server
│   └── run_tests.ps1                # Windows: Run tests
├── app.py                           # FastAPI application (thin wrapper)
├── config.py                        # Configuration management
├── requirements.txt
├── .env.example                     # Environment variables template
├── .gitignore
├── example_data_*.{json,csv}        # Sample data files
├── LICENSE
└── README.md
```

**Key Points:**
- **Business logic:** All in `src/l3_m13_launch_prep_marketing/__init__.py`
- **API layer:** Thin `app.py` that imports from `src/`
- **Tests:** Standard `tests/` directory for pytest discovery
- **Notebooks:** Organized in `notebooks/` subdirectory
- **Scripts:** PowerShell scripts for Windows-first development

---

## Environment Variables

The module works in **offline/limited mode** by default—no external API keys required for core functionality (pricing, CAC/LTV, funnels). Optional analytics integrations enhance tracking but aren't mandatory.

**Optional Analytics Keys** (`.env` file):

```bash
# Google Analytics 4 (optional - for landing page tracking)
GOOGLE_ANALYTICS_MEASUREMENT_ID=G-XXXXXXXXXX

# Mixpanel (optional - for funnel analysis)
MIXPANEL_PROJECT_TOKEN=your_mixpanel_project_token_here
MIXPANEL_API_SECRET=your_mixpanel_api_secret_here

# Mailchimp (optional - for email campaigns)
MAILCHIMP_API_KEY=your_mailchimp_api_key_here
MAILCHIMP_SERVER_PREFIX=us19

# HubSpot (optional - for CRM integration)
HUBSPOT_API_KEY=your_hubspot_api_key_here

# Landing page URLs
LANDING_PAGE_URL=https://yourcompany.com
PRODUCT_APP_URL=https://app.yourcompany.com

# OFFLINE mode (set to "true" to disable external calls)
OFFLINE=false
```

**What works without keys:**
- ✅ Value proposition validation
- ✅ Pricing calculations (COGS-based + value-based)
- ✅ GTM strategy selection
- ✅ Conversion funnel analysis
- ✅ CAC/LTV calculations
- ✅ UTM URL generation

**What requires keys:**
- ❌ Live analytics tracking (GA4, Mixpanel)
- ❌ Email campaign automation (Mailchimp)
- ❌ CRM integrations (HubSpot)

---

## How It Works

### Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAUNCH PREPARATION FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │ Value Prop   │─────▶│  Pricing     │─────▶│   GTM Plan   │ │
│  │ Creation     │      │  Strategy    │      │   Creation   │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                      │                      │        │
│         │                      │                      │        │
│         ▼                      ▼                      ▼        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              LANDING PAGE LAUNCH                         │ │
│  │  • Hero with specific value prop                         │ │
│  │  • 3-tier pricing (Starter/Pro/Enterprise)               │ │
│  │  • CTA → Signup flow                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         CUSTOMER ACQUISITION CHANNELS                    │ │
│  │  Tier 1: LinkedIn Organic, Cold Email, Product Hunt     │ │
│  │  Tier 2: Google Ads, SEO Content                        │ │
│  │  Tier 3: Conferences, Partnerships                      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │            ANALYTICS & OPTIMIZATION                      │ │
│  │  • Conversion funnel (Visitor→Trial→Paid)               │ │
│  │  • CAC/LTV calculation                                   │ │
│  │  • Attribution analysis (UTM tracking)                   │ │
│  │  • Identify bottlenecks → Optimize                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Value Proposition Builder** (`ValueProposition`)
   - Structured framework: Target customer + Pain + Solution + Differentiator
   - Validation against vague buzzwords
   - Headline/subheadline generation

2. **Pricing Calculator** (`PricingCalculator`)
   - COGS-based minimum pricing (ensure profitability)
   - Value-based pricing (capture 20-30% of value delivered)
   - 3-tier structure with ROI calculations

3. **GTM Strategy Selector** (`GTMStrategySelector`)
   - Recommends motion based on ACV, market size, product complexity
   - Self-service vs. Direct sales vs. Partner-led vs. Freemium

4. **Conversion Funnel Analyzer** (`ConversionFunnelAnalyzer`)
   - Tracks: Visitor → Signup → Activation → Paid
   - Compares against industry benchmarks
   - Identifies weakest funnel step

5. **CAC/LTV Calculator** (`CAC_LTV_Calculator`)
   - Customer Acquisition Cost calculation
   - Lifetime Value based on MRR, margin, churn
   - Unit economics health check (target LTV:CAC > 3:1)

6. **UTM Tracker** (`UTMTracker`)
   - Generate campaign URLs with tracking parameters
   - Parse attribution data from customer records
   - Identify best-performing channels

---

## Common Failures & Fixes

Based on the module script, here are the 5 most common launch failures:

### Failure #1: Unclear Value Proposition (85% bounce rate)

**Symptoms:**
- Visitors spend <10 seconds on page
- High bounce rate (>80%)
- Low signup conversion (<2%)

**Root Cause:**
Vague value prop like "AI-powered document search for modern teams" doesn't communicate who it's for or what problem it solves.

**Fix:**
```python
# Before (vague)
vp = ValueProposition(
    target_customer="modern teams",
    pain_point="need better search",
    solution_outcome="AI-powered search",
    unique_differentiator="semantic search"
)

# After (specific)
vp = ValueProposition(
    target_customer="Financial services compliance teams",
    pain_point="waste 15+ hours per week manually searching regulatory documents",
    solution_outcome="Find any regulatory document in seconds",
    unique_differentiator="AI-powered semantic search with audit-ready citations",
    quantified_benefit="Save 12 hours/week on average"
)
```

**Prevention:** Show landing page to 10 people in your ICP. They should answer "What does this do?" in 10 seconds.

---

### Failure #2: Wrong Pricing (0% conversion or negative margins)

**Symptoms:**
- Scenario A: Lots of trials, zero paid conversions (priced too high)
- Scenario B: Paying customers but losing money (priced too low)

**Root Cause:**
Pricing chosen arbitrarily ("$99 sounds reasonable") without calculating COGS or value delivered.

**Fix:**
```python
# Calculate minimum viable price
calc = PricingCalculator()

# Step 1: Calculate COGS-based minimum
cogs = 70  # Pinecone + OpenAI + infra per customer/month
min_price = calc.calculate_minimum_price(cogs, target_gross_margin=0.67)
# Result: ~$212/month minimum

# Step 2: Calculate value-based price
value_price = calc.calculate_value_based_price(
    hours_saved_per_week=10,
    hourly_labor_cost=100,
    value_capture_rate=0.25  # Capture 25% of $4,000/month value
)
# Result: ~$1,000/month

# Step 3: Use higher of the two
recommended_price = max(min_price, value_price)
```

**Prevention:** Calculate COGS before launch. Price at minimum 3x COGS for healthy margins.

---

### Failure #3: Targeting Wrong Customer Segment

**Symptoms:**
- High traffic, good signups, terrible activation (<20%)
- Trial users don't match your ICP (students, personal emails, wrong job titles)

**Root Cause:**
Marketing to "anyone who works with documents" instead of specific ICP.

**Fix:**
```python
# Tighten ICP definition
icp = IdealCustomerProfile(
    industry="Financial services (banking, wealth management)",
    company_size_min=50,
    company_size_max=5000,
    geography=["United States"],
    job_titles=["Chief Compliance Officer", "Head of Compliance"],
    department="Compliance, Risk, Legal",
    pain_intensity="15+ hours/week wasted on document search",
    budget_range_annual=(10000, 50000),
    tech_maturity="Using cloud tools, open to AI"
)

# Update targeting on all channels
# LinkedIn: Only target these job titles + industries
# Google Ads: Add negative keywords ("free", "student", "personal")
# Cold email: Filter email list to match ALL ICP criteria
```

**Prevention:** Interview 10 paying customers. Ask: "Where do you look for solutions? What do you search for?" Only market there.

---

### Failure #4: Broken Signup Flow (75% error rate)

**Symptoms:**
- People click "Start Trial" but don't complete signup
- High drop-off between form submission and success page

**Root Cause:**
Technical errors: Stripe webhook not configured, timeout during tenant provisioning, email verification fails.

**Fix:**
```bash
# Add error tracking
pip install sentry-sdk

# Test signup flow 10 times before launch
curl -X POST "https://yourapp.com/api/signup" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "Test123"}'

# Verify each step:
# 1. User created in database ✓
# 2. Stripe customer created ✓
# 3. Tenant provisioned ✓
# 4. Welcome email sent ✓
# 5. User can log in ✓
```

**Prevention:** Set up monitoring for signup completion rate. Alert if <80%.

---

### Failure #5: No Analytics (Flying Blind)

**Symptoms:**
- You don't know which marketing channel is working
- Can't identify funnel bottlenecks
- Making decisions on gut feel

**Root Cause:**
Didn't implement UTM tracking or conversion tracking.

**Fix:**
```python
# Generate UTM URLs for all campaigns
tracker = UTMTracker()

linkedin_url = tracker.generate_utm_url(
    base_url="https://yoursite.com",
    source="linkedin",
    medium="cpc",
    campaign="compliance_q4_2024"
)
# https://yoursite.com?utm_source=linkedin&utm_medium=cpc&utm_campaign=compliance_q4_2024

# Analyze attribution after 30 days
attribution = tracker.parse_attribution_data(customer_records)
# Shows: Twitter = 15 customers, LinkedIn = 3 customers
# → Double down on Twitter, pause LinkedIn
```

**Prevention:** Set up GA4 + Mixpanel before launch. Track: page_view, signup, activation, purchase.

---

## Decision Card

**✅ BENEFIT:**
Scalable customer acquisition without hiring sales team. Well-designed landing page converts 5-10% of visitors to trials, 15-25% of trials to paid customers. Revenue scales with marketing spend, not headcount.

**❌ LIMITATION:**
Doesn't work for enterprise sales (>$10K/year contracts) where buyers expect sales-assisted motion. Self-service landing pages generate zero enterprise leads. Also requires 10-20 hours/week marketing effort—landing page alone generates ~10-50 organic visitors in first 6 months.

**💰 COST:**
Time: 40 hours to build landing page, pricing, GTM plan, analytics (1-2 weeks). Money: $14-100/month hosting, $0-200/month analytics, $200-2,000/month paid ads. Ongoing: 10 hours/week for content and optimization. Cheaper than sales reps ($100K-150K/year) but requires strong product-led growth.

**🤔 USE WHEN:**
Targeting SMB/mid-market (50-500 employees) willing to pay $100-1,000/month for clear value prop demonstrable in 14-day trial. Product has low onboarding friction (<30 min to first value), strong differentiation, large addressable market (>10,000 customers). You have time for marketing (10+ hours/week) or budget for ads ($500+/month).

**🚫 AVOID WHEN:**
Targeting enterprise (>$10K/year)—use direct sales. Product requires complex setup (>30 min to understand)—use demo-first sales. You have <5 hours/week for marketing and <$500/month budget—build audience first or offer services instead of SaaS. Pricing <$50/month—unit economics won't support CAC, use freemium.

---

## Troubleshooting

### Issue: "Config validation shows missing items"

**Solution:** This is expected. The module works without external services (GA4, Mixpanel, Mailchimp). Analytics integrations are optional. Core calculations (pricing, CAC/LTV, funnel analysis) work offline.

```bash
# Check config status
python config.py

# Expected output if keys missing:
# ⚠️  Google Analytics not configured
# ⚠️  Mixpanel not configured
# This is OK for local development and planning
```

---

### Offline/Limited Mode

**The module runs in a limited, "degraded" mode if analytics API keys are not set in `.env`.**

The `config.py` file will return `None` for any unconfigured client, and the `app.py` logic will return a `{"skipped": True, "reason": "Service not initialized..."}` response for API endpoints that require external services.

**Core functionality works completely offline:**
- All pricing calculations (COGS-based, value-based, tier design)
- Value proposition validation
- GTM strategy recommendations
- Conversion funnel analysis
- CAC/LTV calculations
- UTM URL generation

**What's disabled in offline mode:**
- Live analytics event tracking (GA4, Mixpanel)
- Email campaign automation (Mailchimp)
- CRM data sync (HubSpot)

**To enable offline mode explicitly:**
```bash
# Set in .env file
OFFLINE=true
```

Or run notebook/scripts with:
```bash
export OFFLINE=true
jupyter lab notebooks/L3_M13_Launch_Preparation_Marketing.ipynb
```

The notebook will display: `⚠️ Running in OFFLINE mode — External API calls will be skipped`

---

### Issue: "Low conversion rates in funnel analysis"

**Diagnostic:**
```python
# Analyze funnel
analyzer = ConversionFunnelAnalyzer()
funnel = analyzer.calculate_funnel_metrics(
    visitors=1000,
    signups=30,  # 3% conversion
    activated=10,  # 33% activation
    paid=2  # 20% paid conversion
)

print(funnel['weakest_step'])  # Identifies bottleneck
print(funnel['issues'])  # Specific recommendations
```

**Fix based on weakest step:**
- **Signup low (<5%):** Value prop unclear or landing page slow
- **Activation low (<50%):** Onboarding too complex or product UX confusing
- **Paid conversion low (<20%):** Pricing too high or trial doesn't demonstrate value

---

### Issue: "CAC exceeds LTV (unsustainable unit economics)"

**Diagnostic:**
```python
calc = CAC_LTV_Calculator()
unit_econ = calc.calculate_unit_economics(cac=750, ltv=600)
# Result: LTV:CAC ratio = 0.8 (CRITICAL)
```

**Fix:**
1. **Reduce CAC:** Focus on organic channels (LinkedIn posts, cold email) instead of paid ads
2. **Increase LTV:** Reduce churn (improve product), increase MRR (upsell to higher tiers)
3. **Pivot GTM:** If ratio stays <1.0, self-service doesn't work for your market—switch to direct sales

---

## Next Module

**Module 13.4: Portfolio Showcase & Career Launch**

- Create 15-minute demo video showcasing your RAG SaaS
- Write case study with real metrics and customer impact
- Build portfolio site highlighting Level 1-3 projects
- Prepare for senior engineer interviews with behavioral + technical prep

**Link:** [M13.4 - Portfolio Showcase](./M13_4_Portfolio_Showcase.md) *(coming soon)*

---

## Example Data

Sample files included for testing and exploration:

- `example_data_icp.json` - Ideal Customer Profile for financial services compliance
- `example_data_gtm_plan.json` - 90-day go-to-market plan with channels and metrics
- `example_data_pricing.json` - 3-tier pricing strategy with ROI justifications
- `example_data_customers.csv` - Sample customer records for attribution analysis

---

## License

Part of CCC Level 3 curriculum - Enterprise RAG SaaS capstone project.

---

## Support

**Issues:** Open a GitHub issue for bugs or questions
**Office Hours:** Tuesdays 6 PM ET for live feedback on landing pages
**Community:** Join #level3-launches Slack channel for peer reviews

---

**🚀 Ready to launch?** Start with the notebook to build your value proposition and pricing strategy, then move to the API for integration with your SaaS backend.
