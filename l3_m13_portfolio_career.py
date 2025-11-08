"""
Module 13.4: Portfolio Showcase & Career Launch

Core functionality for creating portfolio artifacts that attract senior engineering roles.
Implements architecture documentation, demo scripting, case study generation, and interview prep.

Key Functions:
- generate_architecture_doc(): Create comprehensive system documentation
- create_demo_script(): Build 15-minute problem→solution→impact narrative
- generate_case_study(): Write 2000+ word technical case study
- prepare_interview_responses(): Document talking points for system design questions
- evaluate_portfolio_decision(): Decision framework for portfolio investment
- track_application_metrics(): Monitor callback rates and conversion
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CareerLevel(Enum):
    """Target career levels with expected compensation (INR L/year)"""
    SENIOR_RAG_ENGINEER = "25-40L"
    STAFF_ML_ENGINEER = "40-60L"
    GENAI_CONSULTANT = "50-100L"
    FOUNDING_ENGINEER = "20-40L + equity"


class PortfolioArtifact(Enum):
    """Four core portfolio artifacts"""
    ARCHITECTURE_DOC = "architecture_documentation"
    DEMO_VIDEO = "demo_video_script"
    CASE_STUDY = "case_study_writeup"
    INTERVIEW_PREP = "interview_preparation"


@dataclass
class SystemMetrics:
    """Quantified system metrics for portfolio showcase"""
    tenants: int = 100
    queries_per_day: int = 50000
    p95_latency_seconds: float = 2.0
    uptime_percentage: float = 99.5
    time_reduction_percentage: int = 87
    arr_usd: int = 50000
    user_satisfaction: float = 4.8
    cost_per_query_usd: float = 0.008
    cost_savings_percentage: int = 60

    def passes_30_second_test(self) -> bool:
        """Verify metrics meet hiring manager attention threshold"""
        return (
            self.tenants >= 10 and
            self.queries_per_day >= 1000 and
            self.arr_usd > 0
        )


@dataclass
class TechDecision:
    """Trade-off analysis for technology choices"""
    choice: str
    alternatives_considered: List[str]
    rationale: str
    cost_monthly_usd: float
    when_to_choose_alternative: str

    def to_markdown(self) -> str:
        """Format as decision documentation"""
        return f"""
### Decision: {self.choice}

**Alternatives Considered:** {', '.join(self.alternatives_considered)}

**Rationale:** {self.rationale}

**Cost Impact:** ${self.cost_monthly_usd}/month

**When to Choose Alternative:** {when_to_choose_alternative}
"""


@dataclass
class ApplicationMetrics:
    """Track job application performance"""
    applications_sent: int = 0
    callbacks_received: int = 0
    interviews_completed: int = 0
    offers_received: int = 0

    def callback_rate(self) -> float:
        """Calculate callback percentage"""
        if self.applications_sent == 0:
            return 0.0
        return (self.callbacks_received / self.applications_sent) * 100

    def interview_to_offer_rate(self) -> float:
        """Calculate offer conversion percentage"""
        if self.interviews_completed == 0:
            return 0.0
        return (self.offers_received / self.interviews_completed) * 100

    def needs_portfolio_review(self) -> bool:
        """Alert if callback rate below threshold after 20 applications"""
        return self.applications_sent >= 20 and self.callback_rate() < 5.0

    def needs_interview_practice(self) -> bool:
        """Alert if offer rate below threshold after 5 interviews"""
        return self.interviews_completed >= 5 and self.interview_to_offer_rate() < 20.0


def generate_architecture_doc(
    system_name: str,
    metrics: SystemMetrics,
    tech_stack: List[str],
    decisions: List[TechDecision],
    diagram_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate comprehensive architecture documentation with trade-off analysis.

    Args:
        system_name: Project name (e.g., "Enterprise RAG SaaS")
        metrics: Quantified system metrics
        tech_stack: List of technologies used
        decisions: List of design decisions with rationale
        diagram_path: Optional path to architecture diagram

    Returns:
        Dict with 'overview', 'architecture', 'decisions', 'alternatives' sections

    Raises:
        ValueError: If metrics don't pass 30-second test
    """
    logger.info(f"Generating architecture doc for {system_name}")

    if not metrics.passes_30_second_test():
        raise ValueError(
            "Metrics don't meet hiring manager threshold. "
            "Need: >=10 tenants, >=1000 queries/day, ARR > 0"
        )

    # System overview (passes 30-second test)
    overview = f"""# {system_name}

**Problem:** Compliance teams spend 40 hours/week researching regulations

**Solution:** Multi-tenant RAG system reducing research time by {metrics.time_reduction_percentage}%

**Impact:**
- {metrics.tenants}+ tenants across financial institutions
- {metrics.queries_per_day:,} queries/day at <{metrics.p95_latency_seconds}s P95 latency
- ${metrics.arr_usd:,} ARR within 6 months
- {metrics.uptime_percentage}% uptime, {metrics.user_satisfaction}/5 satisfaction
- ${metrics.cost_per_query_usd} cost per query ({metrics.cost_savings_percentage}% savings from caching)

**Tech Stack:** {', '.join(tech_stack)}
"""

    # Architecture section
    architecture = f"""## Architecture

{f'![Architecture Diagram]({diagram_path})' if diagram_path else '*(Generate diagram using diagrams library or draw.io)*'}

### Components
- **API Layer:** Multi-tenant request routing with tenant isolation
- **Vector Database:** Semantic search with hybrid retrieval
- **LLM Services:** Orchestration for generation and re-ranking
- **Agentic Layer:** Tool-calling for regulation retrieval
- **Observability:** OpenTelemetry + Prometheus + Grafana

### Data Flow
1. User query → Tenant authentication & routing
2. Hybrid retrieval (vector + keyword) → Top-K documents
3. LLM re-ranking → Context assembly
4. Generation with citations → Response streaming
5. Metrics collection → Dashboard updates
"""

    # Design decisions
    decisions_md = "## Design Decisions\n\n"
    for decision in decisions:
        decisions_md += decision.to_markdown() + "\n"

    # Alternative approaches
    alternatives = """## Alternative Approaches Considered

### Self-Hosted Vector DB
**When to choose:** >500K queries/day where managed costs exceed $2K/month
**Trade-off:** Save $800-1500/month but add 40+ hours/month operational overhead

### Monolithic Architecture
**When to choose:** <10 tenants with simple requirements
**Trade-off:** Faster initial development but poor isolation and scaling limits

### Serverless-First
**When to choose:** Unpredictable traffic patterns or <100 queries/day
**Trade-off:** Lower idle costs but cold starts hurt P95 latency
"""

    logger.info("Architecture doc generated successfully")

    return {
        "overview": overview,
        "architecture": architecture,
        "decisions": decisions_md,
        "alternatives": alternatives
    }


def create_demo_script(
    problem_statement: str,
    solution_overview: str,
    demo_scenes: List[Dict[str, str]],
    impact_metrics: SystemMetrics
) -> Dict[str, Dict[str, str]]:
    """
    Create 15-minute demo video script with problem→solution→impact narrative.

    Args:
        problem_statement: Pain point hook (2 min)
        solution_overview: Architecture explanation (3 min)
        demo_scenes: List of {"timestamp": "5:00-7:00", "scene": "...", "narrative": "..."}
        impact_metrics: Business impact data

    Returns:
        Dict with timestamped script sections

    Note: NOT a feature walkthrough - narrative arc emphasizing problem-to-impact
    """
    logger.info("Creating demo video script (15-minute format)")

    if len(demo_scenes) != 3:
        logger.warning(f"Optimal demo has 3 scenes, got {len(demo_scenes)}")

    script = {
        "hook": {
            "timestamp": "0:00-2:00",
            "content": f"""**[HOOK: Problem Visual]**

{problem_statement}

**Visual:** Screen recording showing manual compliance research process
- Browser with 20+ tabs open
- Excel spreadsheet with regulation tracking
- Clock showing 2 hours elapsed

**Voiceover:** "{problem_statement[:100]}..."

**Goal:** Viewer feels pain immediately"""
        },

        "solution": {
            "timestamp": "2:00-5:00",
            "content": f"""**[SOLUTION: Architecture Overview]**

{solution_overview}

**Visual:** Clean architecture diagram with animated data flow
- API → Vector DB → LLM → Response
- Highlight: Multi-tenant isolation
- Highlight: <2s latency guarantee

**Voiceover:** "Here's how we solved this..."

**Goal:** Technical credibility established"""
        },

        "demo": {},

        "impact": {
            "timestamp": "12:00-14:00",
            "content": f"""**[IMPACT: Metrics Dashboard]**

**Visual:** Grafana dashboard showing:
- {impact_metrics.tenants}+ tenants (map visualization)
- {impact_metrics.queries_per_day:,} queries/day (live counter)
- {impact_metrics.p95_latency_seconds}s P95 latency (histogram)
- ${impact_metrics.arr_usd:,} ARR (revenue chart)

**Voiceover:** "Results in production..."
- {impact_metrics.time_reduction_percentage}% time reduction (40hrs → 15min)
- {impact_metrics.uptime_percentage}% uptime over 3 months
- {impact_metrics.user_satisfaction}/5 user satisfaction

**Goal:** Business impact quantified"""
        },

        "cta": {
            "timestamp": "14:00-15:00",
            "content": f"""**[CALL TO ACTION]**

**Visual:** GitHub repo + LinkedIn profile + email

**Voiceover:**
"If you're building compliance tools or RAG systems at scale, I'd love to connect.
Find the full architecture docs and code on GitHub: [your-username/enterprise-rag]
Reach out on LinkedIn: [your-profile]"

**Goal:** Clear next step for hiring managers"""
        }
    }

    # Add demo scenes
    for i, scene in enumerate(demo_scenes, 1):
        script["demo"][f"scene_{i}"] = scene

    logger.info(f"Demo script created with {len(demo_scenes)} scenes")

    return script


def generate_case_study(
    challenge: str,
    solution_approach: str,
    implementation_timeline: List[Dict[str, str]],
    results: SystemMetrics,
    learnings_and_regrets: List[str]
) -> str:
    """
    Generate 2000+ word technical case study for publication.

    Args:
        challenge: Specific problem description
        solution_approach: RAG architecture and multi-tenancy design
        implementation_timeline: List of {"month": "Month 1", "milestone": "..."}
        results: Quantified metrics
        learnings_and_regrets: Honest reflection on mistakes

    Returns:
        Formatted markdown case study (2000+ words)

    Note: Suitable for LinkedIn articles, Medium, Dev.to, personal blog
    """
    logger.info("Generating case study writeup")

    timeline_md = "\n".join([
        f"- **{item['month']}:** {item['milestone']}"
        for item in implementation_timeline
    ])

    regrets_md = "\n".join([f"- {regret}" for regret in learnings_and_regrets])

    case_study = f"""# Building an Enterprise RAG SaaS: From 0 to {results.tenants}+ Tenants

## The Challenge

{challenge}

Compliance teams at financial institutions face a unique problem: regulations change constantly,
span thousands of documents, and require expert interpretation. Manual research takes 40+ hours
per week per analyst, creating bottlenecks that slow business decisions.

## Solution Approach

{solution_approach}

We designed a multi-tenant RAG (Retrieval-Augmented Generation) system with three core principles:

1. **Tenant Isolation:** Each financial institution's data completely separated
2. **Sub-2s Latency:** Fast enough for interactive use (not batch processing)
3. **Cost Efficiency:** <$0.01 per query to enable scale

### Architecture Highlights

- **Hybrid Retrieval:** Combine vector similarity (semantic) + BM25 (keyword) for recall
- **LLM Re-ranking:** Use smaller model to re-rank top-100 before expensive generation
- **Smart Caching:** 60% of queries hit cache, saving $0.005 per cached query
- **Agentic Tool-Calling:** LLM decides when to call regulation DB vs generate from context

## Implementation Timeline

{timeline_md}

## Results

After 6 months in production:

### User Impact
- **{results.time_reduction_percentage}% time reduction:** Research drops from 2 hours to 15 minutes
- **{results.user_satisfaction}/5 satisfaction:** Based on 200+ user surveys
- **{results.tenants}+ tenants:** Across financial institutions

### Technical Performance
- **{results.queries_per_day:,} queries/day:** Sustained load with auto-scaling
- **{results.p95_latency_seconds}s P95 latency:** 95% of queries under 2 seconds
- **{results.uptime_percentage}% uptime:** Only 4 hours downtime in 90 days

### Business Metrics
- **${results.arr_usd:,} ARR:** Revenue within 6 months
- **${results.cost_per_query_usd} per query:** Including LLM, vector DB, compute
- **{results.cost_savings_percentage}% cost savings:** From caching strategy

## Key Learnings and Regrets

{regrets_md}

### What I'd Do Differently

**Earlier Load Testing:** We didn't load test until Month 5, discovering PostgreSQL bottlenecks
late in development. Cost us 2 weeks of scrambling to migrate to connection pooling + read replicas.

**Observability from Day 1:** Added OpenTelemetry in Month 4 when debugging latency spikes.
Should have been there from the start—wasted 30+ hours on blind debugging.

**Tenant Onboarding Automation:** Manual onboarding took 2 hours per tenant until Month 6.
Automated scripts reduced this to 10 minutes. Should have built automation at tenant #3, not #30.

## Technical Deep Dive: Hybrid Retrieval

*(Include 300-500 words on most interesting technical challenge)*

The biggest technical win was hybrid retrieval. Pure vector search had 70% recall on compliance
queries (missed exact regulation numbers). Pure keyword search was 60% (missed semantic matches).

Combining both with RRF (Reciprocal Rank Fusion) boosted recall to 91%.

[Include code snippet or architecture diagram]

## Conclusion

Building an enterprise RAG system taught me that production ML systems are 80% engineering
(multi-tenancy, observability, cost optimization) and 20% model selection. The hard parts
weren't choosing embeddings—they were tenant isolation, caching strategies, and operational excellence.

**If you're building similar systems, reach out:** [your-contact]

---

*Technologies used: FastAPI, PostgreSQL, Pinecone, OpenAI GPT-4, LangChain, OpenTelemetry,
Prometheus, Grafana, Docker, AWS ECS*
"""

    word_count = len(case_study.split())
    logger.info(f"Case study generated: {word_count} words")

    if word_count < 2000:
        logger.warning(f"Case study below 2000 words ({word_count}). Consider expanding technical deep dive.")

    return case_study


def prepare_interview_responses(
    system_name: str,
    metrics: SystemMetrics,
    tech_decisions: List[TechDecision],
    scale_bottlenecks: Dict[str, str],
    biggest_regret: str
) -> Dict[str, str]:
    """
    Document talking points for system design interview questions.

    Args:
        system_name: Project name
        metrics: System metrics for credibility
        tech_decisions: List of technology choices with rationale
        scale_bottlenecks: {"10x": "...", "100x": "..."} scaling limitations
        biggest_regret: Honest mistake for "what would you change?" question

    Returns:
        Dict mapping question types to structured 5-minute responses
    """
    logger.info("Preparing interview talking points")

    responses = {
        "architecture_walkthrough": f"""**5-Minute Structured Response:**

**[0:00-1:00] Problem Context**
"{system_name} solves compliance research for financial institutions. Teams spent 40hrs/week
on manual document search. We needed <2s interactive responses."

**[1:00-3:00] Architecture Overview**
"Multi-tenant RAG with 4 layers:
1. API: Tenant routing + auth
2. Retrieval: Hybrid vector + keyword search
3. LLM: Re-ranking + generation
4. Observability: OpenTelemetry tracing

Key design: Tenant isolation via separate namespaces in vector DB, shared infrastructure."

**[3:00-4:30] Trade-offs**
"Chose Pinecone (managed) over self-hosted Qdrant:
- Cost: $200/month vs $150 + operational overhead
- Decision: Managed made sense at <500K queries/day
- Would switch at higher scale when ops cost justifies"

**[4:30-5:00] Results**
"{metrics.tenants}+ tenants, {metrics.queries_per_day:,} queries/day, ${metrics.arr_usd:,} ARR in 6 months."
""",

        "scale_10x": f"""**What Changes at 10x Scale? (500K queries/day)**

**Bottleneck:** {scale_bottlenecks.get('10x', 'Vector DB query latency')}

**Solution:**
- Add read replicas for vector DB (3-5 replicas)
- Implement request coalescing for duplicate queries
- Move to edge caching (CloudFlare Workers)
- Cost increase: ~$800/month → $2,500/month

**Decision point:** At 10x, managed Pinecone still cheaper than self-hosted + ops overhead.
""",

        "scale_100x": f"""**What Changes at 100x Scale? (5M queries/day)**

**Bottleneck:** {scale_bottlenecks.get('100x', 'Managed vector DB costs ($20K/month)')}

**Solution:**
- Migrate to self-hosted Qdrant cluster (4-6 nodes)
- Dedicated embeddings cache (Redis cluster)
- Multi-region deployment for latency
- Cost: $20K managed → $8K self-hosted + $4K ops = $12K savings

**Decision point:** At 100x scale, operational overhead justified by $144K annual savings.
""",

        "biggest_regret": f"""**"What's Your Biggest Regret?"**

**Honest Answer:**
"{biggest_regret}"

**What I Learned:**
- Load testing is not optional—should be Day 1, not Month 5
- Observability pays for itself 10x in debugging time saved
- Automate onboarding at 3 users, not 30

**How It Makes Me Better:**
Now I build projects with "production checklist":
1. Load testing in CI/CD
2. OpenTelemetry from first commit
3. Automation-first for repetitive tasks
""",

        "multi_tenant_isolation": f"""**"How Do You Handle Multi-Tenant Security?"**

**Three Layers:**

1. **Authentication:** JWT with tenant_id claim, validated at API gateway
2. **Authorization:** Middleware enforces tenant_id on all queries (never trust client)
3. **Data Isolation:** Separate namespaces in Pinecone, PostgreSQL RLS policies

**Key Learning:**
Never filter by tenant_id in application code—use database-level Row-Level Security (RLS).
Prevents accidental data leaks from bugs.

**Testing:**
Automated tests attempt cross-tenant queries, should fail with 403.
"""
    }

    logger.info(f"Interview responses prepared for {len(responses)} question types")

    return responses


def evaluate_portfolio_decision(
    target_companies: int,
    current_callback_rate: float,
    job_search_timeline_months: int,
    referral_strength: str,  # "strong" | "weak" | "none"
    target_level: str  # "senior" | "staff" | "principal"
) -> Dict[str, any]:
    """
    Decision framework: Should you invest 40-80 hours in portfolio creation?

    Args:
        target_companies: Number of companies applying to
        current_callback_rate: Current callback percentage (0-100)
        job_search_timeline_months: Time available for job search
        referral_strength: Quality of referral network
        target_level: Career level targeting

    Returns:
        Dict with 'recommendation', 'reasoning', 'time_investment_hours', 'expected_callback_rate'

    Decision Logic (from script Section 10):
    - USE portfolio: 50+ companies, <10% callback, 3-6mo timeline, weak referrals, senior roles
    - SKIP portfolio: <10 companies, >30% callback, <4wk timeline, strong referrals, staff+ roles
    """
    logger.info(f"Evaluating portfolio decision: {target_companies} companies, "
                f"{current_callback_rate}% callback, {job_search_timeline_months}mo timeline")

    reasons_to_skip = []
    reasons_to_invest = []

    # Evaluate criteria
    if referral_strength == "strong":
        reasons_to_skip.append("Strong referrals bypass portfolio review (10x conversion)")

    if target_companies < 10:
        reasons_to_skip.append(f"Only {target_companies} companies—targeted outreach > generic portfolio")

    if target_level in ["staff", "principal"]:
        reasons_to_skip.append(f"{target_level.title()} roles value reputation > portfolio (focus on design docs, speaking)")

    if job_search_timeline_months < 1:
        reasons_to_skip.append(f"{job_search_timeline_months}mo insufficient for 40-80hr portfolio")

    if current_callback_rate > 30:
        reasons_to_skip.append(f"{current_callback_rate}% callback indicates interview skills issue, not application materials")

    # Reasons to invest
    if target_companies >= 50:
        reasons_to_invest.append(f"{target_companies} companies justifies portfolio investment (economies of scale)")

    if current_callback_rate < 10:
        reasons_to_invest.append(f"{current_callback_rate}% callback below threshold—portfolio can improve to 10-20%")

    if 3 <= job_search_timeline_months <= 6:
        reasons_to_invest.append(f"{job_search_timeline_months}mo timeline ideal for portfolio + applications")

    if referral_strength in ["weak", "none"]:
        reasons_to_invest.append("Weak referral network—portfolio provides differentiation")

    if target_level == "senior":
        reasons_to_invest.append("Senior roles value production portfolio (vs staff+ reputation)")

    # Decision logic
    if len(reasons_to_skip) > len(reasons_to_invest):
        recommendation = "SKIP_PORTFOLIO"
        expected_callback = current_callback_rate
        time_investment = 0
        alternative = "Focus on: 1) Warm introductions via network, 2) Targeted outreach (1hr per company), 3) Interview prep (LeetCode + system design)"
    else:
        recommendation = "INVEST_IN_PORTFOLIO"
        expected_callback = min(current_callback_rate * 2, 20)  # Portfolio typically 2x callback, cap at 20%
        time_investment = 48  # Minimum 48 hours for medium-quality portfolio
        alternative = "Hybrid: 7hr/week portfolio + 4hr/week networking + 3hr/week blog + 1hr/week open source"

    result = {
        "recommendation": recommendation,
        "reasoning": {
            "invest_reasons": reasons_to_invest,
            "skip_reasons": reasons_to_skip
        },
        "time_investment_hours": time_investment,
        "expected_callback_rate": expected_callback,
        "alternative_strategy": alternative,
        "monthly_cost_inr": "7000-17000" if recommendation == "INVEST_IN_PORTFOLIO" else "2000-5000"
    }

    logger.info(f"Decision: {recommendation} (expected callback: {expected_callback}%)")

    return result


def track_application_metrics(metrics: ApplicationMetrics) -> Dict[str, any]:
    """
    Monitor job application performance and trigger alerts.

    Args:
        metrics: ApplicationMetrics with current counts

    Returns:
        Dict with 'callback_rate', 'offer_rate', 'alerts', 'recommendations'

    Alert Thresholds (from script Section 9):
    - <5% callback after 20 applications: Portfolio/resume needs review
    - <20% offer rate after 5 finals: Interview performance issue
    """
    logger.info(f"Tracking metrics: {metrics.applications_sent} apps, "
                f"{metrics.callbacks_received} callbacks, {metrics.offers_received} offers")

    alerts = []
    recommendations = []

    callback_rate = metrics.callback_rate()
    offer_rate = metrics.interview_to_offer_rate()

    # Check alert thresholds
    if metrics.needs_portfolio_review():
        alerts.append("🚨 Callback rate <5% after 20 applications")
        recommendations.append("Review portfolio and resume—30-second test failing")
        recommendations.append("Get feedback from hiring managers or recruiters")

    if metrics.needs_interview_practice():
        alerts.append("🚨 Offer rate <20% after 5 final interviews")
        recommendations.append("Interview performance issue (not application materials)")
        recommendations.append("Focus on: LeetCode (2hr/day), system design (1hr/day), behavioral prep")

    # Positive feedback
    if callback_rate >= 10 and metrics.applications_sent >= 10:
        recommendations.append("✅ Callback rate healthy (10%+)—portfolio working")

    if offer_rate >= 25 and metrics.interviews_completed >= 4:
        recommendations.append("✅ Offer rate strong (25%+)—interview skills solid")

    # Expected timeline
    if metrics.applications_sent >= 50:
        expected_timeline = "3-4 months from portfolio start to offer (typical mid/senior in India)"
    else:
        expected_timeline = "Send 50+ applications for statistical significance"

    result = {
        "callback_rate": round(callback_rate, 1),
        "offer_rate": round(offer_rate, 1),
        "alerts": alerts,
        "recommendations": recommendations,
        "expected_timeline": expected_timeline,
        "target_callback_rate": 10,  # With portfolio
        "target_offer_rate": 20  # For senior roles
    }

    logger.info(f"Metrics summary: {callback_rate}% callback, {offer_rate}% offer rate")

    return result


def passes_30_second_test(readme_content: str) -> Tuple[bool, List[str]]:
    """
    Validate GitHub README passes hiring manager 30-second test.

    Args:
        readme_content: First 100 words of README

    Returns:
        Tuple of (passes: bool, missing_elements: List[str])

    Requirements (from script Section 8, Failure 5):
    - Problem statement
    - Impact metric (e.g., "87% reduction")
    - Scale indicator (e.g., "100+ tenants", "$50K ARR")
    """
    logger.info("Validating README 30-second test")

    first_100_words = ' '.join(readme_content.split()[:100]).lower()

    missing = []

    # Check for problem statement keywords
    problem_keywords = ['problem', 'challenge', 'pain', 'spent', 'manual', 'slow']
    if not any(kw in first_100_words for kw in problem_keywords):
        missing.append("Problem statement (e.g., 'Compliance teams spent 40hrs/week...')")

    # Check for impact metrics
    impact_keywords = ['%', 'reduction', 'increase', 'saved', 'improved']
    if not any(kw in first_100_words for kw in impact_keywords):
        missing.append("Impact metric (e.g., '87% time reduction')")

    # Check for scale indicators
    scale_keywords = ['tenants', 'users', 'queries', 'arr', '$', '₹', 'revenue']
    if not any(kw in first_100_words for kw in scale_keywords):
        missing.append("Scale indicator (e.g., '100+ tenants, $50K ARR')")

    passes = len(missing) == 0

    if passes:
        logger.info("✅ README passes 30-second test")
    else:
        logger.warning(f"❌ README fails 30-second test. Missing: {', '.join(missing)}")

    return passes, missing


# CLI Usage Examples
if __name__ == "__main__":
    print("=== Module 13.4: Portfolio Showcase & Career Launch ===\n")

    # Example 1: Generate architecture documentation
    print("1. Architecture Documentation")
    print("-" * 50)

    metrics = SystemMetrics(
        tenants=100,
        queries_per_day=50000,
        p95_latency_seconds=1.8,
        arr_usd=50000
    )

    decisions = [
        TechDecision(
            choice="Pinecone (Managed Vector DB)",
            alternatives_considered=["Qdrant", "Weaviate", "Milvus"],
            rationale="Managed service reduces ops overhead at <500K queries/day",
            cost_monthly_usd=200,
            when_to_choose_alternative="Choose self-hosted at >500K queries/day ($20K/mo managed costs)"
        )
    ]

    try:
        arch_doc = generate_architecture_doc(
            system_name="Enterprise RAG SaaS",
            metrics=metrics,
            tech_stack=["FastAPI", "PostgreSQL", "Pinecone", "OpenAI GPT-4", "LangChain"],
            decisions=decisions
        )
        print(arch_doc["overview"][:200] + "...")
        print("✅ Architecture doc generated\n")
    except Exception as e:
        logger.error(f"Architecture doc failed: {e}")

    # Example 2: Evaluate portfolio decision
    print("2. Portfolio Investment Decision")
    print("-" * 50)

    decision = evaluate_portfolio_decision(
        target_companies=60,
        current_callback_rate=4.0,
        job_search_timeline_months=4,
        referral_strength="weak",
        target_level="senior"
    )

    print(f"Recommendation: {decision['recommendation']}")
    print(f"Expected callback rate: {decision['expected_callback_rate']}%")
    print(f"Time investment: {decision['time_investment_hours']} hours")
    print(f"Reasons to invest: {len(decision['reasoning']['invest_reasons'])}")
    print()

    # Example 3: Track application metrics
    print("3. Application Performance Tracking")
    print("-" * 50)

    app_metrics = ApplicationMetrics(
        applications_sent=25,
        callbacks_received=2,
        interviews_completed=1,
        offers_received=0
    )

    tracking = track_application_metrics(app_metrics)
    print(f"Callback rate: {tracking['callback_rate']}% (target: {tracking['target_callback_rate']}%)")
    print(f"Alerts: {len(tracking['alerts'])}")
    for alert in tracking['alerts']:
        print(f"  {alert}")
    print()

    # Example 4: Validate README
    print("4. README 30-Second Test")
    print("-" * 50)

    sample_readme = """
    # Enterprise RAG SaaS

    Compliance teams spend 40 hours/week researching regulations. This multi-tenant RAG system
    reduces research time by 87% (40hrs → 15min). Currently serving 100+ tenants with $50K ARR,
    processing 50,000 queries/day at <2s latency.
    """

    passes, missing = passes_30_second_test(sample_readme)
    print(f"Passes test: {passes}")
    if missing:
        print(f"Missing elements: {', '.join(missing)}")
    else:
        print("✅ All elements present")

    print("\n" + "=" * 50)
    print("For full notebook walkthrough: L3_M13_Portfolio_Showcase_Career_Launch.ipynb")
