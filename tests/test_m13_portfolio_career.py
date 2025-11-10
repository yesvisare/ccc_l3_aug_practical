"""
Smoke tests for Module 13.4: Portfolio Showcase & Career Launch

Minimal tests to verify:
1. Config loads properly
2. Core functions return expected shapes
3. No external API dependencies (all runs locally)
"""

import pytest
import json
from pathlib import Path

# Import modules
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.l3_m13_portfolio_career import (
    SystemMetrics, TechDecision, ApplicationMetrics,
    generate_architecture_doc, create_demo_script, generate_case_study,
    prepare_interview_responses, evaluate_portfolio_decision,
    track_application_metrics, passes_30_second_test,
    CareerLevel, PortfolioArtifact
)
import src.l3_m13_portfolio_career as portfolio
from config import PortfolioConfig, OutputPaths, validate_config


def test_config_loads():
    """Test that configuration loads successfully"""
    assert PortfolioConfig.TOTAL_MINIMUM_HOURS > 0
    assert PortfolioConfig.TARGET_CALLBACK_RATE > 0
    assert validate_config() is True


def test_output_paths_exist():
    """Test that output directory can be created"""
    assert OutputPaths.OUTPUT_DIR.exists()
    assert OutputPaths.OUTPUT_DIR.is_dir()


def test_system_metrics_validation():
    """Test SystemMetrics dataclass and 30-second test"""
    metrics = portfolio.SystemMetrics(
        tenants=100,
        queries_per_day=50000,
        p95_latency_seconds=1.8,
        arr_usd=50000
    )

    assert metrics.tenants == 100
    assert metrics.passes_30_second_test() is True

    # Test failure case
    bad_metrics = portfolio.SystemMetrics(tenants=5, queries_per_day=100, arr_usd=0)
    assert bad_metrics.passes_30_second_test() is False


def test_tech_decision_markdown():
    """Test TechDecision formatting"""
    decision = portfolio.TechDecision(
        choice="Pinecone",
        alternatives_considered=["Qdrant", "Weaviate"],
        rationale="Managed service reduces ops overhead",
        cost_monthly_usd=200,
        when_to_choose_alternative="At >500K queries/day"
    )

    markdown = decision.to_markdown()
    assert "Pinecone" in markdown
    assert "Qdrant" in markdown
    assert "$200" in markdown


def test_application_metrics_calculations():
    """Test ApplicationMetrics callback and offer rate calculations"""
    metrics = portfolio.ApplicationMetrics(
        applications_sent=50,
        callbacks_received=8,
        interviews_completed=6,
        offers_received=2
    )

    assert metrics.callback_rate() == 16.0
    assert metrics.interview_to_offer_rate() == pytest.approx(33.33, rel=0.1)
    assert metrics.needs_portfolio_review() is False
    assert metrics.needs_interview_practice() is False

    # Test alert conditions
    bad_metrics = portfolio.ApplicationMetrics(
        applications_sent=25,
        callbacks_received=1,
        interviews_completed=1,
        offers_received=0
    )
    assert bad_metrics.needs_portfolio_review() is True


def test_generate_architecture_doc():
    """Test architecture documentation generation"""
    metrics = portfolio.SystemMetrics(
        tenants=100,
        queries_per_day=50000,
        arr_usd=50000
    )

    decisions = [
        portfolio.TechDecision(
            choice="Pinecone",
            alternatives_considered=["Qdrant"],
            rationale="Test rationale",
            cost_monthly_usd=200,
            when_to_choose_alternative="Test alternative"
        )
    ]

    arch_doc = portfolio.generate_architecture_doc(
        system_name="Test System",
        metrics=metrics,
        tech_stack=["FastAPI", "PostgreSQL"],
        decisions=decisions
    )

    assert "overview" in arch_doc
    assert "architecture" in arch_doc
    assert "decisions" in arch_doc
    assert "Test System" in arch_doc["overview"]


def test_generate_architecture_doc_fails_without_metrics():
    """Test that architecture doc generation fails with bad metrics"""
    bad_metrics = portfolio.SystemMetrics(tenants=5, queries_per_day=100, arr_usd=0)

    with pytest.raises(ValueError, match="30-second test"):
        portfolio.generate_architecture_doc(
            system_name="Test",
            metrics=bad_metrics,
            tech_stack=[],
            decisions=[]
        )


def test_create_demo_script():
    """Test demo script generation"""
    metrics = portfolio.SystemMetrics()

    demo_scenes = [
        {"timestamp": "5:00-7:00", "scene": "Scene 1", "narrative": "Test"},
        {"timestamp": "7:00-10:00", "scene": "Scene 2", "narrative": "Test"},
        {"timestamp": "10:00-12:00", "scene": "Scene 3", "narrative": "Test"}
    ]

    script = portfolio.create_demo_script(
        problem_statement="Test problem",
        solution_overview="Test solution",
        demo_scenes=demo_scenes,
        impact_metrics=metrics
    )

    assert "hook" in script
    assert "solution" in script
    assert "demo" in script
    assert "impact" in script
    assert "cta" in script
    assert len(script["demo"]) == 3


def test_generate_case_study():
    """Test case study generation"""
    metrics = portfolio.SystemMetrics()

    timeline = [
        {"month": "Month 1", "milestone": "MVP"},
        {"month": "Month 2", "milestone": "Multi-tenancy"}
    ]

    regrets = [
        "Didn't load test early",
        "Added observability late"
    ]

    case_study = portfolio.generate_case_study(
        challenge="Test challenge",
        solution_approach="Test solution",
        implementation_timeline=timeline,
        results=metrics,
        learnings_and_regrets=regrets
    )

    assert len(case_study) > 500  # Should be substantial
    assert "Month 1" in case_study
    assert "Didn't load test" in case_study

    # Check word count
    word_count = len(case_study.split())
    assert word_count >= 1500  # Should be close to 2000+ words


def test_prepare_interview_responses():
    """Test interview preparation generation"""
    metrics = portfolio.SystemMetrics()

    decisions = [
        portfolio.TechDecision(
            choice="Pinecone",
            alternatives_considered=["Qdrant"],
            rationale="Test",
            cost_monthly_usd=200,
            when_to_choose_alternative="Test"
        )
    ]

    responses = portfolio.prepare_interview_responses(
        system_name="Test System",
        metrics=metrics,
        tech_decisions=decisions,
        scale_bottlenecks={"10x": "Vector DB latency", "100x": "Managed costs"},
        biggest_regret="Didn't load test early"
    )

    assert "architecture_walkthrough" in responses
    assert "scale_10x" in responses
    assert "scale_100x" in responses
    assert "biggest_regret" in responses
    assert "multi_tenant_isolation" in responses


def test_evaluate_portfolio_decision_invest():
    """Test portfolio decision evaluation - invest scenario"""
    decision = portfolio.evaluate_portfolio_decision(
        target_companies=60,
        current_callback_rate=4.0,
        job_search_timeline_months=4,
        referral_strength="weak",
        target_level="senior"
    )

    assert decision["recommendation"] == "INVEST_IN_PORTFOLIO"
    assert decision["expected_callback_rate"] > 4.0
    assert decision["time_investment_hours"] > 0


def test_evaluate_portfolio_decision_skip():
    """Test portfolio decision evaluation - skip scenario"""
    decision = portfolio.evaluate_portfolio_decision(
        target_companies=8,
        current_callback_rate=35.0,
        job_search_timeline_months=2,
        referral_strength="strong",
        target_level="senior"
    )

    assert decision["recommendation"] == "SKIP_PORTFOLIO"
    assert decision["time_investment_hours"] == 0


def test_track_application_metrics():
    """Test application metrics tracking"""
    metrics = portfolio.ApplicationMetrics(
        applications_sent=50,
        callbacks_received=8,
        interviews_completed=6,
        offers_received=2
    )

    tracking = portfolio.track_application_metrics(metrics)

    assert "callback_rate" in tracking
    assert "offer_rate" in tracking
    assert "alerts" in tracking
    assert "recommendations" in tracking
    assert tracking["callback_rate"] == 16.0


def test_passes_30_second_test_good():
    """Test README validation - good example"""
    good_readme = """
    # Enterprise RAG SaaS

    Compliance teams spend 40 hours/week researching regulations. This multi-tenant RAG system
    reduces research time by 87% (40hrs → 15min). Currently serving 100+ tenants with $50K ARR.
    """

    passes, missing = portfolio.passes_30_second_test(good_readme)
    assert passes is True
    assert len(missing) == 0


def test_passes_30_second_test_bad():
    """Test README validation - bad example"""
    bad_readme = """
    # Enterprise RAG SaaS

    A RAG system built with Python.

    ## Installation
    pip install -r requirements.txt
    """

    passes, missing = portfolio.passes_30_second_test(bad_readme)
    assert passes is False
    assert len(missing) > 0


def test_example_data_loads():
    """Test that example_data.json loads properly"""
    example_path = Path(__file__).parent / "example_data.json"
    assert example_path.exists()

    with open(example_path, 'r') as f:
        data = json.load(f)

    assert "system_metrics" in data
    assert "tech_stack" in data
    assert "tech_decisions" in data
    assert "implementation_timeline" in data

    # Verify metrics can be instantiated
    metrics = portfolio.SystemMetrics(**data['system_metrics'])
    assert metrics.tenants > 0


def test_career_levels_defined():
    """Test that career levels are properly defined"""
    assert portfolio.CareerLevel.SENIOR_RAG_ENGINEER.value == "25-40L"
    assert portfolio.CareerLevel.STAFF_ML_ENGINEER.value == "40-60L"
    assert portfolio.CareerLevel.GENAI_CONSULTANT.value == "50-100L"
    assert portfolio.CareerLevel.FOUNDING_ENGINEER.value == "20-40L + equity"


def test_portfolio_artifacts_enum():
    """Test that portfolio artifacts are properly defined"""
    assert portfolio.PortfolioArtifact.ARCHITECTURE_DOC.value == "architecture_documentation"
    assert portfolio.PortfolioArtifact.DEMO_VIDEO.value == "demo_video_script"
    assert portfolio.PortfolioArtifact.CASE_STUDY.value == "case_study_writeup"
    assert portfolio.PortfolioArtifact.INTERVIEW_PREP.value == "interview_preparation"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
