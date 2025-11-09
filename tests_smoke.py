"""
Module 7.2: Application Performance Monitoring - Smoke Tests
Minimal tests to verify basic functionality
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
from l2_m7_application_performance_monitoring import (
    ProfiledRAGPipeline,
    MemoryProfiledComponent,
    monitor_memory_leak,
    apm_manager,
    DDTRACE_AVAILABLE
)


class TestConfiguration:
    """Test configuration loading"""

    def test_config_loads(self):
        """Test that config module loads without errors"""
        assert config.apm_config is not None
        assert config.app_config is not None
        assert config.otel_config is not None

    def test_apm_config_has_defaults(self):
        """Test APM config has expected attributes"""
        assert hasattr(config.apm_config, 'DD_SERVICE')
        assert hasattr(config.apm_config, 'DD_ENV')
        assert hasattr(config.apm_config, 'DD_TRACE_SAMPLE_RATE')
        assert hasattr(config.apm_config, 'DD_PROFILING_CAPTURE_PCT')

    def test_safe_defaults(self):
        """Test that default sampling rates are production-safe"""
        # Profiling should be ≤5% for production safety
        assert config.apm_config.DD_PROFILING_CAPTURE_PCT <= 5
        # Trace sampling should be reasonable
        assert 0 <= config.apm_config.DD_TRACE_SAMPLE_RATE <= 1.0

    def test_get_clients(self):
        """Test client initialization function"""
        clients = config.get_clients()
        assert isinstance(clients, dict)
        assert 'apm_enabled' in clients


class TestAPMManager:
    """Test APM Manager functionality"""

    def test_apm_manager_exists(self):
        """Test APM manager is initialized"""
        assert apm_manager is not None

    def test_apm_manager_initialize_without_keys(self):
        """Test APM manager handles missing keys gracefully"""
        # Should not crash even without DD_API_KEY
        # Will log warning but return False
        result = apm_manager.initialize()
        # Either succeeds (if keys present) or fails gracefully
        assert isinstance(result, bool)

    def test_apm_manager_shutdown(self):
        """Test APM manager can shutdown safely"""
        # Should not crash even if not initialized
        apm_manager.shutdown()
        assert True  # If we get here, shutdown worked


class TestProfiledRAGPipeline:
    """Test RAG Pipeline with profiling"""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return ProfiledRAGPipeline()

    def test_pipeline_creation(self, pipeline):
        """Test pipeline can be created"""
        assert pipeline is not None
        assert hasattr(pipeline, 'process_query')

    def test_process_query_basic(self, pipeline):
        """Test basic query processing"""
        result = pipeline.process_query(
            query="What is the compliance policy?",
            user_id="test_user"
        )

        # Verify response structure
        assert isinstance(result, dict)
        assert 'query' in result
        assert 'response' in result
        assert 'context_length' in result
        assert 'num_results' in result

    def test_process_query_returns_expected_shape(self, pipeline):
        """Test query response has expected fields and types"""
        result = pipeline.process_query(
            query="Test query",
            user_id="test"
        )

        assert isinstance(result['query'], str)
        assert isinstance(result['response'], str)
        assert isinstance(result['context_length'], int)
        assert isinstance(result['num_results'], int)
        assert result['num_results'] > 0

    def test_embed_query(self, pipeline):
        """Test embedding function"""
        embeddings = pipeline._embed_query("test query")
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1536  # Expected embedding dimension
        assert all(isinstance(x, float) for x in embeddings)

    def test_search_vectordb(self, pipeline):
        """Test vector search function"""
        embeddings = [0.1] * 1536
        results = pipeline._search_vectordb(embeddings, top_k=5)

        assert isinstance(results, list)
        assert len(results) == 5
        assert all('id' in r and 'score' in r for r in results)

    def test_process_context(self, pipeline):
        """Test context processing"""
        results = [
            {"id": "doc_1", "score": 0.9},
            {"id": "doc_2", "score": 0.8}
        ]
        context = pipeline._process_context(results)

        assert isinstance(context, str)
        assert len(context) > 0


class TestMemoryProfiling:
    """Test memory profiling functionality"""

    @pytest.fixture
    def profiler(self):
        """Create memory profiler instance"""
        return MemoryProfiledComponent()

    def test_profiler_creation(self, profiler):
        """Test profiler can be created"""
        assert profiler is not None
        assert hasattr(profiler, 'cache_documents')
        assert hasattr(profiler, 'get_memory_stats')

    def test_get_memory_stats(self, profiler):
        """Test memory stats collection"""
        stats = profiler.get_memory_stats()

        assert isinstance(stats, dict)
        assert 'current_mb' in stats
        assert 'peak_mb' in stats
        assert 'baseline_mb' in stats
        assert 'growth_mb' in stats

        # All values should be non-negative
        assert all(v >= 0 for v in stats.values())

    def test_cache_documents(self, profiler):
        """Test document caching"""
        docs = ["Document 1", "Document 2", "Document 3"]

        # Should not raise exception
        profiler.cache_documents(docs)

        # Memory stats should show some growth
        stats = profiler.get_memory_stats()
        assert stats['current_mb'] >= stats['baseline_mb']

    def test_cleanup(self, profiler):
        """Test cleanup doesn't crash"""
        profiler.cleanup()
        assert True

    def test_monitor_memory_leak(self):
        """Test memory leak monitoring function"""
        # Run with minimal iterations
        results = monitor_memory_leak(iterations=2)

        assert isinstance(results, dict)
        assert 'iterations' in results
        assert 'final_growth_mb' in results
        assert 'leak_detected' in results
        assert results['iterations'] == 2


class TestGracefulDegradation:
    """Test that code works without APM dependencies"""

    def test_works_without_ddtrace(self):
        """Test that pipeline works even if ddtrace not available"""
        # Pipeline should work regardless of DDTRACE_AVAILABLE
        pipeline = ProfiledRAGPipeline()
        result = pipeline.process_query("test", "user")
        assert result is not None

    def test_memory_profiling_without_apm(self):
        """Test memory profiling works without APM"""
        profiler = MemoryProfiledComponent()
        profiler.cache_documents(["test"])
        stats = profiler.get_memory_stats()
        assert stats is not None


class TestExampleData:
    """Test example data files exist and are valid"""

    def test_example_json_exists(self):
        """Test example_data.json exists"""
        path = Path(__file__).parent / "example_data.json"
        assert path.exists()

    def test_example_json_valid(self):
        """Test example_data.json is valid JSON"""
        import json
        path = Path(__file__).parent / "example_data.json"

        with open(path) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert 'sample_queries' in data
        assert 'sample_documents' in data

    def test_example_txt_exists(self):
        """Test example_data.txt exists"""
        path = Path(__file__).parent / "example_data.txt"
        assert path.exists()


class TestIntegration:
    """Integration tests for full workflow"""

    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow"""
        # Initialize pipeline
        pipeline = ProfiledRAGPipeline()

        # Process query
        result = pipeline.process_query(
            query="What are the compliance requirements?",
            user_id="integration_test"
        )

        # Verify complete response
        assert result is not None
        assert len(result['response']) > 0
        assert result['num_results'] > 0

    def test_memory_monitoring_workflow(self):
        """Test complete memory monitoring workflow"""
        # Run leak detection
        results = monitor_memory_leak(iterations=3)

        # Verify results structure
        assert results['iterations'] == 3
        assert isinstance(results['leak_detected'], bool)


# ==========================================
# Test Runner
# ==========================================

if __name__ == "__main__":
    """
    Run smoke tests

    Usage:
        python tests_smoke.py
        pytest tests_smoke.py -v
    """
    print("=" * 60)
    print("Running Module 7.2 Smoke Tests")
    print("=" * 60)
    print()

    # Run with pytest if available, otherwise basic checks
    try:
        import pytest
        exit_code = pytest.main([__file__, "-v", "--tb=short"])
        sys.exit(exit_code)
    except ImportError:
        print("pytest not available, running basic checks...")

        # Basic manual checks
        print("✓ Testing configuration...")
        assert config.apm_config is not None

        print("✓ Testing APM manager...")
        assert apm_manager is not None

        print("✓ Testing pipeline...")
        pipeline = ProfiledRAGPipeline()
        result = pipeline.process_query("test", "user")
        assert result is not None

        print("✓ Testing memory profiling...")
        profiler = MemoryProfiledComponent()
        stats = profiler.get_memory_stats()
        assert stats is not None

        print()
        print("=" * 60)
        print("All basic checks passed!")
        print("Install pytest for full test suite: pip install pytest")
        print("=" * 60)
