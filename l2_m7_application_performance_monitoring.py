"""
Module 7.2: Application Performance Monitoring
Implements Datadog APM integration with OpenTelemetry for RAG system profiling

This module demonstrates:
- Datadog APM initialization with OpenTelemetry compatibility
- Custom profiling decorators for RAG pipeline functions
- Memory leak detection
- Database query profiling
- Production-safe configuration with overhead limits
"""

import time
import logging
import tracemalloc
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

# APM imports (graceful degradation if not available)
try:
    from ddtrace import tracer, patch_all
    from ddtrace.profiling import Profiler
    DDTRACE_AVAILABLE = True
except ImportError:
    DDTRACE_AVAILABLE = False
    logging.warning("ddtrace not available - APM features disabled")

# OpenTelemetry imports (M7.1 prerequisite)
try:
    from opentelemetry import trace as otel_trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available")

from config import apm_config, APMConfig

logger = logging.getLogger(__name__)


# ==========================================
# APM Manager - Core Initialization
# ==========================================

class APMManager:
    """
    Manages Datadog APM lifecycle and OpenTelemetry bridge

    Handles:
    - Datadog tracer configuration
    - OpenTelemetry compatibility bridge
    - Continuous profiler startup/shutdown
    - Production safety limits
    """

    def __init__(self, config: APMConfig):
        """
        Initialize APM manager

        Args:
            config: APM configuration object
        """
        self.config = config
        self.profiler: Optional[Any] = None
        self._initialized = False
        self._ddtrace_available = DDTRACE_AVAILABLE

    def initialize(self) -> bool:
        """
        Initialize Datadog APM with OpenTelemetry compatibility

        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            logger.warning("APM already initialized")
            return True

        if not self._ddtrace_available:
            logger.warning("⚠️ Datadog ddtrace not available - skipping APM initialization")
            return False

        if not self.config.is_configured:
            logger.warning("⚠️ APM not configured (missing DD_API_KEY) - skipping initialization")
            return False

        try:
            # Configure Datadog tracer
            tracer.configure(
                hostname=self.config.DD_SITE,
                # Service tagging (unified service tags)
                service=self.config.DD_SERVICE,
                env=self.config.DD_ENV,
                version=self.config.DD_VERSION,
                # Sampling configuration
                sample_rate=self.config.DD_TRACE_SAMPLE_RATE,
                analytics_enabled=self.config.DD_TRACE_ANALYTICS_ENABLED,
                # Performance limits
                profiling=self.config.DD_PROFILING_ENABLED,
            )

            # Bridge OpenTelemetry and Datadog (CRITICAL for M7.1 compatibility)
            if OTEL_AVAILABLE:
                try:
                    from ddtrace.opentelemetry import TracerProvider as DDTracerProvider
                    dd_provider = DDTracerProvider()
                    otel_trace.set_tracer_provider(dd_provider)
                    logger.info("✅ OpenTelemetry bridge enabled")
                except Exception as e:
                    logger.warning(f"OpenTelemetry bridge failed: {e}")

            # Auto-instrument common libraries
            # This adds APM detail to existing spans WITHOUT creating duplicates
            patch_all(
                logging=True,  # Correlate logs with traces
                httpx=True,    # HTTP client profiling
                requests=True,
                asyncio=True,  # Async profiling
            )

            # Start continuous profiler
            if self.config.DD_PROFILING_ENABLED:
                self._start_profiler()

            self._initialized = True
            logger.info(f"✅ APM initialized: {self.config.DD_SERVICE} ({self.config.DD_ENV})")
            logger.info(f"   Profiling: {self.config.DD_PROFILING_ENABLED}")
            logger.info(f"   Sample rate: {self.config.DD_TRACE_SAMPLE_RATE * 100}%")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize APM: {e}")
            return False

    def _start_profiler(self) -> None:
        """Start continuous profiler with safety limits"""
        try:
            self.profiler = Profiler(
                service=self.config.DD_SERVICE,
                env=self.config.DD_ENV,
                version=self.config.DD_VERSION,
                # CPU profiling
                cpu_time_enabled=self.config.DD_PROFILING_CPU_ENABLED,
                # Memory profiling
                memory_enabled=self.config.DD_PROFILING_MEMORY_ENABLED,
                # Capture settings
                capture_pct=self.config.DD_PROFILING_CAPTURE_PCT,
                max_time_usage_pct=self.config.DD_PROFILING_MAX_TIME_USAGE_PCT,
            )
            self.profiler.start()
            logger.info("✅ Continuous profiler started")
        except Exception as e:
            logger.error(f"❌ Failed to start profiler: {e}")
            # Non-fatal - APM still works without profiler

    def shutdown(self) -> None:
        """Graceful shutdown"""
        if self.profiler:
            try:
                self.profiler.stop()
                logger.info("Profiler stopped")
            except Exception as e:
                logger.error(f"Error stopping profiler: {e}")

        if self._ddtrace_available and self._initialized:
            tracer.shutdown()

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if APM is initialized"""
        return self._initialized


# Global APM manager instance
apm_manager = APMManager(apm_config)


# ==========================================
# Profiled RAG Pipeline
# ==========================================

def wrap_with_apm(name: str, service: Optional[str] = None):
    """
    Decorator wrapper that works with or without ddtrace

    Args:
        name: Span name for profiling
        service: Service name (optional)

    Returns:
        Decorator function
    """
    def decorator(func):
        if DDTRACE_AVAILABLE and apm_manager.is_initialized:
            return tracer.wrap(name=name, service=service or apm_config.DD_SERVICE)(func)
        else:
            # Passthrough if ddtrace not available
            return func
    return decorator


class ProfiledRAGPipeline:
    """
    RAG Pipeline with APM instrumentation

    Demonstrates custom profiling annotations for:
    - Query embedding
    - Vector search
    - Context processing (with O(n²) bottleneck simulation)
    - Response generation
    """

    def __init__(self):
        """Initialize profiled RAG pipeline"""
        self.cache = {}
        logger.info("ProfiledRAGPipeline initialized")

    @wrap_with_apm("rag.query")
    def process_query(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Main query processing with APM profiling

        The @wrap_with_apm decorator:
        1. Creates a custom span in Datadog
        2. Automatically captures exceptions
        3. Tags span with service name
        4. Profiles everything inside this function

        Args:
            query: User query string
            user_id: User identifier for tagging

        Returns:
            Dict containing response and metadata
        """
        if DDTRACE_AVAILABLE and apm_manager.is_initialized:
            span = tracer.current_span()
            if span:
                span.set_tag("user.id", user_id)
                span.set_tag("query.length", len(query))

        try:
            # Step 1: Embed query (usually fast)
            embeddings = self._embed_query(query)

            # Step 2: Search vector database (network call)
            results = self._search_vectordb(embeddings)

            # Step 3: Process context (THIS is where bottlenecks often hide)
            context = self._process_context(results)

            # Step 4: Generate response
            response = self._generate_response(query, context)

            # Tag success
            if DDTRACE_AVAILABLE and apm_manager.is_initialized:
                span = tracer.current_span()
                if span:
                    span.set_tag("response.success", True)
                    span.set_tag("response.length", len(response))

            return {
                "query": query,
                "response": response,
                "context_length": len(context),
                "num_results": len(results)
            }

        except Exception as e:
            # APM automatically captures exception details
            if DDTRACE_AVAILABLE and apm_manager.is_initialized:
                span = tracer.current_span()
                if span:
                    span.set_tag("error", True)
                    span.set_tag("error.type", type(e).__name__)

            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise

    @wrap_with_apm("rag.embed_query")
    def _embed_query(self, query: str) -> List[float]:
        """
        Embedding with profiling

        Args:
            query: Query string to embed

        Returns:
            Embedding vector (simulated)
        """
        # Simulate embedding time (replace with actual embedding call)
        time.sleep(0.05)
        return [0.1] * 1536  # Simulated 1536-dimensional embedding

    @wrap_with_apm("rag.search_vectordb")
    def _search_vectordb(self, embeddings: List[float], top_k: int = 10) -> List[Dict]:
        """
        Vector database search with profiling

        APM will show:
        - Network latency to vector DB
        - Serialization overhead
        - Response parsing time

        Args:
            embeddings: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results
        """
        # Simulate search time
        time.sleep(0.1)
        return [{"id": f"doc_{i}", "score": 0.9 - (i * 0.05)} for i in range(top_k)]

    @wrap_with_apm("rag.process_context")
    def _process_context(self, results: List[Dict]) -> str:
        """
        Context processing with detailed profiling

        This is typically where bottlenecks hide:
        - Chunk filtering algorithms
        - Overlap calculations
        - Ranking logic

        Args:
            results: Search results from vector DB

        Returns:
            Formatted context string
        """
        if DDTRACE_AVAILABLE and apm_manager.is_initialized:
            span = tracer.current_span()
            if span:
                span.set_tag("results.count", len(results))

        # BOTTLENECK SIMULATION: O(n²) overlap check
        # APM will show this consuming most of the time
        filtered_results = self._remove_overlapping_chunks(results)

        # Format context
        context = "\n\n".join([f"Document {r['id']}: Score {r['score']}" for r in filtered_results])
        return context

    def _remove_overlapping_chunks(self, results: List[Dict]) -> List[Dict]:
        """
        Simulated O(n²) bottleneck - APM will catch this

        In real code, this might be:
        - Complex regex operations
        - Multiple nested loops
        - Inefficient data structure operations

        Args:
            results: List of search results

        Returns:
            Filtered results without overlaps
        """
        filtered = []
        for i, r1 in enumerate(results):
            has_overlap = False
            for j, r2 in enumerate(results):
                if i != j:
                    # Simulated expensive comparison (10ms each)
                    time.sleep(0.01)
                    # In reality: calculate chunk overlap, check duplicates, etc
            if not has_overlap:
                filtered.append(r1)
        return filtered

    @wrap_with_apm("rag.generate_response")
    def _generate_response(self, query: str, context: str) -> str:
        """
        LLM generation with profiling

        APM will show:
        - Time waiting for LLM API
        - Token encoding overhead
        - Response parsing time

        Args:
            query: User query
            context: Formatted context from search

        Returns:
            Generated response string
        """
        # Simulate LLM time
        time.sleep(0.2)
        return f"Response to '{query[:50]}...' based on {len(context)} chars of context"


# ==========================================
# Memory Profiling
# ==========================================

class MemoryProfiledComponent:
    """
    Component with memory profiling for leak detection

    Demonstrates:
    - Memory tracking with tracemalloc
    - Memory leak simulation
    - APM metric reporting
    """

    def __init__(self):
        """Initialize memory profiling"""
        # Start memory tracking
        tracemalloc.start()
        self._baseline_memory = tracemalloc.get_traced_memory()[0]
        self.cache = {}
        logger.info("Memory profiling enabled")

    @wrap_with_apm("memory.cache_documents")
    def cache_documents(self, documents: List[str]) -> None:
        """
        Cache documents with memory profiling

        Memory profiler will show:
        - Peak memory usage
        - Memory allocated per operation
        - Memory not freed (potential leak)

        Args:
            documents: List of documents to cache
        """
        if DDTRACE_AVAILABLE and apm_manager.is_initialized:
            span = tracer.current_span()
        else:
            span = None

        # Track memory before
        mem_before = tracemalloc.get_traced_memory()[0]

        # Simulate caching (in reality: Redis, in-memory dict, etc)
        for i, doc in enumerate(documents):
            # POTENTIAL LEAK: Not clearing old entries
            self.cache[f"doc_{i}"] = doc * 10  # Multiply to simulate leak

        # Track memory after
        mem_after = tracemalloc.get_traced_memory()[0]
        mem_delta = mem_after - mem_before

        # Report to APM
        if span:
            span.set_metric("memory.allocated_mb", mem_delta / (1024 * 1024))
            span.set_metric("memory.total_mb", mem_after / (1024 * 1024))

        # Alert if memory grew significantly
        if mem_delta > 100 * 1024 * 1024:  # >100MB
            logger.warning(f"⚠️  Large memory allocation: {mem_delta / (1024*1024):.1f} MB")
            if span:
                span.set_tag("memory.large_allocation", True)

        logger.info(f"Cached {len(documents)} documents, allocated {mem_delta / (1024*1024):.2f} MB")

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory usage statistics

        Returns:
            Dict with memory metrics in MB
        """
        current, peak = tracemalloc.get_traced_memory()

        return {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "baseline_mb": self._baseline_memory / (1024 * 1024),
            "growth_mb": (current - self._baseline_memory) / (1024 * 1024)
        }

    def cleanup(self) -> None:
        """Cleanup memory tracking"""
        tracemalloc.stop()


def monitor_memory_leak(iterations: int = 10) -> Dict[str, Any]:
    """
    Monitor memory growth over multiple requests to detect leaks

    Args:
        iterations: Number of iterations to monitor

    Returns:
        Dict with leak detection results
    """
    component = MemoryProfiledComponent()
    growth_history = []

    logger.info(f"Starting memory leak monitoring ({iterations} iterations)...")

    for i in range(iterations):
        # Simulate document caching
        docs = [f"Document {j} content " * 100 for j in range(100)]
        component.cache_documents(docs)

        # Check memory
        stats = component.get_memory_stats()
        growth_history.append(stats['growth_mb'])

        if i % 5 == 0:
            logger.info(f"Iteration {i}: {stats['growth_mb']:.1f} MB growth")

        # Alert if memory grew >500MB (potential leak)
        if stats['growth_mb'] > 500:
            logger.error(f"🚨 MEMORY LEAK DETECTED: {stats['growth_mb']:.1f} MB growth")
            break

    component.cleanup()

    return {
        "iterations": len(growth_history),
        "final_growth_mb": growth_history[-1] if growth_history else 0,
        "average_growth_per_iteration_mb": sum(growth_history) / len(growth_history) if growth_history else 0,
        "leak_detected": growth_history[-1] > 500 if growth_history else False
    }


# ==========================================
# CLI Usage Examples
# ==========================================

if __name__ == "__main__":
    """
    CLI usage examples

    Run with: python l2_m7_application_performance_monitoring.py
    """
    print("=" * 60)
    print("Module 7.2: Application Performance Monitoring")
    print("=" * 60)
    print()

    # Initialize APM
    print("1. Initializing APM...")
    success = apm_manager.initialize()
    if success:
        print("   ✅ APM initialized successfully")
    else:
        print("   ⚠️  APM initialization skipped (no keys or ddtrace not installed)")
    print()

    # Test RAG pipeline
    print("2. Testing Profiled RAG Pipeline...")
    pipeline = ProfiledRAGPipeline()
    result = pipeline.process_query("What are the compliance requirements?", user_id="test_user")
    print(f"   Query: {result['query'][:50]}...")
    print(f"   Response: {result['response'][:80]}...")
    print(f"   Context length: {result['context_length']} chars")
    print(f"   Results: {result['num_results']}")
    print()

    # Test memory profiling
    print("3. Testing Memory Profiling...")
    leak_results = monitor_memory_leak(iterations=5)
    print(f"   Iterations: {leak_results['iterations']}")
    print(f"   Final growth: {leak_results['final_growth_mb']:.1f} MB")
    print(f"   Average growth: {leak_results['average_growth_per_iteration_mb']:.1f} MB/iter")
    print(f"   Leak detected: {leak_results['leak_detected']}")
    print()

    # Shutdown APM
    print("4. Shutting down APM...")
    apm_manager.shutdown()
    print("   ✅ APM shutdown complete")
    print()

    print("=" * 60)
    print("Demo complete! Check Datadog APM UI if configured.")
    print("=" * 60)
