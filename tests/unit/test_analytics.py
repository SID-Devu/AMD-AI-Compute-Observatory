"""
AACO Unit Tests - Analytics Module
Working tests for DerivedMetricsEngine.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
import numpy as np

from aaco.analytics.metrics import DerivedMetricsEngine
from aaco.core.schema import (
    InferenceIteration,
    PhaseMetrics,
)


class TestDerivedMetricsEngine:
    """Tests for DerivedMetricsEngine."""
    
    def create_inference_results(self, n=100, mean=10.0, std=1.0, phase="measurement"):
        """Helper to create synthetic inference results."""
        latencies = np.random.normal(mean, std, n)
        latencies = np.maximum(latencies, 1.0)  # Ensure positive
        
        results = []
        t = 0
        for i, lat in enumerate(latencies):
            dur_ns = int(lat * 1_000_000)
            results.append(InferenceIteration(
                iter_idx=i,
                phase=phase,
                latency_ms=lat,
                t_start_ns=t,
                t_end_ns=t + dur_ns,
            ))
            t += dur_ns
        
        return results
    
    def test_compute_basic(self):
        """Test basic metrics computation."""
        engine = DerivedMetricsEngine()
        
        # Add warmup and measurement results
        warmup = self.create_inference_results(10, mean=15.0, phase="warmup")
        measurement = self.create_inference_results(100, mean=10.0, phase="measurement")
        
        engine.add_inference_results(warmup + measurement)
        
        metrics = engine.compute()
        
        assert metrics.warmup_phase.iterations == 10
        assert metrics.measurement_phase.iterations == 100
        assert metrics.measurement_phase.mean_ms < 15.0  # Should be around 10
    
    def test_throughput_computation(self):
        """Test throughput metrics."""
        engine = DerivedMetricsEngine()
        
        # 100 iterations at 10ms each = ~100 infer/sec
        results = self.create_inference_results(100, mean=10.0, std=0.1, phase="measurement")
        engine.add_inference_results(results)
        
        metrics = engine.compute()
        
        # Throughput should be around 100 infer/sec
        ips = metrics.throughput.get("inferences_per_sec", 0)
        assert 80 < ips < 120
    
    def test_latency_statistics(self):
        """Test latency statistical calculations."""
        engine = DerivedMetricsEngine()
        
        results = self.create_inference_results(1000, mean=10.0, std=2.0, phase="measurement")
        engine.add_inference_results(results)
        
        metrics = engine.compute()
        phase = metrics.measurement_phase
        
        # Statistical properties
        assert 9 < phase.mean_ms < 11
        assert phase.std_ms > 0
        assert phase.p50_ms < phase.p99_ms  # P99 should be higher
        assert phase.min_ms < phase.max_ms
