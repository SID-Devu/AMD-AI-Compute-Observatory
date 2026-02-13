"""
AACO Unit Tests - Analytics Module
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from aaco.analytics.metrics import DerivedMetricsEngine
from aaco.analytics.classify import BottleneckClassifier, BottleneckCategory
from aaco.analytics.diff import RegressionDetector, RegressionThresholds
from aaco.core.schema import (
    InferenceResult,
    KernelMetrics,
    KernelSummary,
    PhaseMetrics,
    DerivedMetrics,
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
            results.append(InferenceResult(
                iteration=i,
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


class TestBottleneckClassifier:
    """Tests for BottleneckClassifier."""
    
    def create_mock_metrics(self, **kwargs):
        """Create mock DerivedMetrics with custom values."""
        defaults = {
            "gpu_active_ratio": 0.5,
            "microkernel_pct": 10.0,
            "kar": 2.0,
            "gpu_util_pct": 60.0,
            "mem_util_pct": 40.0,
            "cpu_pct": 30.0,
            "temp_c": 70.0,
        }
        defaults.update(kwargs)
        
        metrics = MagicMock()
        metrics.efficiency = {
            "gpu_active_ratio": defaults["gpu_active_ratio"],
            "microkernel_pct": defaults["microkernel_pct"],
            "kernel_amplification_ratio": defaults["kar"],
        }
        metrics.gpu = {
            "gpu_util_mean_pct": defaults["gpu_util_pct"],
            "mem_util_mean_pct": defaults["mem_util_pct"],
            "temp_max_c": defaults["temp_c"],
            "power_max_w": 100,
            "sclk_range_mhz": 50,
        }
        metrics.system = {
            "cpu_max_pct": defaults["cpu_pct"],
            "rss_delta_mb": 10,
        }
        metrics.measurement_phase = MagicMock()
        metrics.measurement_phase.cov_pct = 5.0
        metrics.latency = {"warmup_effect_pct": 10.0}
        
        return metrics
    
    def test_classify_launch_overhead(self):
        """Test launch overhead detection."""
        classifier = BottleneckClassifier()
        
        metrics = self.create_mock_metrics(
            microkernel_pct=50.0,
            kar=15.0,
            gpu_active_ratio=0.2,
        )
        
        result = classifier.classify(metrics=metrics)
        
        assert result.primary == "launch_overhead"
        assert result.confidence > 0.5
        assert len(result.evidence) > 0
    
    def test_classify_compute_bound(self):
        """Test compute-bound detection."""
        classifier = BottleneckClassifier()
        
        metrics = self.create_mock_metrics(
            gpu_util_pct=92.0,
            gpu_active_ratio=0.85,
            microkernel_pct=5.0,
        )
        
        result = classifier.classify(metrics=metrics)
        
        assert result.primary == "compute_bound"
    
    def test_classify_memory_bound(self):
        """Test memory-bound detection."""
        classifier = BottleneckClassifier()
        
        metrics = self.create_mock_metrics(
            mem_util_pct=90.0,
            gpu_util_pct=40.0,
        )
        
        result = classifier.classify(metrics=metrics)
        
        assert result.primary == "memory_bound"
    
    def test_classify_thermal(self):
        """Test thermal throttle detection."""
        classifier = BottleneckClassifier()
        
        metrics = self.create_mock_metrics(temp_c=92.0)
        
        result = classifier.classify(metrics=metrics)
        
        assert "thermal" in result.primary or result.primary == "thermal_throttle"
    
    def test_recommendations_provided(self):
        """Test that recommendations are provided."""
        classifier = BottleneckClassifier()
        
        metrics = self.create_mock_metrics(microkernel_pct=50.0)
        
        result = classifier.classify(metrics=metrics)
        
        assert len(result.recommendations) > 0


class TestRegressionDetector:
    """Tests for RegressionDetector."""
    
    def test_detect_regression(self):
        """Test regression detection."""
        detector = RegressionDetector()
        
        # Create baseline (fast) and current (slow) metrics
        baseline = MagicMock()
        baseline.measurement_phase = PhaseMetrics(
            name="measurement",
            iterations=100,
            total_time_ms=1000,
            mean_ms=10.0,
            std_ms=1.0,
            p50_ms=9.8,
            p90_ms=11.0,
            p99_ms=12.0,
            min_ms=8.0,
            max_ms=14.0,
            iqr_ms=2.0,
            cov_pct=10.0,
        )
        baseline.throughput = {"inferences_per_sec": 100}
        baseline.efficiency = {"gpu_active_ratio": 0.7}
        baseline.gpu = {"gpu_util_mean_pct": 75}
        baseline.system = {}
        baseline.latency = {}
        
        current = MagicMock()
        current.measurement_phase = PhaseMetrics(
            name="measurement",
            iterations=100,
            total_time_ms=1200,
            mean_ms=12.0,  # 20% regression
            std_ms=1.5,
            p50_ms=11.8,
            p90_ms=13.5,
            p99_ms=15.0,
            min_ms=9.0,
            max_ms=18.0,
            iqr_ms=3.0,
            cov_pct=12.5,
        )
        current.throughput = {"inferences_per_sec": 83}
        current.efficiency = {"gpu_active_ratio": 0.65}
        current.gpu = {"gpu_util_mean_pct": 70}
        current.system = {}
        current.latency = {}
        
        verdict = detector.compare_metrics(baseline, current)
        
        assert verdict.verdict == "REGRESSION"
        assert len(verdict.regressions) > 0
    
    def test_detect_improvement(self):
        """Test improvement detection."""
        detector = RegressionDetector()
        
        baseline = MagicMock()
        baseline.measurement_phase = PhaseMetrics(
            name="measurement", iterations=100, total_time_ms=1000,
            mean_ms=10.0, std_ms=1.0, p50_ms=9.8, p90_ms=11.0,
            p99_ms=12.0, min_ms=8.0, max_ms=14.0, iqr_ms=2.0, cov_pct=10.0,
        )
        baseline.throughput = {"inferences_per_sec": 100}
        baseline.efficiency = {}
        baseline.gpu = {}
        baseline.system = {}
        baseline.latency = {}
        
        current = MagicMock()
        current.measurement_phase = PhaseMetrics(
            name="measurement", iterations=100, total_time_ms=800,
            mean_ms=8.0,  # 20% faster
            std_ms=0.8, p50_ms=7.9, p90_ms=8.8,
            p99_ms=9.5, min_ms=6.5, max_ms=10.0, iqr_ms=1.5, cov_pct=10.0,
        )
        current.throughput = {"inferences_per_sec": 125}
        current.efficiency = {}
        current.gpu = {}
        current.system = {}
        current.latency = {}
        
        verdict = detector.compare_metrics(baseline, current)
        
        assert verdict.verdict == "IMPROVEMENT"
        assert len(verdict.improvements) > 0
    
    def test_statistical_significance(self):
        """Test statistical significance calculation."""
        detector = RegressionDetector()
        
        # Same distribution - should not be significant
        baseline_raw = list(np.random.normal(10, 1, 100))
        current_raw = list(np.random.normal(10, 1, 100))
        
        p_value = detector._statistical_test(baseline_raw, current_raw)
        
        # P-value should be high (not significant) for same distribution
        assert p_value > 0.05
        
        # Different distributions - should be significant
        current_raw_different = list(np.random.normal(12, 1, 100))
        p_value_diff = detector._statistical_test(baseline_raw, current_raw_different)
        
        # P-value should be low (significant) for different distributions
        assert p_value_diff < 0.05
