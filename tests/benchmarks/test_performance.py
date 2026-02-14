"""
AACO Test Suite - Benchmark Tests
=================================
Performance benchmarks for critical paths.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
import numpy as np


@pytest.fixture
def large_dataset():
    """Generate large dataset for benchmarks."""
    np.random.seed(42)
    return np.random.normal(100.0, 10.0, 100000)


class TestStatisticsBenchmarks:
    """Benchmarks for statistical functions."""
    
    @pytest.mark.slow
    def test_summary_performance(self, benchmark, large_dataset):
        """Benchmark StatisticalSummary computation."""
        from aaco.analytics import StatisticalSummary
        
        result = benchmark(StatisticalSummary, large_dataset)
        assert result.mean is not None
    
    @pytest.mark.slow
    def test_outlier_detection_performance(self, benchmark, large_dataset):
        """Benchmark outlier detection."""
        from aaco.analytics import OutlierDetector
        
        detector = OutlierDetector(method="iqr")
        result = benchmark(detector.detect, large_dataset)
        assert result is not None
    
    @pytest.mark.slow
    def test_percentile_performance(self, benchmark, large_dataset):
        """Benchmark percentile calculation."""
        result = benchmark(np.percentile, large_dataset, [50, 90, 95, 99])
        assert len(result) == 4


class TestAnalysisBenchmarks:
    """Benchmarks for analysis functions."""
    
    @pytest.mark.slow
    def test_bottleneck_classification_performance(self, benchmark):
        """Benchmark bottleneck classification."""
        from aaco.analytics import BottleneckClassifier
        
        metrics = {
            "gpu_utilization": 85.0,
            "sq_busy": 0.8,
            "memory_bandwidth_utilization": 0.5,
            "l2_hit_rate": 0.9,
        }
        
        classifier = BottleneckClassifier()
        result = benchmark(classifier.classify, metrics)
        assert result.category is not None


class TestCollectorBenchmarks:
    """Benchmarks for collectors."""
    
    @pytest.mark.slow
    def test_timing_collector_overhead(self, benchmark):
        """Benchmark timing collector overhead."""
        from aaco.collectors import TimingCollector
        
        def timing_cycle():
            collector = TimingCollector()
            collector.start()
            # Minimal work
            x = 1 + 1
            collector.stop()
            return collector.get_results()
        
        result = benchmark(timing_cycle)
        # Overhead should be minimal
        assert result["elapsed_ns"] < 1_000_000  # < 1ms overhead
