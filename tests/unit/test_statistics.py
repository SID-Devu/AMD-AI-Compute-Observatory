"""
AACO Test Suite - Statistical Analysis Tests
============================================
Tests for statistical analysis functions.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

# Check if aaco.analytics module is available
try:
    from aaco import analytics
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ANALYTICS_AVAILABLE, reason="aaco.analytics module not installed")


class TestStatisticalSummary:
    """Tests for StatisticalSummary class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for tests."""
        np.random.seed(42)
        return np.random.normal(100.0, 5.0, 1000)
    
    def test_mean_calculation(self, sample_data):
        """Test mean calculation."""
        from aaco.analytics import StatisticalSummary
        
        summary = StatisticalSummary(sample_data)
        expected_mean = np.mean(sample_data)
        
        assert abs(summary.mean - expected_mean) < 1e-10
    
    def test_median_calculation(self, sample_data):
        """Test median calculation."""
        from aaco.analytics import StatisticalSummary
        
        summary = StatisticalSummary(sample_data)
        expected_median = np.median(sample_data)
        
        assert abs(summary.median - expected_median) < 1e-10
    
    def test_std_calculation(self, sample_data):
        """Test standard deviation calculation."""
        from aaco.analytics import StatisticalSummary
        
        summary = StatisticalSummary(sample_data)
        expected_std = np.std(sample_data, ddof=1)
        
        assert abs(summary.std - expected_std) < 1e-10
    
    def test_percentiles(self, sample_data):
        """Test percentile calculations."""
        from aaco.analytics import StatisticalSummary
        
        summary = StatisticalSummary(sample_data)
        
        p50 = summary.percentile(50)
        p95 = summary.percentile(95)
        p99 = summary.percentile(99)
        
        assert p50 < p95 < p99
        assert abs(p50 - summary.median) < 1e-10
    
    def test_coefficient_of_variation(self, sample_data):
        """Test CV calculation."""
        from aaco.analytics import StatisticalSummary
        
        summary = StatisticalSummary(sample_data)
        expected_cv = np.std(sample_data, ddof=1) / np.mean(sample_data)
        
        assert abs(summary.cv - expected_cv) < 1e-10
    
    def test_empty_data(self):
        """Test handling of empty data."""
        from aaco.analytics import StatisticalSummary
        
        with pytest.raises(ValueError):
            StatisticalSummary([])
    
    def test_single_value(self):
        """Test handling of single value."""
        from aaco.analytics import StatisticalSummary
        
        summary = StatisticalSummary([100.0])
        assert summary.mean == 100.0
        assert summary.median == 100.0


class TestOutlierDetection:
    """Tests for outlier detection."""
    
    @pytest.fixture
    def data_with_outliers(self):
        """Generate data with known outliers."""
        np.random.seed(42)
        data = np.random.normal(100.0, 5.0, 100).tolist()
        # Add outliers
        data.extend([150.0, 200.0, 50.0, 10.0])
        return np.array(data)
    
    def test_iqr_detection(self, data_with_outliers):
        """Test IQR-based outlier detection."""
        from aaco.analytics import OutlierDetector
        
        detector = OutlierDetector(method="iqr", threshold=1.5)
        outliers = detector.detect(data_with_outliers)
        
        # Should detect the extreme values
        assert len(outliers) > 0
        assert 200.0 in data_with_outliers[outliers]
        assert 10.0 in data_with_outliers[outliers]
    
    def test_zscore_detection(self, data_with_outliers):
        """Test Z-score based outlier detection."""
        from aaco.analytics import OutlierDetector
        
        detector = OutlierDetector(method="zscore", threshold=3.0)
        outliers = detector.detect(data_with_outliers)
        
        assert len(outliers) > 0
    
    def test_mad_detection(self, data_with_outliers):
        """Test MAD-based outlier detection."""
        from aaco.analytics import OutlierDetector
        
        detector = OutlierDetector(method="mad", threshold=3.5)
        outliers = detector.detect(data_with_outliers)
        
        assert len(outliers) > 0
    
    def test_filter_outliers(self, data_with_outliers):
        """Test outlier filtering."""
        from aaco.analytics import OutlierDetector
        
        detector = OutlierDetector(method="iqr")
        clean_data = detector.filter(data_with_outliers)
        
        assert len(clean_data) < len(data_with_outliers)
        assert 200.0 not in clean_data


class TestDriftDetection:
    """Tests for drift detection."""
    
    @pytest.fixture
    def baseline_data(self):
        """Generate baseline data."""
        np.random.seed(42)
        return np.random.normal(100.0, 5.0, 50)
    
    @pytest.fixture
    def drifted_data(self):
        """Generate data with drift."""
        np.random.seed(43)
        return np.random.normal(110.0, 5.0, 50)  # 10% increase
    
    @pytest.fixture
    def stable_data(self):
        """Generate stable data (no drift)."""
        np.random.seed(44)
        return np.random.normal(100.0, 5.0, 50)
    
    def test_detect_drift(self, baseline_data, drifted_data):
        """Test drift detection with drifted data."""
        from aaco.analytics import DriftDetector
        
        detector = DriftDetector(method="ewma_cusum")
        result = detector.detect(baseline_data, drifted_data)
        
        assert result.has_drift is True
        assert result.magnitude > 0.05  # More than 5% change
    
    def test_no_drift(self, baseline_data, stable_data):
        """Test drift detection with stable data."""
        from aaco.analytics import DriftDetector
        
        detector = DriftDetector(method="ewma_cusum", threshold=5.0)
        result = detector.detect(baseline_data, stable_data)
        
        # Should not detect significant drift
        assert result.has_drift is False or result.magnitude < 0.05
    
    def test_ewma_method(self, baseline_data, drifted_data):
        """Test EWMA method specifically."""
        from aaco.analytics import DriftDetector
        
        detector = DriftDetector(method="ewma", alpha=0.3)
        result = detector.detect(baseline_data, drifted_data)
        
        assert result is not None


class TestBottleneckClassification:
    """Tests for bottleneck classification."""
    
    @pytest.fixture
    def compute_bound_metrics(self):
        """Metrics indicating compute-bound workload."""
        return {
            "gpu_utilization": 95.0,
            "sq_busy": 0.92,
            "memory_bandwidth_utilization": 0.30,
            "l2_hit_rate": 0.95,
        }
    
    @pytest.fixture
    def memory_bound_metrics(self):
        """Metrics indicating memory-bound workload."""
        return {
            "gpu_utilization": 70.0,
            "sq_busy": 0.40,
            "memory_bandwidth_utilization": 0.95,
            "l2_hit_rate": 0.60,
        }
    
    def test_classify_compute_bound(self, compute_bound_metrics):
        """Test compute-bound classification."""
        from aaco.analytics import BottleneckClassifier
        
        classifier = BottleneckClassifier()
        result = classifier.classify(compute_bound_metrics)
        
        assert result.category == "compute_bound"
        assert result.confidence > 0.7
    
    def test_classify_memory_bound(self, memory_bound_metrics):
        """Test memory-bound classification."""
        from aaco.analytics import BottleneckClassifier
        
        classifier = BottleneckClassifier()
        result = classifier.classify(memory_bound_metrics)
        
        assert result.category == "memory_bound"
        assert result.confidence > 0.7
    
    def test_classification_evidence(self, compute_bound_metrics):
        """Test that classification provides evidence."""
        from aaco.analytics import BottleneckClassifier
        
        classifier = BottleneckClassifier()
        result = classifier.classify(compute_bound_metrics)
        
        assert result.evidence is not None
        assert len(result.evidence) > 0


class TestRootCauseAnalysis:
    """Tests for Bayesian root cause analysis."""
    
    @pytest.fixture
    def session_metrics(self):
        """Session metrics for root cause analysis."""
        return {
            "latency_ms": [10.0] * 100,
            "kernel_metrics": {
                "occupancy": 0.5,
                "register_usage": 128,
                "shared_memory": 32768,
            },
            "memory_metrics": {
                "l2_hit_rate": 0.6,
                "hbm_bandwidth": 0.8,
            }
        }
    
    def test_root_cause_ranking(self, session_metrics):
        """Test root cause ranking."""
        from aaco.analytics import BayesianRootCause
        
        analyzer = BayesianRootCause()
        result = analyzer.analyze(session_metrics)
        
        assert len(result.ranked_causes) > 0
        
        # Causes should be sorted by posterior probability
        posteriors = [c.posterior for c in result.ranked_causes]
        assert posteriors == sorted(posteriors, reverse=True)
    
    def test_posterior_probabilities_sum_to_one(self, session_metrics):
        """Test that posteriors sum to approximately 1."""
        from aaco.analytics import BayesianRootCause
        
        analyzer = BayesianRootCause()
        result = analyzer.analyze(session_metrics)
        
        total = sum(c.posterior for c in result.ranked_causes)
        assert abs(total - 1.0) < 0.01
    
    def test_min_confidence_filter(self, session_metrics):
        """Test minimum confidence filtering."""
        from aaco.analytics import BayesianRootCause
        
        analyzer = BayesianRootCause(min_confidence=0.9)
        result = analyzer.analyze(session_metrics)
        
        # All returned causes should meet minimum confidence
        for cause in result.ranked_causes:
            assert cause.posterior >= 0.9 or len(result.ranked_causes) == 0
