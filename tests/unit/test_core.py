"""
AACO Unit Tests - Core Module
Working tests for core functionality.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import json
import pytest
from pathlib import Path

from aaco.core.schema import (
    InferenceIteration,
    KernelSummary,
    KernelMetrics,
    PhaseMetrics,
)
from aaco.core.utils import get_monotonic_ns


class TestUtils:
    """Tests for utility functions."""
    
    def test_monotonic_timer(self):
        """Test monotonic nanosecond timer."""
        t1 = get_monotonic_ns()
        t2 = get_monotonic_ns()
        
        assert t2 >= t1
        assert isinstance(t1, int)


class TestSchema:
    """Tests for schema dataclasses."""
    
    def test_inference_iteration(self):
        """Test InferenceIteration dataclass."""
        result = InferenceIteration(
            iter_idx=0,
            phase="measurement",
            latency_ms=10.5,
            t_start_ns=1000000,
            t_end_ns=11500000,
        )
        
        assert result.iter_idx == 0
        assert result.phase == "measurement"
        assert result.latency_ms == 10.5
    
    def test_kernel_summary(self):
        """Test KernelSummary dataclass."""
        summary = KernelSummary(
            kernel_name="gemm_kernel",
            calls=100,
            total_time_ms=50.0,
            avg_time_us=500.0,
            min_time_us=450.0,
            max_time_us=600.0,
            std_time_us=25.0,
            pct_total=45.0,
        )
        
        assert summary.kernel_name == "gemm_kernel"
        assert summary.calls == 100
        assert summary.pct_total == 45.0
    
    def test_phase_metrics(self):
        """Test PhaseMetrics dataclass."""
        phase = PhaseMetrics(
            name="measurement",
            iterations=100,
            total_time_ms=1000.0,
            mean_ms=10.0,
            std_ms=1.0,
            p50_ms=9.8,
            p90_ms=11.5,
            p99_ms=12.5,
            min_ms=8.0,
            max_ms=15.0,
            iqr_ms=2.0,
            cov_pct=10.0,
        )
        
        assert phase.name == "measurement"
        assert phase.iterations == 100
        assert phase.mean_ms == 10.0


class TestSchemaDict:
    """Test schema serialization."""
    
    def test_inference_iteration_to_dict(self):
        """Test InferenceIteration __dict__ serialization."""
        result = InferenceIteration(
            iter_idx=5,
            phase="warmup",
            latency_ms=12.3,
            t_start_ns=0,
            t_end_ns=12300000,
        )
        
        d = result.__dict__
        
        assert d["iter_idx"] == 5
        assert d["phase"] == "warmup"
        assert d["latency_ms"] == 12.3
        
        # Test JSON serialization
        json_str = json.dumps(d)
        assert "warmup" in json_str
