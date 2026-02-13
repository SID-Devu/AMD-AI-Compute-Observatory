"""
AACO Unit Tests - Core Module
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from aaco.core.session import Session, SessionManager
from aaco.core.schema import (
    SessionMetadata,
    InferenceResult,
    KernelSummary,
    KernelMetrics,
    PhaseMetrics,
    BottleneckClassification,
)
from aaco.core.utils import get_monotonic_ns, Timer, RateTracker


class TestSession:
    """Tests for Session class."""
    
    def test_session_creation(self):
        """Test session ID and path creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(base_dir=Path(tmpdir))
            
            assert session.session_id is not None
            assert len(session.session_id) > 10
            assert session.path.exists()
    
    def test_session_with_tag(self):
        """Test session with custom tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(base_dir=Path(tmpdir), tag="test_run")
            
            assert session.tag == "test_run"
    
    def test_save_artifact(self):
        """Test artifact saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(base_dir=Path(tmpdir))
            
            data = {"test": "data", "value": 123}
            session.save_artifact("test.json", data)
            
            artifact_path = session.path / "test.json"
            assert artifact_path.exists()
            
            with open(artifact_path) as f:
                loaded = json.load(f)
            
            assert loaded == data
    
    def test_session_metadata(self):
        """Test session metadata collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(base_dir=Path(tmpdir))
            
            meta = session.get_metadata()
            
            assert meta["session_id"] == session.session_id
            assert "hostname" in meta
            assert "platform" in meta
            assert "timestamp" in meta


class TestSessionManager:
    """Tests for SessionManager class."""
    
    def test_create_session(self):
        """Test session creation via manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(base_dir=Path(tmpdir))
            session = manager.create_session(tag="manager_test")
            
            assert session.tag == "manager_test"
            assert session.path.exists()
    
    def test_list_sessions(self):
        """Test listing sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(base_dir=Path(tmpdir))
            
            # Create multiple sessions
            s1 = manager.create_session(tag="first")
            s2 = manager.create_session(tag="second")
            
            sessions = manager.list_sessions()
            assert len(sessions) >= 2


class TestUtils:
    """Tests for utility functions."""
    
    def test_monotonic_timer(self):
        """Test monotonic nanosecond timer."""
        t1 = get_monotonic_ns()
        t2 = get_monotonic_ns()
        
        assert t2 >= t1
        assert isinstance(t1, int)
    
    def test_timer_context(self):
        """Test Timer context manager."""
        import time
        
        with Timer() as t:
            time.sleep(0.01)  # 10ms
        
        assert t.elapsed_ms >= 9  # Allow some tolerance
        assert t.elapsed_ms < 50  # But not too much
    
    def test_rate_tracker(self):
        """Test RateTracker."""
        tracker = RateTracker(window_size=5)
        
        for i in range(10):
            tracker.add(1.0)
        
        assert tracker.get_rate() > 0


class TestSchema:
    """Tests for schema dataclasses."""
    
    def test_inference_result(self):
        """Test InferenceResult dataclass."""
        result = InferenceResult(
            iteration=0,
            phase="measurement",
            latency_ms=10.5,
            t_start_ns=1000000,
            t_end_ns=11500000,
        )
        
        assert result.iteration == 0
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
            p90_ms=11.2,
            p99_ms=12.5,
            min_ms=8.0,
            max_ms=15.0,
            iqr_ms=2.0,
            cov_pct=10.0,
        )
        
        assert phase.name == "measurement"
        assert phase.mean_ms == 10.0
        assert phase.p99_ms == 12.5
    
    def test_bottleneck_classification(self):
        """Test BottleneckClassification dataclass."""
        classification = BottleneckClassification(
            primary="launch_overhead",
            secondary=["cpu_bound"],
            confidence=0.85,
            indicators={"microkernel_pct": 45.0},
            evidence=["High microkernel percentage"],
            recommendations=["Enable kernel fusion"],
        )
        
        assert classification.primary == "launch_overhead"
        assert classification.confidence == 0.85
        assert len(classification.evidence) > 0


class TestSchemaDict:
    """Test schema serialization."""
    
    def test_inference_result_to_dict(self):
        """Test InferenceResult __dict__ serialization."""
        result = InferenceResult(
            iteration=5,
            phase="warmup",
            latency_ms=12.3,
            t_start_ns=0,
            t_end_ns=12300000,
        )
        
        d = result.__dict__
        
        assert d["iteration"] == 5
        assert d["phase"] == "warmup"
        assert d["latency_ms"] == 12.3
        
        # Test JSON serialization
        json_str = json.dumps(d)
        assert "warmup" in json_str
