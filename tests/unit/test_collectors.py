"""
AACO Test Suite - Collector Tests
=================================
Tests for data collectors.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestTimingCollector:
    """Tests for TimingCollector."""
    
    def test_basic_timing(self):
        """Test basic timing collection."""
        from aaco.collectors import TimingCollector
        
        collector = TimingCollector()
        
        collector.start()
        time.sleep(0.01)  # 10ms
        collector.stop()
        
        results = collector.get_results()
        
        # Should be approximately 10ms
        assert results["elapsed_ns"] > 9_000_000  # > 9ms
        assert results["elapsed_ns"] < 50_000_000  # < 50ms
    
    def test_multiple_samples(self):
        """Test collecting multiple timing samples."""
        from aaco.collectors import TimingCollector
        
        collector = TimingCollector()
        samples = []
        
        for _ in range(5):
            collector.start()
            time.sleep(0.001)  # 1ms
            collector.stop()
            samples.append(collector.get_results()["elapsed_ns"])
        
        assert len(samples) == 5
        assert all(s > 0 for s in samples)
    
    def test_reset(self):
        """Test collector reset."""
        from aaco.collectors import TimingCollector
        
        collector = TimingCollector()
        
        collector.start()
        time.sleep(0.001)
        collector.stop()
        
        collector.reset()
        
        # After reset, should have no results
        results = collector.get_results()
        assert results is None or results.get("elapsed_ns", 0) == 0


class TestMemoryCollector:
    """Tests for MemoryCollector."""
    
    def test_cpu_memory_tracking(self):
        """Test CPU memory tracking."""
        from aaco.collectors import MemoryCollector
        
        collector = MemoryCollector(track_cpu=True, track_gpu=False)
        
        collector.start()
        # Allocate some memory
        data = [0] * 1_000_000
        collector.stop()
        
        results = collector.get_results()
        
        assert "cpu_used_mb" in results or "rss_mb" in results
    
    @pytest.mark.gpu
    def test_gpu_memory_tracking(self):
        """Test GPU memory tracking (requires GPU)."""
        from aaco.collectors import MemoryCollector
        
        collector = MemoryCollector(track_cpu=False, track_gpu=True)
        
        # This would require actual GPU allocation
        collector.start()
        collector.stop()
        
        results = collector.get_results()
        assert results is not None


class TestCounterCollector:
    """Tests for GPU counter collector."""
    
    def test_available_counters(self):
        """Test listing available counters."""
        from aaco.collectors import CounterCollector
        
        counters = CounterCollector.available_counters()
        
        # Should return a list (may be empty without ROCm)
        assert isinstance(counters, (list, dict))
    
    @pytest.mark.rocm
    def test_counter_collection(self):
        """Test actual counter collection (requires ROCm)."""
        from aaco.collectors import CounterCollector
        
        collector = CounterCollector(
            device_id=0,
            counters=["GRBM_COUNT", "GRBM_GUI_ACTIVE"]
        )
        
        collector.start()
        # Some GPU workload would go here
        collector.stop()
        
        results = collector.get_results()
        assert results is not None


class TestTraceCollector:
    """Tests for trace collector."""
    
    def test_trace_levels(self):
        """Test different trace levels."""
        from aaco.collectors import TraceCollector
        
        for level in ["api", "kernel", "full"]:
            collector = TraceCollector(trace_level=level)
            assert collector.trace_level == level
    
    def test_output_formats(self):
        """Test different output formats."""
        from aaco.collectors import TraceCollector
        
        for fmt in ["json", "perfetto", "chrome"]:
            collector = TraceCollector(output_format=fmt)
            assert collector.output_format == fmt


class TestBaseCollector:
    """Tests for BaseCollector interface."""
    
    def test_interface_methods(self):
        """Test that base collector defines required interface."""
        from aaco.collectors import BaseCollector
        
        # BaseCollector should define these methods
        assert hasattr(BaseCollector, 'start')
        assert hasattr(BaseCollector, 'stop')
        assert hasattr(BaseCollector, 'reset')
        assert hasattr(BaseCollector, 'get_results')
    
    def test_custom_collector(self):
        """Test creating a custom collector."""
        from aaco.collectors import BaseCollector
        
        class CustomCollector(BaseCollector):
            def __init__(self):
                super().__init__()
                self.data = []
            
            def start(self):
                self.data.append("started")
            
            def stop(self):
                self.data.append("stopped")
            
            def reset(self):
                self.data = []
            
            def get_results(self):
                return {"events": self.data}
        
        collector = CustomCollector()
        collector.start()
        collector.stop()
        
        results = collector.get_results()
        assert results["events"] == ["started", "stopped"]


class TestCollectorIntegration:
    """Integration tests for collectors working together."""
    
    def test_multiple_collectors(self):
        """Test running multiple collectors simultaneously."""
        from aaco.collectors import TimingCollector, MemoryCollector
        
        timing = TimingCollector()
        memory = MemoryCollector(track_cpu=True, track_gpu=False)
        
        # Start both
        timing.start()
        memory.start()
        
        # Do some work
        data = [x ** 2 for x in range(10000)]
        
        # Stop both
        memory.stop()
        timing.stop()
        
        timing_results = timing.get_results()
        memory_results = memory.get_results()
        
        assert timing_results is not None
        assert memory_results is not None
