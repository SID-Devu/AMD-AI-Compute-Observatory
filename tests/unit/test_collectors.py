"""
AACO Test Suite - Collectors Tests  
===================================
Tests for data collectors.

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import pytest
from unittest.mock import patch, MagicMock

from aaco.collectors.sys_sampler import SystemSampler, SystemSample
from aaco.collectors.clocks import ClockMonitor


class TestSystemSampler:
    """Tests for SystemSampler."""
    
    def test_create_sampler(self):
        """Test SystemSampler instantiation."""
        sampler = SystemSampler()
        assert sampler is not None
    
    def test_sampler_default_interval(self):
        """Test sampler has default interval."""
        sampler = SystemSampler()
        assert sampler.interval_ms == 200
    
    def test_sampler_custom_interval(self):
        """Test sampler with custom interval."""
        sampler = SystemSampler(interval_ms=100)
        assert sampler.interval_ms == 100
    
    def test_sampler_has_samples_list(self):
        """Test sampler has samples list."""
        sampler = SystemSampler()
        assert isinstance(sampler.samples, list)


class TestSystemSample:
    """Tests for SystemSample dataclass."""
    
    def test_create_sample(self):
        """Test SystemSample instantiation."""
        sample = SystemSample(
            t_ns=1000000,
            cpu_pct=25.0,
            rss_mb=512.0,
            ctx_switches=100,
            ctx_switches_delta=5,
            majfaults=10,
            majfault_delta=1,
            load1=1.5,
            load5=1.2,
            procs_running=3
        )
        assert sample.t_ns == 1000000
        assert sample.cpu_pct == 25.0
        assert sample.rss_mb == 512.0


class TestClockMonitor:
    """Tests for ClockMonitor."""
    
    def test_create_monitor(self):
        """Test ClockMonitor instantiation."""
        monitor = ClockMonitor()
        assert monitor is not None
    
    def test_monitor_has_methods(self):
        """Test ClockMonitor has expected methods."""
        monitor = ClockMonitor()
        assert hasattr(monitor, 'get_cpu_governor')
        assert hasattr(monitor, 'get_all_cpu_governors')
        assert hasattr(monitor, 'get_cpu_frequency')


class TestROCmSMISampler:
    """Tests for ROCmSMISampler (may require ROCm)."""
    
    def test_import_sampler(self):
        """Test ROCmSMISampler can be imported."""
        from aaco.collectors.rocm_smi_sampler import ROCmSMISampler, GPUSample
        assert ROCmSMISampler is not None
        assert GPUSample is not None
