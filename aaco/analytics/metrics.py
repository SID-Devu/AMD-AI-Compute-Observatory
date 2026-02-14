"""
Derived Metrics Engine
Computes phase-specific and aggregated performance metrics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from aaco.core.schema import (
    InferenceIteration,
    KernelMetrics,
    DerivedMetrics,
    PhaseMetrics,
)
from aaco.collectors.sys_sampler import SystemSample
from aaco.collectors.rocm_smi_sampler import GPUSample

logger = logging.getLogger(__name__)


class DerivedMetricsEngine:
    """
    Computes derived performance metrics from raw measurements.
    Combines inference timing, kernel profiling, and telemetry into unified metrics.
    """
    
    def __init__(self):
        self.inference_results: List[InferenceIteration] = []
        self.kernel_metrics: Optional[KernelMetrics] = None
        self.sys_samples: List[SystemSample] = []
        self.gpu_samples: List[GPUSample] = []
    
    def add_inference_results(self, results: List[InferenceIteration]) -> None:
        """Add inference timing results."""
        self.inference_results.extend(results)
    
    def set_kernel_metrics(self, metrics: KernelMetrics) -> None:
        """Set kernel profiling metrics."""
        self.kernel_metrics = metrics
    
    def add_system_samples(self, samples: List[SystemSample]) -> None:
        """Add system telemetry samples."""
        self.sys_samples.extend(samples)
    
    def add_gpu_samples(self, samples: List[GPUSample]) -> None:
        """Add GPU telemetry samples."""
        self.gpu_samples.extend(samples)
    
    def compute(self) -> DerivedMetrics:
        """
        Compute all derived metrics from collected data.
        
        Returns:
            DerivedMetrics containing all computed values.
        """
        # Compute per-phase metrics
        warmup_phase = self._compute_phase("warmup")
        measurement_phase = self._compute_phase("measurement")
        
        # Compute throughput metrics
        throughput = self._compute_throughput()
        
        # Compute efficiency metrics
        efficiency = self._compute_efficiency()
        
        # Aggregate latency statistics
        latency = self._compute_latency_stats()
        
        # System resource utilization
        system = self._compute_system_utilization()
        
        # GPU utilization
        gpu = self._compute_gpu_utilization()
        
        return DerivedMetrics(
            warmup_phase=warmup_phase,
            measurement_phase=measurement_phase,
            throughput=throughput,
            efficiency=efficiency,
            latency=latency,
            system=system,
            gpu=gpu,
        )
    
    def _compute_phase(self, phase: str) -> PhaseMetrics:
        """Compute metrics for a specific phase."""
        phase_results = [r for r in self.inference_results if r.phase == phase]
        
        if not phase_results:
            return PhaseMetrics(
                name=phase,
                iterations=0,
                total_time_ms=0,
                mean_ms=0,
                std_ms=0,
                p50_ms=0,
                p90_ms=0,
                p99_ms=0,
                min_ms=0,
                max_ms=0,
                iqr_ms=0,
                cov_pct=0,
            )
        
        latencies = np.array([r.latency_ms for r in phase_results])
        
        return PhaseMetrics(
            name=phase,
            iterations=len(phase_results),
            total_time_ms=float(np.sum(latencies)),
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p90_ms=float(np.percentile(latencies, 90)),
            p99_ms=float(np.percentile(latencies, 99)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            iqr_ms=float(np.percentile(latencies, 75) - np.percentile(latencies, 25)),
            cov_pct=float(100 * np.std(latencies) / np.mean(latencies)) if np.mean(latencies) > 0 else 0,
        )
    
    def _compute_throughput(self) -> Dict[str, float]:
        """Compute throughput metrics."""
        measure_results = [r for r in self.inference_results if r.phase == "measurement"]
        
        if not measure_results:
            return {"inferences_per_sec": 0, "tokens_per_sec": 0, "samples_per_sec": 0}
        
        total_time_s = sum(r.latency_ms for r in measure_results) / 1000.0
        count = len(measure_results)
        
        ips = count / total_time_s if total_time_s > 0 else 0
        
        return {
            "inferences_per_sec": ips,
            "samples_per_sec": ips,  # Alias for batch_size=1
            "mean_latency_ms": sum(r.latency_ms for r in measure_results) / count if count > 0 else 0,
        }
    
    def _compute_efficiency(self) -> Dict[str, float]:
        """Compute efficiency metrics combining kernel and wall clock data."""
        if not self.kernel_metrics:
            return {
                "gpu_active_ratio": 0,
                "kernel_amplification_ratio": 0,
                "launch_overhead_pct": 0,
            }
        
        # GPU Active Ratio: kernel time / wall time
        gpu_active = self.kernel_metrics.gpu_active_ratio
        
        # KAR
        kar = self.kernel_metrics.kernel_amplification_ratio
        
        # Launch overhead estimation
        # High microkernel % + high launch rate = high launch overhead
        launch_overhead = self.kernel_metrics.launch_tax_score * 10
        
        return {
            "gpu_active_ratio": gpu_active,
            "kernel_amplification_ratio": kar,
            "launch_overhead_pct": min(100, launch_overhead),
            "microkernel_pct": self.kernel_metrics.microkernel_pct,
        }
    
    def _compute_latency_stats(self) -> Dict[str, float]:
        """Compute aggregate latency statistics."""
        measure_results = [r for r in self.inference_results if r.phase == "measurement"]
        
        if not measure_results:
            return {}
        
        latencies = np.array([r.latency_ms for r in measure_results])
        
        # Stability metrics
        warmup_results = [r for r in self.inference_results if r.phase == "warmup"]
        warmup_effect = 0
        if warmup_results and measure_results:
            warmup_mean = np.mean([r.latency_ms for r in warmup_results])
            measure_mean = np.mean(latencies)
            warmup_effect = ((warmup_mean - measure_mean) / measure_mean * 100) if measure_mean > 0 else 0
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "jitter_ms": float(np.max(latencies) - np.min(latencies)),
            "cov_pct": float(100 * np.std(latencies) / np.mean(latencies)) if np.mean(latencies) > 0 else 0,
            "warmup_effect_pct": warmup_effect,
        }
    
    def _compute_system_utilization(self) -> Dict[str, float]:
        """Compute system resource utilization statistics."""
        if not self.sys_samples:
            return {}
        
        cpu_pcts = [s.cpu_pct for s in self.sys_samples if s.cpu_pct is not None]
        rss_mbs = [s.rss_mb for s in self.sys_samples if s.rss_mb is not None]
        
        result = {}
        
        if cpu_pcts:
            result["cpu_mean_pct"] = float(np.mean(cpu_pcts))
            result["cpu_max_pct"] = float(np.max(cpu_pcts))
            result["cpu_std_pct"] = float(np.std(cpu_pcts))
        
        if rss_mbs:
            result["rss_mean_mb"] = float(np.mean(rss_mbs))
            result["rss_max_mb"] = float(np.max(rss_mbs))
            result["rss_delta_mb"] = float(max(rss_mbs) - min(rss_mbs))
        
        return result
    
    def _compute_gpu_utilization(self) -> Dict[str, float]:
        """Compute GPU utilization statistics."""
        if not self.gpu_samples:
            return {}
        
        # Extract values, filtering None
        def safe_extract(samples: List, attr: str) -> List[float]:
            return [getattr(s, attr) for s in samples if getattr(s, attr, None) is not None]
        
        gpu_utils = safe_extract(self.gpu_samples, "gpu_util_pct")
        mem_utils = safe_extract(self.gpu_samples, "mem_util_pct")
        powers = safe_extract(self.gpu_samples, "power_w")
        temps = safe_extract(self.gpu_samples, "temp_c")
        vram_useds = safe_extract(self.gpu_samples, "vram_used_mb")
        sclks = safe_extract(self.gpu_samples, "sclk_mhz")
        mclks = safe_extract(self.gpu_samples, "mclk_mhz")
        
        result = {}
        
        if gpu_utils:
            result["gpu_util_mean_pct"] = float(np.mean(gpu_utils))
            result["gpu_util_max_pct"] = float(np.max(gpu_utils))
        
        if mem_utils:
            result["mem_util_mean_pct"] = float(np.mean(mem_utils))
            result["mem_util_max_pct"] = float(np.max(mem_utils))
        
        if powers:
            result["power_mean_w"] = float(np.mean(powers))
            result["power_max_w"] = float(np.max(powers))
        
        if temps:
            result["temp_mean_c"] = float(np.mean(temps))
            result["temp_max_c"] = float(np.max(temps))
        
        if vram_useds:
            result["vram_mean_mb"] = float(np.mean(vram_useds))
            result["vram_max_mb"] = float(np.max(vram_useds))
        
        if sclks:
            result["sclk_mean_mhz"] = float(np.mean(sclks))
            result["sclk_range_mhz"] = float(np.max(sclks) - np.min(sclks))
        
        if mclks:
            result["mclk_mean_mhz"] = float(np.mean(mclks))
        
        return result
    
    def summary_dict(self) -> Dict[str, Any]:
        """Get a dictionary summary of all computed metrics."""
        metrics = self.compute()
        
        return {
            "warmup_iterations": metrics.warmup_phase.iterations,
            "warmup_mean_ms": metrics.warmup_phase.mean_ms,
            "measurement_iterations": metrics.measurement_phase.iterations,
            "measurement_mean_ms": metrics.measurement_phase.mean_ms,
            "measurement_std_ms": metrics.measurement_phase.std_ms,
            "measurement_p99_ms": metrics.measurement_phase.p99_ms,
            "throughput_ips": metrics.throughput.get("inferences_per_sec", 0),
            "gpu_active_ratio": metrics.efficiency.get("gpu_active_ratio", 0),
            "kar": metrics.efficiency.get("kernel_amplification_ratio", 0),
            "microkernel_pct": metrics.efficiency.get("microkernel_pct", 0),
            **metrics.gpu,
            **metrics.system,
        }
