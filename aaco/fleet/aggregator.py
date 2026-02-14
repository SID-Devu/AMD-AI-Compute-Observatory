"""
AACO-SIGMA Fleet Aggregator

Aggregates profiling results across the GPU fleet.
Provides statistical analysis and cross-system comparison.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import math

from .job_scheduler import JobResult


@dataclass
class StatisticalSummary:
    """Statistical summary of measurements."""
    
    # Central tendency
    mean: float = 0.0
    median: float = 0.0
    mode: float = 0.0
    
    # Dispersion
    std_dev: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    range_val: float = 0.0
    
    # Percentiles
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    
    # Distribution
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Sample info
    count: int = 0
    
    # Confidence
    confidence_interval_95: tuple = (0.0, 0.0)


@dataclass
class NodeComparison:
    """Comparison of performance across nodes."""
    
    metric_name: str
    
    # Per-node values
    node_values: Dict[str, float] = field(default_factory=dict)
    
    # Ranking
    best_node: str = ""
    worst_node: str = ""
    
    # Spread
    coefficient_of_variation: float = 0.0
    
    # Outliers
    outlier_nodes: List[str] = field(default_factory=list)


@dataclass
class GPUComparison:
    """Comparison across GPU types."""
    
    metric_name: str
    
    # Per-GPU type
    gpu_values: Dict[str, StatisticalSummary] = field(default_factory=dict)
    
    # Relative performance
    normalized_to: str = ""  # Reference GPU
    normalized_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated results across fleet."""
    
    # Workload identification
    workload_id: str
    
    # Time range
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Sample counts
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    
    # By node
    nodes_participated: List[str] = field(default_factory=list)
    node_sample_counts: Dict[str, int] = field(default_factory=dict)
    
    # Metrics summaries
    latency_summary: StatisticalSummary = field(default_factory=StatisticalSummary)
    throughput_summary: StatisticalSummary = field(default_factory=StatisticalSummary)
    
    # Counter summaries
    counter_summaries: Dict[str, StatisticalSummary] = field(default_factory=dict)
    
    # Comparisons
    node_comparisons: List[NodeComparison] = field(default_factory=list)
    gpu_comparisons: List[GPUComparison] = field(default_factory=list)
    
    # Anomalies detected
    anomaly_count: int = 0
    anomaly_details: List[Dict[str, Any]] = field(default_factory=list)


class FleetAggregator:
    """
    Aggregates and analyzes results across the fleet.
    
    Provides:
    - Statistical analysis of measurements
    - Cross-node comparison
    - Cross-GPU comparison
    - Anomaly detection
    """
    
    def __init__(self):
        self._results: Dict[str, List[JobResult]] = {}  # workload_id -> results
    
    def add_result(self, workload_id: str, result: JobResult) -> None:
        """
        Add a result for aggregation.
        """
        if workload_id not in self._results:
            self._results[workload_id] = []
        self._results[workload_id].append(result)
    
    def aggregate(self, workload_id: str) -> Optional[AggregatedResult]:
        """
        Aggregate all results for a workload.
        
        Args:
            workload_id: Workload to aggregate
            
        Returns:
            Aggregated results or None
        """
        if workload_id not in self._results:
            return None
        
        results = self._results[workload_id]
        if not results:
            return None
        
        agg = AggregatedResult(workload_id=workload_id)
        
        # Basic counts
        agg.total_samples = len(results)
        agg.successful_samples = sum(1 for r in results if r.success)
        agg.failed_samples = agg.total_samples - agg.successful_samples
        
        # By node
        for result in results:
            if result.node_id not in agg.node_sample_counts:
                agg.node_sample_counts[result.node_id] = 0
                agg.nodes_participated.append(result.node_id)
            agg.node_sample_counts[result.node_id] += 1
        
        # Latency summary
        latencies = [r.latency_ms for r in results if r.success and r.latency_ms > 0]
        if latencies:
            agg.latency_summary = self._compute_statistics(latencies)
        
        # Throughput summary
        throughputs = [r.throughput for r in results if r.success and r.throughput > 0]
        if throughputs:
            agg.throughput_summary = self._compute_statistics(throughputs)
        
        # Counter summaries
        counter_values: Dict[str, List[float]] = {}
        for result in results:
            if result.success:
                for counter, value in result.counters.items():
                    if counter not in counter_values:
                        counter_values[counter] = []
                    counter_values[counter].append(value)
        
        for counter, values in counter_values.items():
            agg.counter_summaries[counter] = self._compute_statistics(values)
        
        # Node comparison
        agg.node_comparisons.append(
            self._compare_nodes(results, "latency_ms", lambda r: r.latency_ms)
        )
        
        # Detect anomalies
        anomalies = self._detect_anomalies(results)
        agg.anomaly_count = len(anomalies)
        agg.anomaly_details = anomalies
        
        return agg
    
    def _compute_statistics(self, values: List[float]) -> StatisticalSummary:
        """Compute statistical summary of values."""
        n = len(values)
        if n == 0:
            return StatisticalSummary()
        
        sorted_vals = sorted(values)
        
        # Central tendency
        mean = sum(values) / n
        median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        
        # Dispersion
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
        std_dev = math.sqrt(variance)
        min_val = sorted_vals[0]
        max_val = sorted_vals[-1]
        
        # Percentiles
        def percentile(vals, p):
            idx = int(len(vals) * p / 100)
            return vals[min(idx, len(vals) - 1)]
        
        p50 = percentile(sorted_vals, 50)
        p90 = percentile(sorted_vals, 90)
        p95 = percentile(sorted_vals, 95)
        p99 = percentile(sorted_vals, 99)
        
        # Skewness and kurtosis
        if std_dev > 0 and n > 2:
            skewness = sum((x - mean) ** 3 for x in values) / (n * std_dev ** 3)
            kurtosis = sum((x - mean) ** 4 for x in values) / (n * std_dev ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        # Confidence interval (95%, assuming normal distribution)
        if n > 1:
            se = std_dev / math.sqrt(n)
            ci_95 = (mean - 1.96 * se, mean + 1.96 * se)
        else:
            ci_95 = (mean, mean)
        
        return StatisticalSummary(
            mean=mean,
            median=median,
            std_dev=std_dev,
            variance=variance,
            min_val=min_val,
            max_val=max_val,
            range_val=max_val - min_val,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
            skewness=skewness,
            kurtosis=kurtosis,
            count=n,
            confidence_interval_95=ci_95,
        )
    
    def _compare_nodes(self, results: List[JobResult],
                      metric_name: str,
                      extractor: callable) -> NodeComparison:
        """Compare metric across nodes."""
        comparison = NodeComparison(metric_name=metric_name)
        
        # Group by node
        node_values: Dict[str, List[float]] = {}
        for result in results:
            if result.success:
                value = extractor(result)
                if value > 0:
                    if result.node_id not in node_values:
                        node_values[result.node_id] = []
                    node_values[result.node_id].append(value)
        
        # Compute mean per node
        for node_id, values in node_values.items():
            comparison.node_values[node_id] = sum(values) / len(values)
        
        if comparison.node_values:
            # Best/worst
            comparison.best_node = min(comparison.node_values, key=comparison.node_values.get)
            comparison.worst_node = max(comparison.node_values, key=comparison.node_values.get)
            
            # Coefficient of variation
            values = list(comparison.node_values.values())
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            comparison.coefficient_of_variation = std / mean if mean > 0 else 0
            
            # Outliers (using IQR method)
            sorted_values = sorted(values)
            n = len(sorted_values)
            q1 = sorted_values[n // 4]
            q3 = sorted_values[3 * n // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for node_id, value in comparison.node_values.items():
                if value < lower_bound or value > upper_bound:
                    comparison.outlier_nodes.append(node_id)
        
        return comparison
    
    def _detect_anomalies(self, results: List[JobResult]) -> List[Dict[str, Any]]:
        """Detect anomalies in results."""
        anomalies = []
        
        # Get successful results
        successful = [r for r in results if r.success]
        if len(successful) < 3:
            return anomalies
        
        # Check latency outliers
        latencies = [r.latency_ms for r in successful]
        mean_lat = sum(latencies) / len(latencies)
        std_lat = math.sqrt(sum((l - mean_lat) ** 2 for l in latencies) / len(latencies))
        
        for result in successful:
            if std_lat > 0:
                z_score = (result.latency_ms - mean_lat) / std_lat
                if abs(z_score) > 3:
                    anomalies.append({
                        "type": "latency_outlier",
                        "job_id": result.job_id,
                        "node_id": result.node_id,
                        "value": result.latency_ms,
                        "z_score": z_score,
                    })
        
        # Check for failing nodes
        node_failures: Dict[str, int] = {}
        for result in results:
            if not result.success:
                if result.node_id not in node_failures:
                    node_failures[result.node_id] = 0
                node_failures[result.node_id] += 1
        
        for node_id, failures in node_failures.items():
            if failures > 2:
                anomalies.append({
                    "type": "node_failures",
                    "node_id": node_id,
                    "failure_count": failures,
                })
        
        return anomalies
    
    def compare_gpus(self, workload_id: str,
                    gpu_node_mapping: Dict[str, str]) -> GPUComparison:
        """
        Compare performance across GPU types.
        
        Args:
            workload_id: Workload to compare
            gpu_node_mapping: Map node_id -> gpu_type
            
        Returns:
            GPU comparison results
        """
        comparison = GPUComparison(metric_name="latency_ms")
        
        if workload_id not in self._results:
            return comparison
        
        # Group by GPU type
        gpu_values: Dict[str, List[float]] = {}
        
        for result in self._results[workload_id]:
            if result.success:
                gpu_type = gpu_node_mapping.get(result.node_id, "unknown")
                if gpu_type not in gpu_values:
                    gpu_values[gpu_type] = []
                gpu_values[gpu_type].append(result.latency_ms)
        
        # Compute statistics per GPU
        for gpu_type, values in gpu_values.items():
            comparison.gpu_values[gpu_type] = self._compute_statistics(values)
        
        # Normalize to fastest
        if comparison.gpu_values:
            fastest_mean = min(s.mean for s in comparison.gpu_values.values())
            comparison.normalized_to = min(
                comparison.gpu_values.keys(),
                key=lambda k: comparison.gpu_values[k].mean
            )
            
            for gpu_type, summary in comparison.gpu_values.items():
                comparison.normalized_values[gpu_type] = summary.mean / fastest_mean if fastest_mean > 0 else 0
        
        return comparison
    
    def export_summary(self, workload_id: str) -> Dict[str, Any]:
        """Export summary as dictionary."""
        agg = self.aggregate(workload_id)
        if not agg:
            return {}
        
        return {
            "workload_id": agg.workload_id,
            "total_samples": agg.total_samples,
            "successful": agg.successful_samples,
            "failed": agg.failed_samples,
            "nodes": agg.nodes_participated,
            "latency": {
                "mean_ms": agg.latency_summary.mean,
                "p50_ms": agg.latency_summary.p50,
                "p95_ms": agg.latency_summary.p95,
                "p99_ms": agg.latency_summary.p99,
                "std_dev": agg.latency_summary.std_dev,
            },
            "throughput": {
                "mean": agg.throughput_summary.mean,
                "p50": agg.throughput_summary.p50,
            },
            "anomalies": agg.anomaly_count,
        }
