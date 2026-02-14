"""
rocprof Output Parser
Parses rocprof CSV traces and computes kernel-level metrics.
"""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from aaco.core.schema import KernelExecution, KernelSummary, KernelMetrics

logger = logging.getLogger(__name__)


@dataclass
class KernelTrace:
    """Parsed kernel execution trace."""

    kernel_name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    queue_id: int = 0
    device_id: int = 0
    grid_size: Optional[Tuple[int, int, int]] = None
    block_size: Optional[Tuple[int, int, int]] = None

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


class RocprofParser:
    """
    Parser for rocprof output files.
    Extracts kernel traces and computes summary statistics.
    """

    # Common column name variations
    KERNEL_NAME_COLS = ["Name", "KernelName", "Kernel", "kernel_name", "name"]
    START_COLS = ["BeginNs", "Start", "start_ns", "StartNs", "begin"]
    END_COLS = ["EndNs", "End", "end_ns", "EndNs", "end"]
    DURATION_COLS = ["DurationNs", "Duration", "dur", "duration_ns", "DurNs"]

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path
        self.traces: List[KernelTrace] = []
        self.raw_data: List[Dict] = []

        if csv_path:
            self.parse(csv_path)

    def parse(self, csv_path: Path) -> List[KernelTrace]:
        """Parse rocprof CSV file and extract kernel traces."""
        self.csv_path = csv_path
        self.traces = []
        self.raw_data = []

        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return []

        try:
            with open(csv_path, "r", newline="") as f:
                # Try to detect dialect
                f.read(4096)
                f.seek(0)

                # Skip any comment lines at the start
                lines = []
                for line in f:
                    if not line.startswith("#") and line.strip():
                        lines.append(line)

                if not lines:
                    logger.warning(f"No data in CSV: {csv_path}")
                    return []

                # Parse CSV
                reader = csv.DictReader(lines)

                for row in reader:
                    self.raw_data.append(row)
                    trace = self._parse_row(row)
                    if trace:
                        self.traces.append(trace)

            logger.info(f"Parsed {len(self.traces)} kernel traces from {csv_path}")
            return self.traces

        except Exception as e:
            logger.error(f"Failed to parse CSV {csv_path}: {e}")
            return []

    def _parse_row(self, row: Dict) -> Optional[KernelTrace]:
        """Parse a single CSV row into a KernelTrace."""
        # Find kernel name
        kernel_name = None
        for col in self.KERNEL_NAME_COLS:
            if col in row and row[col]:
                kernel_name = row[col].strip()
                break

        if not kernel_name:
            return None

        # Clean kernel name (remove template parameters if too long)
        if len(kernel_name) > 200:
            kernel_name = re.sub(r"<[^>]+>", "<...>", kernel_name)

        # Find timing columns
        start_ns = 0
        end_ns = 0
        duration_ns = 0

        for col in self.START_COLS:
            if col in row:
                try:
                    start_ns = int(float(row[col]))
                    break
                except (ValueError, TypeError):
                    pass

        for col in self.END_COLS:
            if col in row:
                try:
                    end_ns = int(float(row[col]))
                    break
                except (ValueError, TypeError):
                    pass

        for col in self.DURATION_COLS:
            if col in row:
                try:
                    duration_ns = int(float(row[col]))
                    break
                except (ValueError, TypeError):
                    pass

        # Calculate duration if not provided
        if duration_ns == 0 and end_ns > start_ns:
            duration_ns = end_ns - start_ns

        if duration_ns <= 0:
            return None

        # Extract queue/device IDs if available
        queue_id = 0
        device_id = 0

        if "QueueId" in row or "queue_id" in row:
            try:
                queue_id = int(row.get("QueueId", row.get("queue_id", 0)))
            except (ValueError, TypeError):
                pass

        if "DeviceId" in row or "device_id" in row or "gpu-id" in row:
            try:
                device_id = int(row.get("DeviceId", row.get("device_id", row.get("gpu-id", 0))))
            except (ValueError, TypeError):
                pass

        # Extract grid/block sizes if available
        grid_size = None
        block_size = None

        if "grd" in row:
            try:
                grid_size = tuple(map(int, row["grd"].split(",")))
            except:
                pass

        if "wgr" in row:
            try:
                block_size = tuple(map(int, row["wgr"].split(",")))
            except:
                pass

        return KernelTrace(
            kernel_name=kernel_name,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ns=duration_ns,
            queue_id=queue_id,
            device_id=device_id,
            grid_size=grid_size,
            block_size=block_size,
        )

    def compute_summaries(self, top_n: int = 20) -> List[KernelSummary]:
        """Compute per-kernel summary statistics."""
        if not self.traces:
            return []

        # Group by kernel name
        kernel_times: Dict[str, List[int]] = {}
        for trace in self.traces:
            if trace.kernel_name not in kernel_times:
                kernel_times[trace.kernel_name] = []
            kernel_times[trace.kernel_name].append(trace.duration_ns)

        # Compute total time for percentage calculation
        total_time_ns = sum(t.duration_ns for t in self.traces)

        summaries = []
        for kernel_name, times in kernel_times.items():
            arr = np.array(times)
            total_ns = np.sum(arr)

            summaries.append(
                KernelSummary(
                    kernel_name=kernel_name,
                    calls=len(times),
                    total_time_ms=total_ns / 1_000_000,
                    avg_time_us=float(np.mean(arr)) / 1000,
                    min_time_us=float(np.min(arr)) / 1000,
                    max_time_us=float(np.max(arr)) / 1000,
                    std_time_us=float(np.std(arr)) / 1000,
                    pct_total=100 * total_ns / total_time_ns if total_time_ns > 0 else 0,
                )
            )

        # Sort by total time descending
        summaries.sort(key=lambda s: s.total_time_ms, reverse=True)

        return summaries[:top_n]

    def compute_metrics(
        self,
        wall_time_ms: float = 0,
        onnx_node_count: int = 0,
        microkernel_threshold_us: float = 10.0,
    ) -> KernelMetrics:
        """
        Compute derived kernel metrics for bottleneck analysis.

        Args:
            wall_time_ms: Total wall clock time of the inference
            onnx_node_count: Number of nodes in ONNX graph (for KAR)
            microkernel_threshold_us: Threshold for "microkernel" classification
        """
        if not self.traces:
            return KernelMetrics(
                total_kernel_count=0,
                unique_kernel_count=0,
                total_kernel_time_ms=0,
                avg_kernel_duration_us=0,
                microkernel_count=0,
                microkernel_pct=0,
                microkernel_threshold_us=microkernel_threshold_us,
                launch_rate_per_sec=0,
                launch_tax_score=0,
                kernel_amplification_ratio=0,
                gpu_active_ratio=0,
                top_kernels=[],
            )

        # Basic counts
        total_count = len(self.traces)
        unique_count = len(set(t.kernel_name for t in self.traces))

        # Timing
        durations_us = [t.duration_ns / 1000.0 for t in self.traces]
        total_kernel_time_ms = sum(durations_us) / 1000.0
        avg_duration_us = np.mean(durations_us)

        # Microkernel analysis
        threshold_ns = microkernel_threshold_us * 1000
        microkernel_count = sum(1 for t in self.traces if t.duration_ns < threshold_ns)
        microkernel_pct = 100 * microkernel_count / total_count if total_count > 0 else 0

        # Launch rate (kernels per second)
        if self.traces:
            time_span_s = (self.traces[-1].end_ns - self.traces[0].start_ns) / 1e9
            launch_rate = total_count / time_span_s if time_span_s > 0 else 0
        else:
            launch_rate = 0

        # Launch tax score: combination of microkernel % and launch rate
        launch_tax_score = (microkernel_pct / 100) * (launch_rate / 1000)

        # Kernel Amplification Ratio
        kar = total_count / onnx_node_count if onnx_node_count > 0 else 0

        # GPU active ratio
        gpu_active_ratio = total_kernel_time_ms / wall_time_ms if wall_time_ms > 0 else 0

        # Top kernels
        top_kernels = self.compute_summaries(top_n=10)

        return KernelMetrics(
            total_kernel_count=total_count,
            unique_kernel_count=unique_count,
            total_kernel_time_ms=total_kernel_time_ms,
            avg_kernel_duration_us=avg_duration_us,
            microkernel_count=microkernel_count,
            microkernel_pct=microkernel_pct,
            microkernel_threshold_us=microkernel_threshold_us,
            launch_rate_per_sec=launch_rate,
            launch_tax_score=launch_tax_score,
            kernel_amplification_ratio=kar,
            gpu_active_ratio=gpu_active_ratio,
            top_kernels=top_kernels,
        )

    def to_dataframe(self):
        """Convert traces to pandas DataFrame."""
        import pandas as pd

        if not self.traces:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "kernel_name": t.kernel_name,
                    "start_ns": t.start_ns,
                    "end_ns": t.end_ns,
                    "duration_ns": t.duration_ns,
                    "duration_us": t.duration_us,
                    "duration_ms": t.duration_ms,
                    "queue_id": t.queue_id,
                    "device_id": t.device_id,
                }
                for t in self.traces
            ]
        )

    def to_executions(self) -> List[KernelExecution]:
        """Convert traces to KernelExecution schema objects."""
        return [
            KernelExecution(
                t_start_ns=t.start_ns,
                t_end_ns=t.end_ns,
                dur_ns=t.duration_ns,
                kernel_name=t.kernel_name,
                queue_id=t.queue_id,
            )
            for t in self.traces
        ]

    def get_kernel_histogram(
        self,
        bins: int = 50,
        log_scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get histogram of kernel durations.

        Returns:
            Tuple of (bin_edges, counts)
        """
        if not self.traces:
            return np.array([]), np.array([])

        durations_us = [t.duration_us for t in self.traces]

        if log_scale:
            # Use log-spaced bins
            min_dur = max(0.1, min(durations_us))
            max_dur = max(durations_us)
            bin_edges = np.logspace(np.log10(min_dur), np.log10(max_dur), bins + 1)
        else:
            bin_edges = bins

        counts, edges = np.histogram(durations_us, bins=bin_edges)
        return edges, counts
