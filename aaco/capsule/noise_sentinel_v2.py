"""
AACO-SIGMA Noise Sentinel V2

Advanced interference detection for measurement integrity.
Detects: IRQ storms, memory reclaim, IO pressure, scheduler interference,
context switch storms, page faults, NUMA migrations.
"""

import re
import json
import threading
import time
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import deque


class NoiseSourceV2(Enum):
    """Sources of measurement interference."""

    IRQ_STORM = auto()  # High interrupt rate
    SOFTIRQ_STORM = auto()  # High softirq rate
    CONTEXT_SWITCH = auto()  # Excessive context switches
    MEMORY_RECLAIM = auto()  # Page reclaim activity
    MEMORY_PRESSURE = auto()  # PSI memory pressure
    IO_PRESSURE = auto()  # PSI IO pressure
    CPU_PRESSURE = auto()  # PSI CPU pressure
    PAGE_FAULT = auto()  # Major page faults
    NUMA_MIGRATION = auto()  # Cross-NUMA memory migration
    SCHEDULER_DELAY = auto()  # High runqueue latency
    WORKQUEUE = auto()  # Kernel workqueue storms
    NETWORK = auto()  # Network interrupt storms
    DISK = auto()  # Disk IO interference
    POWER_CAPPING = auto()  # Power limit throttling
    THERMAL = auto()  # Thermal throttling


class NoiseClassification(Enum):
    """Classification of noise severity."""

    NEGLIGIBLE = auto()  # < 1% impact
    MINOR = auto()  # 1-5% impact
    MODERATE = auto()  # 5-10% impact
    SIGNIFICANT = auto()  # 10-20% impact
    SEVERE = auto()  # > 20% impact


@dataclass
class NoiseEventV2:
    """A detected noise event."""

    timestamp_ns: int
    source: NoiseSourceV2
    classification: NoiseClassification

    # Measured values
    measured_value: float = 0.0
    threshold: float = 0.0
    baseline_value: float = 0.0

    # Duration
    duration_ns: int = 0

    # Impact estimate
    estimated_impact_pct: float = 0.0

    # Context
    cpu_id: Optional[int] = None
    process_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.name
        d["classification"] = self.classification.name
        return d


@dataclass
class NoiseReportV2:
    """Comprehensive noise report for a measurement session."""

    session_id: str = ""
    duration_ns: int = 0

    # Overall assessment
    is_clean: bool = True
    noise_score: float = 0.0  # 0 = clean, 100 = unusable

    # Event counts by source
    event_counts: Dict[str, int] = field(default_factory=dict)

    # Most significant events
    significant_events: List[Dict[str, Any]] = field(default_factory=list)

    # Estimated total impact
    estimated_total_impact_pct: float = 0.0

    # Time with noise (% of session)
    noisy_time_pct: float = 0.0

    # Baseline statistics
    baseline_context_switches_per_sec: float = 0.0
    baseline_irqs_per_sec: float = 0.0

    # Peak values
    peak_context_switches_per_sec: float = 0.0
    peak_irqs_per_sec: float = 0.0
    peak_memory_pressure: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class NoiseThresholdsV2:
    """Thresholds for noise detection."""

    # Context switches (per second per effective core)
    context_switch_warning: int = 8000
    context_switch_severe: int = 20000

    # IRQ rate (per second)
    irq_warning: int = 5000
    irq_severe: int = 15000

    # SoftIRQ rate
    softirq_warning: int = 3000
    softirq_severe: int = 10000

    # Memory pressure (PSI avg10)
    memory_pressure_warning: float = 5.0
    memory_pressure_severe: float = 20.0

    # IO pressure (PSI avg10)
    io_pressure_warning: float = 10.0
    io_pressure_severe: float = 30.0

    # CPU pressure (PSI avg10)
    cpu_pressure_warning: float = 5.0
    cpu_pressure_severe: float = 20.0

    # Page faults (major, per second)
    page_fault_warning: int = 100
    page_fault_severe: int = 1000

    # Runqueue latency (microseconds)
    runqueue_latency_warning: int = 1000
    runqueue_latency_severe: int = 10000


class InterferenceSampler:
    """
    Collects interference metrics from various kernel interfaces.
    """

    PROC_STAT = Path("/proc/stat")
    PROC_VMSTAT = Path("/proc/vmstat")
    PROC_PRESSURE = Path("/proc/pressure")
    PROC_SOFTIRQS = Path("/proc/softirqs")

    def __init__(self):
        self._prev_sample: Optional[Dict[str, Any]] = None

    def sample(self) -> Dict[str, Any]:
        """Collect a comprehensive interference sample."""
        sample = {"timestamp_ns": time.time_ns()}

        # Parse /proc/stat
        stat_data = self._parse_proc_stat()
        sample.update(stat_data)

        # Parse /proc/vmstat
        vmstat_data = self._parse_vmstat()
        sample.update(vmstat_data)

        # Parse PSI
        psi_data = self._parse_psi()
        sample.update(psi_data)

        # Parse softirqs
        softirq_data = self._parse_softirqs()
        sample.update(softirq_data)

        return sample

    def sample_delta(self) -> Optional[Dict[str, Any]]:
        """Get sample with deltas from previous sample."""
        current = self.sample()

        if self._prev_sample is None:
            self._prev_sample = current
            return None

        prev = self._prev_sample
        self._prev_sample = current

        # Calculate time delta
        dt_ns = current["timestamp_ns"] - prev["timestamp_ns"]
        dt_sec = dt_ns / 1e9

        delta = {
            "timestamp_ns": current["timestamp_ns"],
            "delta_ns": dt_ns,
        }

        # Calculate rates
        rate_keys = [
            "context_switches",
            "irq_total",
            "softirq_total",
            "pgfault",
            "pgmajfault",
            "pswpin",
            "pswpout",
            "pgpgin",
            "pgpgout",
        ]

        for key in rate_keys:
            if key in current and key in prev:
                delta_val = current[key] - prev[key]
                delta[f"{key}_per_sec"] = delta_val / dt_sec if dt_sec > 0 else 0

        # Copy current PSI values
        for key in ["memory_psi_avg10", "io_psi_avg10", "cpu_psi_avg10"]:
            if key in current:
                delta[key] = current[key]

        return delta

    def _parse_proc_stat(self) -> Dict[str, Any]:
        """Parse /proc/stat for system-wide metrics."""
        data = {}

        try:
            content = self.PROC_STAT.read_text()
            for line in content.split("\n"):
                if line.startswith("ctxt "):
                    data["context_switches"] = int(line.split()[1])
                elif line.startswith("intr "):
                    parts = line.split()
                    data["irq_total"] = int(parts[1])
                elif line.startswith("softirq "):
                    parts = line.split()
                    data["softirq_total"] = int(parts[1])
                elif line.startswith("procs_running "):
                    data["procs_running"] = int(line.split()[1])
                elif line.startswith("procs_blocked "):
                    data["procs_blocked"] = int(line.split()[1])
        except:
            pass

        return data

    def _parse_vmstat(self) -> Dict[str, Any]:
        """Parse /proc/vmstat for memory metrics."""
        data = {}

        try:
            content = self.PROC_VMSTAT.read_text()
            for line in content.split("\n"):
                parts = line.split()
                if len(parts) != 2:
                    continue

                key, value = parts
                if key in [
                    "pgfault",
                    "pgmajfault",
                    "pswpin",
                    "pswpout",
                    "pgpgin",
                    "pgpgout",
                    "numa_miss",
                    "numa_foreign",
                    "nr_writeback",
                    "nr_dirty",
                ]:
                    data[key] = int(value)
        except:
            pass

        return data

    def _parse_psi(self) -> Dict[str, Any]:
        """Parse /proc/pressure/* for PSI metrics."""
        data = {}

        for resource in ["memory", "io", "cpu"]:
            psi_file = self.PROC_PRESSURE / resource
            if psi_file.exists():
                try:
                    content = psi_file.read_text()
                    for line in content.split("\n"):
                        if line.startswith("some"):
                            # Parse: some avg10=X.XX avg60=Y.YY avg300=Z.ZZ total=N
                            match = re.search(r"avg10=(\d+\.?\d*)", line)
                            if match:
                                data[f"{resource}_psi_avg10"] = float(match.group(1))
                except:
                    pass

        return data

    def _parse_softirqs(self) -> Dict[str, Any]:
        """Parse /proc/softirqs for softirq breakdown."""
        data = {}

        try:
            content = self.PROC_SOFTIRQS.read_text()
            lines = content.strip().split("\n")

            if len(lines) > 1:
                # Header line has CPU columns
                # Following lines have: TYPE: counts...
                for line in lines[1:]:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        irq_type = parts[0].strip()
                        counts = [int(c) for c in parts[1].split() if c.isdigit()]
                        data[f"softirq_{irq_type.lower()}"] = sum(counts)
        except:
            pass

        return data


class InterferenceClassifier:
    """
    Classifies detected interference and estimates impact.
    """

    def __init__(self, thresholds: Optional[NoiseThresholdsV2] = None):
        self.thresholds = thresholds or NoiseThresholdsV2()

    def classify_context_switches(self, rate: float) -> Tuple[NoiseClassification, float]:
        """Classify context switch rate."""
        if rate < self.thresholds.context_switch_warning:
            return NoiseClassification.NEGLIGIBLE, 0.0
        elif rate < self.thresholds.context_switch_severe:
            impact = (
                (rate - self.thresholds.context_switch_warning)
                / (self.thresholds.context_switch_severe - self.thresholds.context_switch_warning)
                * 10
            )
            return NoiseClassification.MODERATE, min(impact, 10)
        else:
            impact = 10 + (rate - self.thresholds.context_switch_severe) / 10000 * 10
            return NoiseClassification.SEVERE, min(impact, 30)

    def classify_irq_rate(self, rate: float) -> Tuple[NoiseClassification, float]:
        """Classify IRQ rate."""
        if rate < self.thresholds.irq_warning:
            return NoiseClassification.NEGLIGIBLE, 0.0
        elif rate < self.thresholds.irq_severe:
            impact = (
                (rate - self.thresholds.irq_warning)
                / (self.thresholds.irq_severe - self.thresholds.irq_warning)
                * 5
            )
            return NoiseClassification.MODERATE, min(impact, 5)
        else:
            impact = 5 + (rate - self.thresholds.irq_severe) / 10000 * 10
            return NoiseClassification.SEVERE, min(impact, 20)

    def classify_memory_pressure(self, psi_avg10: float) -> Tuple[NoiseClassification, float]:
        """Classify memory pressure from PSI."""
        if psi_avg10 < self.thresholds.memory_pressure_warning:
            return NoiseClassification.NEGLIGIBLE, 0.0
        elif psi_avg10 < self.thresholds.memory_pressure_severe:
            return NoiseClassification.MODERATE, psi_avg10 * 0.5
        else:
            return NoiseClassification.SEVERE, min(psi_avg10, 50)

    def classify_io_pressure(self, psi_avg10: float) -> Tuple[NoiseClassification, float]:
        """Classify IO pressure from PSI."""
        if psi_avg10 < self.thresholds.io_pressure_warning:
            return NoiseClassification.NEGLIGIBLE, 0.0
        elif psi_avg10 < self.thresholds.io_pressure_severe:
            return NoiseClassification.MINOR, psi_avg10 * 0.2
        else:
            return NoiseClassification.MODERATE, min(psi_avg10 * 0.3, 20)

    def classify_page_faults(self, rate: float) -> Tuple[NoiseClassification, float]:
        """Classify major page fault rate."""
        if rate < self.thresholds.page_fault_warning:
            return NoiseClassification.NEGLIGIBLE, 0.0
        elif rate < self.thresholds.page_fault_severe:
            impact = rate / self.thresholds.page_fault_severe * 10
            return NoiseClassification.MODERATE, min(impact, 10)
        else:
            return NoiseClassification.SEVERE, min(rate / 100, 30)


class NoiseSentinelV2:
    """
    AACO-SIGMA Noise Sentinel V2

    Advanced interference detection with:
    - Multi-source monitoring (IRQ, PSI, vmstat, scheduler)
    - Impact estimation
    - Classification
    - Detailed reporting
    """

    def __init__(self, thresholds: Optional[NoiseThresholdsV2] = None):
        self.thresholds = thresholds or NoiseThresholdsV2()
        self.sampler = InterferenceSampler()
        self.classifier = InterferenceClassifier(thresholds)

        self._events: List[NoiseEventV2] = []
        self._samples: deque = deque(maxlen=10000)  # Keep recent samples

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        self._session_id: str = ""
        self._start_time_ns: int = 0
        self._baseline_established: bool = False
        self._baseline: Dict[str, float] = {}

    def start(self, session_id: str = "") -> None:
        """Start noise monitoring."""
        self._session_id = session_id
        self._start_time_ns = time.time_ns()
        self._events.clear()
        self._samples.clear()
        self._baseline_established = False

        # Take initial baseline samples
        self._establish_baseline()

        # Start monitoring thread
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> NoiseReportV2:
        """Stop monitoring and generate report."""
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        return self._generate_report()

    def _establish_baseline(self, num_samples: int = 5, interval: float = 0.1) -> None:
        """Establish baseline interference levels."""
        samples = []
        for _ in range(num_samples):
            delta = self.sampler.sample_delta()
            if delta:
                samples.append(delta)
            time.sleep(interval)

        if samples:
            # Calculate baseline as mean of samples
            self._baseline = {
                "context_switches_per_sec": sum(
                    s.get("context_switches_per_sec", 0) for s in samples
                )
                / len(samples),
                "irq_total_per_sec": sum(s.get("irq_total_per_sec", 0) for s in samples)
                / len(samples),
                "softirq_total_per_sec": sum(s.get("softirq_total_per_sec", 0) for s in samples)
                / len(samples),
                "pgmajfault_per_sec": sum(s.get("pgmajfault_per_sec", 0) for s in samples)
                / len(samples),
            }
            self._baseline_established = True

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        interval = 0.1  # 100ms

        while not self._stop_monitoring.is_set():
            sample = self.sampler.sample_delta()
            if sample:
                self._samples.append(sample)
                self._analyze_sample(sample)

            self._stop_monitoring.wait(interval)

    def _analyze_sample(self, sample: Dict[str, Any]) -> None:
        """Analyze sample for interference."""
        timestamp = sample["timestamp_ns"]

        # Check context switches
        cs_rate = sample.get("context_switches_per_sec", 0)
        if cs_rate > 0:
            classification, impact = self.classifier.classify_context_switches(cs_rate)
            if classification != NoiseClassification.NEGLIGIBLE:
                self._events.append(
                    NoiseEventV2(
                        timestamp_ns=timestamp,
                        source=NoiseSourceV2.CONTEXT_SWITCH,
                        classification=classification,
                        measured_value=cs_rate,
                        threshold=self.thresholds.context_switch_warning,
                        baseline_value=self._baseline.get("context_switches_per_sec", 0),
                        estimated_impact_pct=impact,
                        details={"rate": cs_rate},
                    )
                )

        # Check IRQ rate
        irq_rate = sample.get("irq_total_per_sec", 0)
        if irq_rate > 0:
            classification, impact = self.classifier.classify_irq_rate(irq_rate)
            if classification != NoiseClassification.NEGLIGIBLE:
                self._events.append(
                    NoiseEventV2(
                        timestamp_ns=timestamp,
                        source=NoiseSourceV2.IRQ_STORM,
                        classification=classification,
                        measured_value=irq_rate,
                        threshold=self.thresholds.irq_warning,
                        baseline_value=self._baseline.get("irq_total_per_sec", 0),
                        estimated_impact_pct=impact,
                        details={"rate": irq_rate},
                    )
                )

        # Check memory pressure
        mem_psi = sample.get("memory_psi_avg10", 0)
        if mem_psi > 0:
            classification, impact = self.classifier.classify_memory_pressure(mem_psi)
            if classification != NoiseClassification.NEGLIGIBLE:
                self._events.append(
                    NoiseEventV2(
                        timestamp_ns=timestamp,
                        source=NoiseSourceV2.MEMORY_PRESSURE,
                        classification=classification,
                        measured_value=mem_psi,
                        threshold=self.thresholds.memory_pressure_warning,
                        estimated_impact_pct=impact,
                        details={"psi_avg10": mem_psi},
                    )
                )

        # Check IO pressure
        io_psi = sample.get("io_psi_avg10", 0)
        if io_psi > 0:
            classification, impact = self.classifier.classify_io_pressure(io_psi)
            if classification != NoiseClassification.NEGLIGIBLE:
                self._events.append(
                    NoiseEventV2(
                        timestamp_ns=timestamp,
                        source=NoiseSourceV2.IO_PRESSURE,
                        classification=classification,
                        measured_value=io_psi,
                        threshold=self.thresholds.io_pressure_warning,
                        estimated_impact_pct=impact,
                        details={"psi_avg10": io_psi},
                    )
                )

        # Check major page faults
        pgmajfault_rate = sample.get("pgmajfault_per_sec", 0)
        if pgmajfault_rate > 0:
            classification, impact = self.classifier.classify_page_faults(pgmajfault_rate)
            if classification != NoiseClassification.NEGLIGIBLE:
                self._events.append(
                    NoiseEventV2(
                        timestamp_ns=timestamp,
                        source=NoiseSourceV2.PAGE_FAULT,
                        classification=classification,
                        measured_value=pgmajfault_rate,
                        threshold=self.thresholds.page_fault_warning,
                        baseline_value=self._baseline.get("pgmajfault_per_sec", 0),
                        estimated_impact_pct=impact,
                        details={"rate": pgmajfault_rate},
                    )
                )

    def _generate_report(self) -> NoiseReportV2:
        """Generate comprehensive noise report."""
        end_time_ns = time.time_ns()
        duration_ns = end_time_ns - self._start_time_ns

        report = NoiseReportV2(
            session_id=self._session_id,
            duration_ns=duration_ns,
        )

        # Count events by source
        for source in NoiseSourceV2:
            count = sum(1 for e in self._events if e.source == source)
            if count > 0:
                report.event_counts[source.name] = count

        # Get most significant events
        significant = [
            e
            for e in self._events
            if e.classification
            in [
                NoiseClassification.MODERATE,
                NoiseClassification.SIGNIFICANT,
                NoiseClassification.SEVERE,
            ]
        ]
        significant.sort(key=lambda e: -e.estimated_impact_pct)
        report.significant_events = [e.to_dict() for e in significant[:20]]

        # Estimate total impact (non-additive, use max overlapping)
        if self._events:
            report.estimated_total_impact_pct = max(e.estimated_impact_pct for e in self._events)

        # Calculate noisy time percentage
        if self._samples:
            noisy_samples = sum(
                1
                for s in self._samples
                if s.get("context_switches_per_sec", 0) > self.thresholds.context_switch_warning
                or s.get("memory_psi_avg10", 0) > self.thresholds.memory_pressure_warning
            )
            report.noisy_time_pct = noisy_samples / len(self._samples) * 100

        # Set baseline statistics
        report.baseline_context_switches_per_sec = self._baseline.get("context_switches_per_sec", 0)
        report.baseline_irqs_per_sec = self._baseline.get("irq_total_per_sec", 0)

        # Calculate peaks
        if self._samples:
            report.peak_context_switches_per_sec = max(
                s.get("context_switches_per_sec", 0) for s in self._samples
            )
            report.peak_irqs_per_sec = max(s.get("irq_total_per_sec", 0) for s in self._samples)
            report.peak_memory_pressure = max(s.get("memory_psi_avg10", 0) for s in self._samples)

        # Calculate noise score (0 = clean, 100 = unusable)
        score = 0.0
        severe_count = sum(
            1 for e in self._events if e.classification == NoiseClassification.SEVERE
        )
        moderate_count = sum(
            1 for e in self._events if e.classification == NoiseClassification.MODERATE
        )

        score += severe_count * 5
        score += moderate_count * 1
        score += report.noisy_time_pct * 0.5
        score += report.estimated_total_impact_pct

        report.noise_score = min(100, score)
        report.is_clean = report.noise_score < 20

        # Generate recommendations
        if report.peak_context_switches_per_sec > self.thresholds.context_switch_severe:
            report.recommendations.append("Consider isolating CPU cores or reducing system load")

        if report.peak_memory_pressure > self.thresholds.memory_pressure_severe:
            report.recommendations.append(
                "Increase available memory or reduce memory pressure from other processes"
            )

        if NoiseSourceV2.IRQ_STORM.name in report.event_counts:
            report.recommendations.append("Check for IRQ storms from network or storage devices")

        if NoiseSourceV2.PAGE_FAULT.name in report.event_counts:
            report.recommendations.append(
                "Ensure working set fits in memory; consider mlocking critical pages"
            )

        if not report.is_clean:
            report.recommendations.append(
                "Consider re-running measurement during lower system activity"
            )

        return report

    def get_events(self) -> List[NoiseEventV2]:
        """Get all detected noise events."""
        return self._events.copy()

    def get_current_noise_level(self) -> float:
        """Get current estimated noise level (0-100)."""
        if not self._samples:
            return 0.0

        # Use most recent samples
        recent = list(self._samples)[-10:]

        score = 0.0
        for sample in recent:
            cs_rate = sample.get("context_switches_per_sec", 0)
            if cs_rate > self.thresholds.context_switch_warning:
                score += 1
            if cs_rate > self.thresholds.context_switch_severe:
                score += 2

            mem_psi = sample.get("memory_psi_avg10", 0)
            if mem_psi > self.thresholds.memory_pressure_warning:
                score += mem_psi / 10

        return min(100, score * 10 / len(recent))
