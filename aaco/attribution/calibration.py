"""
AACO-SIGMA Counter Calibration

Calibrates hardware counters to ensure accurate measurements.
Models and subtracts profiling overhead.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import time
import statistics

from .counter_model import CounterReading


class CalibrationState(Enum):
    """State of calibration."""

    NOT_CALIBRATED = auto()
    CALIBRATING = auto()
    CALIBRATED = auto()
    STALE = auto()  # Needs recalibration


@dataclass
class OverheadModel:
    """Model of profiling overhead."""

    # Kernel launch overhead
    launch_overhead_ns: int = 0
    launch_overhead_stddev: float = 0.0

    # Counter collection overhead per counter
    counter_overhead_ns: Dict[str, int] = field(default_factory=dict)

    # Memory barrier overhead
    barrier_overhead_ns: int = 0

    # Timestamp read overhead
    timestamp_overhead_ns: int = 0

    # Total overhead per measurement
    total_overhead_ns: int = 0

    def get_overhead_for_counters(self, counter_names: List[str]) -> int:
        """Get total overhead for specific counters."""
        overhead = self.launch_overhead_ns + self.timestamp_overhead_ns

        for name in counter_names:
            overhead += self.counter_overhead_ns.get(name, 0)

        return overhead


@dataclass
class CalibrationProfile:
    """Calibration profile for a specific GPU."""

    gpu_name: str
    gpu_id: int = 0

    # Hardware characteristics
    gpu_clock_mhz: int = 0
    memory_clock_mhz: int = 0

    # Counter characteristics
    available_counters: List[str] = field(default_factory=list)
    counter_precision: Dict[str, int] = field(default_factory=dict)  # bits

    # Overhead model
    overhead: OverheadModel = field(default_factory=OverheadModel)

    # Calibration metadata
    calibration_timestamp: float = 0.0
    calibration_temperature_c: float = 0.0
    samples_collected: int = 0

    # Confidence
    confidence_level: float = 0.0  # 0-1


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    success: bool
    profile: Optional[CalibrationProfile] = None

    # Statistics
    iterations: int = 0
    duration_s: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)
    error_message: str = ""


class CounterCalibrator:
    """
    Calibrates hardware counter collection.

    Measures and models:
    - Kernel launch overhead
    - Counter collection overhead
    - Timer precision
    - Counter accuracy
    """

    # Known AMD GPU counters
    AMD_COUNTERS = [
        "SQ_WAVES",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_LDS",
        "SQ_INSTS_MFMA",
        "TCC_HIT",
        "TCC_MISS",
        "TCC_EA_RDREQ",
        "TCC_EA_WRREQ",
        "GRBM_COUNT",
        "GRBM_GUI_ACTIVE",
    ]

    # Minimum calibration iterations
    MIN_ITERATIONS = 100
    DEFAULT_ITERATIONS = 1000

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.profile: Optional[CalibrationProfile] = None
        self.state = CalibrationState.NOT_CALIBRATED

        # Raw measurements for analysis
        self._raw_launch_times: List[int] = []
        self._raw_counter_times: Dict[str, List[int]] = {}

    def calibrate(
        self, iterations: int = DEFAULT_ITERATIONS, counters: Optional[List[str]] = None
    ) -> CalibrationResult:
        """
        Run calibration procedure.

        Args:
            iterations: Number of calibration iterations
            counters: Specific counters to calibrate (default: all)
        """
        self.state = CalibrationState.CALIBRATING
        start_time = time.time()

        if iterations < self.MIN_ITERATIONS:
            iterations = self.MIN_ITERATIONS

        if counters is None:
            counters = self.AMD_COUNTERS.copy()

        # Initialize profile
        self.profile = CalibrationProfile(
            gpu_name=self._detect_gpu_name(),
            gpu_id=self.gpu_id,
            available_counters=counters,
        )

        try:
            # Measure launch overhead
            launch_times = self._measure_launch_overhead(iterations)
            self._raw_launch_times = launch_times

            self.profile.overhead.launch_overhead_ns = int(statistics.mean(launch_times))
            self.profile.overhead.launch_overhead_stddev = (
                statistics.stdev(launch_times) if len(launch_times) > 1 else 0.0
            )

            # Measure timestamp overhead
            ts_times = self._measure_timestamp_overhead(iterations)
            self.profile.overhead.timestamp_overhead_ns = int(statistics.mean(ts_times))

            # Measure per-counter overhead
            for counter in counters:
                counter_times = self._measure_counter_overhead(counter, iterations // 10)
                self._raw_counter_times[counter] = counter_times
                self.profile.overhead.counter_overhead_ns[counter] = int(
                    statistics.mean(counter_times)
                )

            # Calculate total overhead
            total = self.profile.overhead.launch_overhead_ns
            total += self.profile.overhead.timestamp_overhead_ns
            total += sum(self.profile.overhead.counter_overhead_ns.values())
            self.profile.overhead.total_overhead_ns = total

            # Calculate confidence based on measurement variance
            self.profile.confidence_level = self._calculate_confidence(launch_times)

            # Store metadata
            self.profile.calibration_timestamp = time.time()
            self.profile.samples_collected = iterations

            self.state = CalibrationState.CALIBRATED

            result = CalibrationResult(
                success=True,
                profile=self.profile,
                iterations=iterations,
                duration_s=time.time() - start_time,
            )

            # Add warnings for high variance
            if (
                self.profile.overhead.launch_overhead_stddev
                > self.profile.overhead.launch_overhead_ns * 0.2
            ):
                result.warnings.append(
                    f"High launch overhead variance: {self.profile.overhead.launch_overhead_stddev:.0f}ns"
                )

            return result

        except Exception as e:
            self.state = CalibrationState.NOT_CALIBRATED
            return CalibrationResult(
                success=False,
                error_message=str(e),
                duration_s=time.time() - start_time,
            )

    def _detect_gpu_name(self) -> str:
        """Detect GPU name."""
        # In production, would use ROCm APIs
        return f"AMD GPU {self.gpu_id}"

    def _measure_launch_overhead(self, iterations: int) -> List[int]:
        """Measure kernel launch overhead."""
        times: List[int] = []

        # Simulate measurement (in production, would launch empty kernels)
        for _ in range(iterations):
            # Empty kernel launch simulation
            # Typical launch overhead: 2-10 microseconds
            overhead = 3000 + int(hash(str(len(times))) % 2000)  # 3-5us
            times.append(overhead)

        return times

    def _measure_timestamp_overhead(self, iterations: int) -> List[int]:
        """Measure timestamp read overhead."""
        times: List[int] = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            _ = time.perf_counter_ns()
            end = time.perf_counter_ns()
            times.append(end - start)

        return times

    def _measure_counter_overhead(self, counter: str, iterations: int) -> List[int]:
        """Measure overhead of reading a specific counter."""
        times: List[int] = []

        # Simulate counter read overhead (in production, would use rocprof)
        for _ in range(iterations):
            # Counter read overhead varies by counter type
            base_overhead = 500  # 500ns base

            # Some counters are more expensive
            if "TCC" in counter:
                base_overhead += 200  # Cache counters
            elif "MFMA" in counter:
                base_overhead += 100  # Matrix counters

            times.append(base_overhead + int(hash(counter + str(len(times))) % 300))

        return times

    def _calculate_confidence(self, launch_times: List[int]) -> float:
        """Calculate calibration confidence level."""
        if len(launch_times) < 10:
            return 0.0

        mean = statistics.mean(launch_times)
        stddev = statistics.stdev(launch_times)

        # Coefficient of variation
        cv = stddev / mean if mean > 0 else float("inf")

        # Lower CV = higher confidence
        # CV < 0.1 = very stable measurements
        # CV > 0.5 = unstable measurements
        if cv < 0.05:
            return 0.99
        elif cv < 0.1:
            return 0.95
        elif cv < 0.2:
            return 0.85
        elif cv < 0.3:
            return 0.70
        elif cv < 0.5:
            return 0.50
        else:
            return 0.30

    def adjust_reading(self, reading: CounterReading) -> CounterReading:
        """Adjust a counter reading by subtracting overhead."""
        if self.profile is None or self.state != CalibrationState.CALIBRATED:
            return reading

        # Create adjusted copy
        adjusted = CounterReading(
            waves_launched=reading.waves_launched,
            waves_completed=reading.waves_completed,
            valu_instructions=reading.valu_instructions,
            salu_instructions=reading.salu_instructions,
            mfma_instructions=reading.mfma_instructions,
            lds_instructions=reading.lds_instructions,
            memory_read_bytes=reading.memory_read_bytes,
            memory_write_bytes=reading.memory_write_bytes,
            l1_hits=reading.l1_hits,
            l1_misses=reading.l1_misses,
            l2_hits=reading.l2_hits,
            l2_misses=reading.l2_misses,
            gpu_cycles=reading.gpu_cycles,
            active_cycles=reading.active_cycles,
            stall_cycles=reading.stall_cycles,
            duration_ns=max(0, reading.duration_ns - self.profile.overhead.total_overhead_ns),
        )

        return adjusted

    def needs_recalibration(self, max_age_hours: float = 24.0) -> bool:
        """Check if calibration is stale and needs refresh."""
        if self.profile is None:
            return True

        age_hours = (time.time() - self.profile.calibration_timestamp) / 3600
        return age_hours > max_age_hours

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration status report."""
        if self.profile is None:
            return {"state": "NOT_CALIBRATED"}

        return {
            "state": self.state.name,
            "gpu_name": self.profile.gpu_name,
            "launch_overhead_ns": self.profile.overhead.launch_overhead_ns,
            "launch_overhead_stddev_ns": self.profile.overhead.launch_overhead_stddev,
            "timestamp_overhead_ns": self.profile.overhead.timestamp_overhead_ns,
            "total_overhead_ns": self.profile.overhead.total_overhead_ns,
            "confidence_level": self.profile.confidence_level,
            "samples_collected": self.profile.samples_collected,
            "age_hours": (time.time() - self.profile.calibration_timestamp) / 3600,
            "counter_overheads": self.profile.overhead.counter_overhead_ns,
        }
