"""
AACO-SIGMA Hardware Performance Counter Integration

Integration with AMD GPU performance counters via rocprofiler.
Provides:
- Counter specification and collection
- Counter-based kernel characterization
- Roofline model metrics
"""

import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum


class CounterDomain(Enum):
    """Hardware counter domains."""

    GPU_COMPUTE = "compute"
    GPU_MEMORY = "memory"
    GPU_SHADER = "shader"
    GPU_TEXTURE = "texture"
    GPU_L1 = "l1"
    GPU_L2 = "l2"
    GPU_VRAM = "vram"


@dataclass
class HardwareCounter:
    """
    Definition of a hardware performance counter.
    """

    name: str
    domain: CounterDomain
    description: str = ""
    unit: str = ""

    # Derived counter info
    is_derived: bool = False
    formula: str = ""  # For derived counters

    # Collection properties
    requires_replay: bool = False  # Counter requires kernel replay
    has_overhead: bool = False

    # AMD-specific
    amd_counter_name: str = ""  # rocprof counter name


@dataclass
class CounterSpec:
    """
    Specification of counters to collect.
    """

    counters: List[HardwareCounter] = field(default_factory=list)

    # Collection settings
    collection_mode: str = "per_kernel"  # per_kernel, per_dispatch, aggregate

    # Filtering
    kernel_filter: str = ""  # Regex filter for kernel names
    min_duration_ns: int = 0

    def to_rocprof_inputs(self) -> List[str]:
        """Generate rocprof input file contents."""
        lines = ["metrics:"]
        for counter in self.counters:
            name = counter.amd_counter_name or counter.name
            lines.append(f"  - {name}")
        return lines


@dataclass
class CounterReading:
    """
    A single counter reading for a kernel.
    """

    kernel_name: str
    dispatch_id: int

    counters: Dict[str, float] = field(default_factory=dict)

    # Timing
    duration_ns: int = 0

    # Derived metrics
    compute_throughput_pct: float = 0.0
    memory_throughput_pct: float = 0.0
    occupancy_pct: float = 0.0
    l2_hit_rate: float = 0.0


# =============================================================================
# Standard AMD GPU Counters
# =============================================================================


class AMDCounters:
    """Registry of AMD GPU hardware counters."""

    # Compute utilization
    GRBM_GUI_ACTIVE = HardwareCounter(
        name="GRBM_GUI_ACTIVE",
        domain=CounterDomain.GPU_COMPUTE,
        description="GPU graphics/compute active cycles",
        amd_counter_name="GRBM_GUI_ACTIVE",
    )

    # Shader activity
    SQ_WAVES = HardwareCounter(
        name="SQ_WAVES",
        domain=CounterDomain.GPU_SHADER,
        description="Number of wavefronts dispatched",
        amd_counter_name="SQ_WAVES",
    )

    SQ_INSTS_VALU = HardwareCounter(
        name="SQ_INSTS_VALU",
        domain=CounterDomain.GPU_SHADER,
        description="Vector ALU instructions",
        amd_counter_name="SQ_INSTS_VALU",
    )

    SQ_INSTS_SALU = HardwareCounter(
        name="SQ_INSTS_SALU",
        domain=CounterDomain.GPU_SHADER,
        description="Scalar ALU instructions",
        amd_counter_name="SQ_INSTS_SALU",
    )

    SQ_INSTS_LDS = HardwareCounter(
        name="SQ_INSTS_LDS",
        domain=CounterDomain.GPU_SHADER,
        description="LDS (shared memory) instructions",
        amd_counter_name="SQ_INSTS_LDS",
    )

    SQ_INSTS_GDS = HardwareCounter(
        name="SQ_INSTS_GDS",
        domain=CounterDomain.GPU_SHADER,
        description="Global data share instructions",
        amd_counter_name="SQ_INSTS_GDS",
    )

    SQ_INSTS_VMEM = HardwareCounter(
        name="SQ_INSTS_VMEM",
        domain=CounterDomain.GPU_SHADER,
        description="Vector memory instructions",
        amd_counter_name="SQ_INSTS_V[MEM_RD]",
    )

    SQ_WAIT_CNT = HardwareCounter(
        name="SQ_WAIT_CNT",
        domain=CounterDomain.GPU_SHADER,
        description="Wait count stalls",
        amd_counter_name="SQ_WAIT_CNT",
    )

    # Memory
    TCP_TCC_READ_REQ_sum = HardwareCounter(
        name="TCP_TCC_READ_REQ",
        domain=CounterDomain.GPU_L2,
        description="L2 cache read requests",
        amd_counter_name="TCP_TCC_READ_REQ_sum",
    )

    TCP_TCC_WRITE_REQ_sum = HardwareCounter(
        name="TCP_TCC_WRITE_REQ",
        domain=CounterDomain.GPU_L2,
        description="L2 cache write requests",
        amd_counter_name="TCP_TCC_WRITE_REQ_sum",
    )

    TCC_HIT_sum = HardwareCounter(
        name="TCC_HIT",
        domain=CounterDomain.GPU_L2,
        description="L2 cache hits",
        amd_counter_name="TCC_HIT_sum",
    )

    TCC_MISS_sum = HardwareCounter(
        name="TCC_MISS",
        domain=CounterDomain.GPU_L2,
        description="L2 cache misses",
        amd_counter_name="TCC_MISS_sum",
    )

    # VRAM
    TCC_EA_RDREQ_sum = HardwareCounter(
        name="TCC_EA_RDREQ",
        domain=CounterDomain.GPU_VRAM,
        description="VRAM read requests",
        amd_counter_name="TCC_EA_RDREQ_sum",
    )

    TCC_EA_WRREQ_sum = HardwareCounter(
        name="TCC_EA_WRREQ",
        domain=CounterDomain.GPU_VRAM,
        description="VRAM write requests",
        amd_counter_name="TCC_EA_WRREQ_sum",
    )

    @classmethod
    def get_compute_counters(cls) -> List[HardwareCounter]:
        """Get compute-focused counters."""
        return [
            cls.GRBM_GUI_ACTIVE,
            cls.SQ_WAVES,
            cls.SQ_INSTS_VALU,
            cls.SQ_INSTS_SALU,
        ]

    @classmethod
    def get_memory_counters(cls) -> List[HardwareCounter]:
        """Get memory-focused counters."""
        return [
            cls.TCP_TCC_READ_REQ_sum,
            cls.TCP_TCC_WRITE_REQ_sum,
            cls.TCC_HIT_sum,
            cls.TCC_MISS_sum,
            cls.TCC_EA_RDREQ_sum,
            cls.TCC_EA_WRREQ_sum,
        ]

    @classmethod
    def get_roofline_counters(cls) -> List[HardwareCounter]:
        """Get counters needed for roofline analysis."""
        return [
            cls.SQ_INSTS_VALU,
            cls.TCC_EA_RDREQ_sum,
            cls.TCC_EA_WRREQ_sum,
        ]


class CounterSession:
    """
    A counter collection session.

    Manages counter specification, collection, and results.
    """

    def __init__(self, spec: Optional[CounterSpec] = None):
        self.spec = spec or CounterSpec()
        self._readings: List[CounterReading] = []
        self._raw_data: Dict[str, Any] = {}

    def add_counter(self, counter: HardwareCounter) -> None:
        """Add counter to collection spec."""
        self.spec.counters.append(counter)

    def add_reading(self, reading: CounterReading) -> None:
        """Add a counter reading."""
        self._readings.append(reading)

    def get_readings(self, kernel_name: Optional[str] = None) -> List[CounterReading]:
        """Get counter readings, optionally filtered by kernel."""
        if kernel_name:
            return [r for r in self._readings if r.kernel_name == kernel_name]
        return self._readings

    def compute_derived_metrics(self) -> None:
        """Compute derived metrics from raw counters."""
        for reading in self._readings:
            counters = reading.counters

            # L2 hit rate
            hits = counters.get("TCC_HIT", 0)
            misses = counters.get("TCC_MISS", 0)
            total = hits + misses
            if total > 0:
                reading.l2_hit_rate = hits / total * 100

            # Compute intensity (simplified)
            valu = counters.get("SQ_INSTS_VALU", 0)
            vmem = counters.get("TCC_EA_RDREQ", 0) + counters.get("TCC_EA_WRREQ", 0)
            if vmem > 0:
                reading.compute_throughput_pct = min(100, valu / vmem * 10)

    def summarize(self) -> Dict[str, Any]:
        """Generate summary of counter session."""
        if not self._readings:
            return {}

        # Aggregate by kernel
        by_kernel: Dict[str, List[CounterReading]] = {}
        for r in self._readings:
            if r.kernel_name not in by_kernel:
                by_kernel[r.kernel_name] = []
            by_kernel[r.kernel_name].append(r)

        summaries = []
        for kernel_name, readings in by_kernel.items():
            # Average counters
            avg_counters = {}
            counter_names = readings[0].counters.keys() if readings else []
            for counter in counter_names:
                values = [r.counters.get(counter, 0) for r in readings]
                avg_counters[counter] = sum(values) / len(values)

            summaries.append(
                {
                    "kernel": kernel_name,
                    "dispatch_count": len(readings),
                    "avg_duration_ns": sum(r.duration_ns for r in readings) / len(readings),
                    "avg_counters": avg_counters,
                    "avg_l2_hit_rate": sum(r.l2_hit_rate for r in readings) / len(readings),
                }
            )

        return {
            "total_readings": len(self._readings),
            "unique_kernels": len(by_kernel),
            "kernels": summaries,
        }


class CounterReader:
    """
    Abstract base for counter readers.
    """

    def is_available(self) -> bool:
        """Check if counter collection is available."""
        raise NotImplementedError

    def collect(self, command: List[str], spec: CounterSpec) -> CounterSession:
        """Collect counters for a command."""
        raise NotImplementedError


class RocprofCounterReader(CounterReader):
    """
    Counter reader using rocprof.
    """

    def __init__(self, rocprof_path: str = "rocprof"):
        self.rocprof_path = rocprof_path
        self._temp_dir = Path("/tmp/aaco_counters")

    def is_available(self) -> bool:
        """Check if rocprof is available."""
        try:
            result = subprocess.run(
                [self.rocprof_path, "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def collect(self, command: List[str], spec: CounterSpec) -> CounterSession:
        """Collect counters for a command execution."""
        session = CounterSession(spec)

        if not self.is_available():
            return session

        # Create temp directory
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate input file
        input_file = self._temp_dir / "input.txt"
        with open(input_file, "w") as f:
            for line in spec.to_rocprof_inputs():
                f.write(line + "\n")

        # Output file
        output_file = self._temp_dir / "results.csv"

        # Run rocprof
        rocprof_cmd = [
            self.rocprof_path,
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "--timestamp",
            "on",
            "--stats",
        ] + command

        try:
            result = subprocess.run(
                rocprof_cmd,
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0 and output_file.exists():
                session = self._parse_results(output_file, spec)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        return session

    def _parse_results(self, output_file: Path, spec: CounterSpec) -> CounterSession:
        """Parse rocprof output."""
        session = CounterSession(spec)

        with open(output_file) as f:
            lines = f.readlines()

        # Parse header
        if not lines:
            return session

        header = lines[0].strip().split(",")

        # Find column indices
        kernel_idx = header.index("KernelName") if "KernelName" in header else -1
        dispatch_idx = header.index("dispatch") if "dispatch" in header else -1
        duration_idx = header.index("DurationNs") if "DurationNs" in header else -1

        counter_indices = {}
        for counter in spec.counters:
            name = counter.amd_counter_name or counter.name
            if name in header:
                counter_indices[counter.name] = header.index(name)

        # Parse data rows
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) <= max(kernel_idx, dispatch_idx, duration_idx):
                continue

            reading = CounterReading(
                kernel_name=parts[kernel_idx] if kernel_idx >= 0 else "",
                dispatch_id=int(parts[dispatch_idx]) if dispatch_idx >= 0 else 0,
                duration_ns=int(parts[duration_idx]) if duration_idx >= 0 else 0,
            )

            for counter_name, idx in counter_indices.items():
                try:
                    reading.counters[counter_name] = float(parts[idx])
                except (ValueError, IndexError):
                    pass

            session.add_reading(reading)

        session.compute_derived_metrics()
        return session

    def get_available_counters(self) -> List[str]:
        """Get list of available counters from rocprof."""
        if not self.is_available():
            return []

        try:
            result = subprocess.run(
                [self.rocprof_path, "--list-basic"], capture_output=True, timeout=10
            )

            if result.returncode == 0:
                return [
                    line.strip()
                    for line in result.stdout.decode().split("\n")
                    if line.strip() and not line.startswith("#")
                ]
        except:
            pass

        return []
