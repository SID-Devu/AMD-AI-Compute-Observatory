"""
AACO-SIGMA GPU Model

Comprehensive hardware model for AMD GPUs.
Contains specifications for CDNA and RDNA architectures.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class GPUArchitecture(Enum):
    """AMD GPU architecture families."""

    CDNA1 = "cdna1"  # MI100
    CDNA2 = "cdna2"  # MI200 series
    CDNA3 = "cdna3"  # MI300 series
    RDNA2 = "rdna2"  # RX 6000 series
    RDNA3 = "rdna3"  # RX 7000 series
    RDNA35 = "rdna3.5"  # RX 8000 series (upcoming)


class MemoryType(Enum):
    """GPU memory types."""

    HBM2 = "hbm2"
    HBM2E = "hbm2e"
    HBM3 = "hbm3"
    GDDR6 = "gddr6"
    GDDR6X = "gddr6x"


@dataclass
class ComputeCapability:
    """Compute capability details."""

    # FP operations per cycle per CU
    fp32_ops_per_cycle: int = 64
    fp16_ops_per_cycle: int = 128
    bf16_ops_per_cycle: int = 128
    fp64_ops_per_cycle: int = 32
    int8_ops_per_cycle: int = 256

    # Matrix operations
    mfma_available: bool = False
    mfma_shapes: List[str] = field(default_factory=list)  # e.g., ["32x32x8", "16x16x16"]
    wmma_available: bool = False

    # Special features
    dot_product_instructions: bool = True
    packed_math: bool = True


@dataclass
class MemorySpecs:
    """Memory subsystem specifications."""

    # Capacity
    vram_gb: float = 0.0

    # Bandwidth
    memory_bandwidth_gbps: float = 0.0

    # Cache hierarchy
    l1_cache_kb_per_cu: int = 16
    l2_cache_mb: float = 0.0
    infinity_cache_mb: float = 0.0

    # Memory type
    memory_type: MemoryType = MemoryType.GDDR6
    memory_bus_width: int = 256
    memory_clock_mhz: int = 2000


@dataclass
class GPUSpecs:
    """Complete GPU specifications."""

    # Identity
    name: str = ""
    codename: str = ""
    gfx_version: str = ""
    architecture: GPUArchitecture = GPUArchitecture.RDNA3

    # Compute units
    compute_units: int = 0
    stream_processors: int = 0  # CU * 64 typically

    # Clocks
    base_clock_mhz: int = 0
    boost_clock_mhz: int = 0
    memory_clock_mhz: int = 0

    # Waves
    max_waves_per_cu: int = 32
    wavefront_size: int = 32  # 32 for RDNA, 64 for CDNA

    # Memory
    memory: MemorySpecs = field(default_factory=MemorySpecs)

    # Compute
    compute: ComputeCapability = field(default_factory=ComputeCapability)

    # Power
    tdp_watts: int = 0

    # Peak theoretical
    @property
    def peak_fp32_tflops(self) -> float:
        """Peak FP32 TFLOPS."""
        ops_per_cycle = self.stream_processors * 2  # FMA = 2 ops
        return ops_per_cycle * self.boost_clock_mhz / 1e6

    @property
    def peak_fp16_tflops(self) -> float:
        """Peak FP16 TFLOPS (with packed math)."""
        return self.peak_fp32_tflops * 2

    @property
    def peak_int8_tops(self) -> float:
        """Peak INT8 TOPS."""
        return self.peak_fp32_tflops * 4


# ============ AMD GPU Database ============

AMD_GPU_DATABASE: Dict[str, GPUSpecs] = {}


def _register_gpu(specs: GPUSpecs) -> None:
    """Register GPU in database."""
    AMD_GPU_DATABASE[specs.gfx_version] = specs
    AMD_GPU_DATABASE[specs.name.lower().replace(" ", "_")] = specs


# --- CDNA GPUs (Data Center) ---

_register_gpu(
    GPUSpecs(
        name="AMD Instinct MI100",
        codename="Arcturus",
        gfx_version="gfx908",
        architecture=GPUArchitecture.CDNA1,
        compute_units=120,
        stream_processors=7680,
        base_clock_mhz=1502,
        boost_clock_mhz=1502,
        max_waves_per_cu=32,
        wavefront_size=64,
        memory=MemorySpecs(
            vram_gb=32.0,
            memory_bandwidth_gbps=1228.8,
            l2_cache_mb=8.0,
            memory_type=MemoryType.HBM2,
            memory_bus_width=4096,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=64,
            fp16_ops_per_cycle=256,
            bf16_ops_per_cycle=256,
            fp64_ops_per_cycle=64,
            mfma_available=True,
            mfma_shapes=["32x32x8", "16x16x16", "4x4x4"],
        ),
        tdp_watts=300,
    )
)

_register_gpu(
    GPUSpecs(
        name="AMD Instinct MI210",
        codename="Aldebaran",
        gfx_version="gfx90a",
        architecture=GPUArchitecture.CDNA2,
        compute_units=104,
        stream_processors=6656,
        base_clock_mhz=1700,
        boost_clock_mhz=1700,
        max_waves_per_cu=32,
        wavefront_size=64,
        memory=MemorySpecs(
            vram_gb=64.0,
            memory_bandwidth_gbps=1638.4,
            l2_cache_mb=8.0,
            memory_type=MemoryType.HBM2E,
            memory_bus_width=4096,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=64,
            fp16_ops_per_cycle=256,
            bf16_ops_per_cycle=256,
            fp64_ops_per_cycle=64,
            mfma_available=True,
            mfma_shapes=["32x32x8", "16x16x16", "4x4x4"],
        ),
        tdp_watts=300,
    )
)

_register_gpu(
    GPUSpecs(
        name="AMD Instinct MI250X",
        codename="Aldebaran",
        gfx_version="gfx90a",  # Same as MI210
        architecture=GPUArchitecture.CDNA2,
        compute_units=220,  # Dual GCD
        stream_processors=14080,
        base_clock_mhz=1700,
        boost_clock_mhz=1700,
        max_waves_per_cu=32,
        wavefront_size=64,
        memory=MemorySpecs(
            vram_gb=128.0,
            memory_bandwidth_gbps=3276.8,  # Dual GCD aggregate
            l2_cache_mb=16.0,
            memory_type=MemoryType.HBM2E,
            memory_bus_width=8192,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=64,
            fp16_ops_per_cycle=256,
            bf16_ops_per_cycle=256,
            fp64_ops_per_cycle=64,
            mfma_available=True,
            mfma_shapes=["32x32x8", "16x16x16", "4x4x4"],
        ),
        tdp_watts=560,
    )
)

_register_gpu(
    GPUSpecs(
        name="AMD Instinct MI300X",
        codename="Aqua Vanjaram",
        gfx_version="gfx942",
        architecture=GPUArchitecture.CDNA3,
        compute_units=304,
        stream_processors=19456,
        base_clock_mhz=2100,
        boost_clock_mhz=2100,
        max_waves_per_cu=32,
        wavefront_size=64,
        memory=MemorySpecs(
            vram_gb=192.0,
            memory_bandwidth_gbps=5300.0,
            l2_cache_mb=256.0,  # Unified memory with CPU
            memory_type=MemoryType.HBM3,
            memory_bus_width=8192,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=64,
            fp16_ops_per_cycle=512,
            bf16_ops_per_cycle=512,
            fp64_ops_per_cycle=64,
            int8_ops_per_cycle=1024,
            mfma_available=True,
            mfma_shapes=["32x32x16", "16x16x32", "4x4x4"],
        ),
        tdp_watts=750,
    )
)

# --- RDNA GPUs (Consumer/Workstation) ---

_register_gpu(
    GPUSpecs(
        name="AMD Radeon RX 7900 XTX",
        codename="Navi 31",
        gfx_version="gfx1100",
        architecture=GPUArchitecture.RDNA3,
        compute_units=96,
        stream_processors=6144,
        base_clock_mhz=1900,
        boost_clock_mhz=2500,
        max_waves_per_cu=32,
        wavefront_size=32,
        memory=MemorySpecs(
            vram_gb=24.0,
            memory_bandwidth_gbps=960.0,
            l2_cache_mb=6.0,
            infinity_cache_mb=96.0,
            memory_type=MemoryType.GDDR6,
            memory_bus_width=384,
            memory_clock_mhz=2500,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=128,  # Dual-issue
            fp16_ops_per_cycle=256,
            bf16_ops_per_cycle=256,
            wmma_available=True,
        ),
        tdp_watts=355,
    )
)

_register_gpu(
    GPUSpecs(
        name="AMD Radeon RX 7900 XT",
        codename="Navi 31",
        gfx_version="gfx1100",
        architecture=GPUArchitecture.RDNA3,
        compute_units=84,
        stream_processors=5376,
        base_clock_mhz=1500,
        boost_clock_mhz=2400,
        max_waves_per_cu=32,
        wavefront_size=32,
        memory=MemorySpecs(
            vram_gb=20.0,
            memory_bandwidth_gbps=800.0,
            l2_cache_mb=6.0,
            infinity_cache_mb=80.0,
            memory_type=MemoryType.GDDR6,
            memory_bus_width=320,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=128,
            fp16_ops_per_cycle=256,
            bf16_ops_per_cycle=256,
            wmma_available=True,
        ),
        tdp_watts=315,
    )
)

_register_gpu(
    GPUSpecs(
        name="AMD Radeon RX 7800 XT",
        codename="Navi 32",
        gfx_version="gfx1101",
        architecture=GPUArchitecture.RDNA3,
        compute_units=60,
        stream_processors=3840,
        base_clock_mhz=1295,
        boost_clock_mhz=2430,
        max_waves_per_cu=32,
        wavefront_size=32,
        memory=MemorySpecs(
            vram_gb=16.0,
            memory_bandwidth_gbps=624.0,
            l2_cache_mb=4.0,
            infinity_cache_mb=64.0,
            memory_type=MemoryType.GDDR6,
            memory_bus_width=256,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=128,
            fp16_ops_per_cycle=256,
            wmma_available=True,
        ),
        tdp_watts=263,
    )
)

_register_gpu(
    GPUSpecs(
        name="AMD Radeon RX 7600",
        codename="Navi 33",
        gfx_version="gfx1102",
        architecture=GPUArchitecture.RDNA3,
        compute_units=32,
        stream_processors=2048,
        base_clock_mhz=1720,
        boost_clock_mhz=2655,
        max_waves_per_cu=32,
        wavefront_size=32,
        memory=MemorySpecs(
            vram_gb=8.0,
            memory_bandwidth_gbps=288.0,
            l2_cache_mb=2.0,
            infinity_cache_mb=32.0,
            memory_type=MemoryType.GDDR6,
            memory_bus_width=128,
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=128,
            fp16_ops_per_cycle=256,
            wmma_available=True,
        ),
        tdp_watts=165,
    )
)

# APU with RDNA3
_register_gpu(
    GPUSpecs(
        name="AMD Ryzen AI (Strix Point)",
        codename="Strix Point",
        gfx_version="gfx1103",
        architecture=GPUArchitecture.RDNA3,
        compute_units=12,
        stream_processors=768,
        base_clock_mhz=2200,
        boost_clock_mhz=2900,
        max_waves_per_cu=32,
        wavefront_size=32,
        memory=MemorySpecs(
            vram_gb=0,  # Shared system memory
            memory_bandwidth_gbps=102.4,  # DDR5-6400 dual channel
            l2_cache_mb=2.0,
            infinity_cache_mb=16.0,
            memory_type=MemoryType.GDDR6,  # Actually DDR5 unified
        ),
        compute=ComputeCapability(
            fp32_ops_per_cycle=128,
            fp16_ops_per_cycle=256,
            wmma_available=True,
        ),
        tdp_watts=28,  # Configurable TDP
    )
)


class GPUModel:
    """
    GPU hardware model for performance analysis.

    Provides hardware-aware performance bounds and predictions.
    """

    def __init__(self, gfx_version: str):
        """
        Initialize GPU model.

        Args:
            gfx_version: GPU version string (e.g., "gfx1100", "gfx90a")
        """
        self.gfx_version = gfx_version
        self.specs = self._lookup_specs(gfx_version)

    def _lookup_specs(self, gfx_version: str) -> GPUSpecs:
        """Look up GPU specs from database."""
        if gfx_version in AMD_GPU_DATABASE:
            return AMD_GPU_DATABASE[gfx_version]

        # Return generic specs if not found
        return GPUSpecs(
            name=f"Unknown GPU ({gfx_version})",
            gfx_version=gfx_version,
        )

    @property
    def peak_compute_tflops(self) -> Dict[str, float]:
        """Get peak compute throughput by precision."""
        return {
            "fp32": self.specs.peak_fp32_tflops,
            "fp16": self.specs.peak_fp16_tflops,
            "int8": self.specs.peak_int8_tops,
        }

    @property
    def memory_bandwidth_gbps(self) -> float:
        """Get memory bandwidth."""
        return self.specs.memory.memory_bandwidth_gbps

    @property
    def ridge_point(self) -> float:
        """
        Calculate roofline ridge point.

        Ridge point = peak_compute / memory_bandwidth
        Above this arithmetic intensity, we're compute-bound.
        """
        peak_flops = self.specs.peak_fp32_tflops * 1e12  # Convert to FLOPS
        bw = self.specs.memory.memory_bandwidth_gbps * 1e9  # Convert to bytes/s

        if bw == 0:
            return 0.0

        return peak_flops / bw  # FLOPS per byte

    def max_occupancy(
        self,
        registers_per_thread: int,
        shared_mem_per_block: int,
        threads_per_block: int,
    ) -> float:
        """
        Calculate maximum achievable occupancy.

        Args:
            registers_per_thread: Registers used per thread
            shared_mem_per_block: Shared memory per block (bytes)
            threads_per_block: Threads per block

        Returns:
            Maximum occupancy as fraction (0-1)
        """
        # Limits per CU
        max_waves = self.specs.max_waves_per_cu
        wavefront_size = self.specs.wavefront_size

        # Register limit
        total_regs_per_cu = 65536  # Typical
        waves_from_regs = total_regs_per_cu // (registers_per_thread * wavefront_size)

        # Shared memory limit
        shared_mem_per_cu = self.specs.memory.l1_cache_kb_per_cu * 1024
        waves_from_shmem = shared_mem_per_cu // max(shared_mem_per_block, 1)

        # Wave limit from block size
        waves_per_block = (threads_per_block + wavefront_size - 1) // wavefront_size

        # Minimum of all limits
        achievable_waves = min(max_waves, waves_from_regs, waves_from_shmem * waves_per_block)

        return achievable_waves / max_waves

    def estimate_kernel_time(
        self,
        flops: int,
        bytes_transferred: int,
        arithmetic_intensity: Optional[float] = None,
    ) -> float:
        """
        Estimate kernel execution time using roofline model.

        Args:
            flops: Total floating point operations
            bytes_transferred: Total memory bytes transferred
            arithmetic_intensity: FLOPS/byte (calculated if not provided)

        Returns:
            Estimated time in milliseconds
        """
        if arithmetic_intensity is None:
            arithmetic_intensity = flops / max(bytes_transferred, 1)

        # Peak performance (FLOPS)
        peak_flops_per_s = self.specs.peak_fp32_tflops * 1e12

        # Memory bandwidth (bytes/s)
        bw = self.specs.memory.memory_bandwidth_gbps * 1e9

        # Roofline bound
        if arithmetic_intensity < self.ridge_point:
            # Memory bound
            achievable_flops_per_s = bw * arithmetic_intensity
        else:
            # Compute bound
            achievable_flops_per_s = peak_flops_per_s

        # Time estimate
        time_s = flops / achievable_flops_per_s
        return time_s * 1000  # Convert to ms

    def get_optimization_potential(
        self, current_efficiency: float, bottleneck: str
    ) -> Dict[str, Any]:
        """
        Estimate optimization potential.

        Args:
            current_efficiency: Current efficiency (0-1)
            bottleneck: Current bottleneck type

        Returns:
            Dict with optimization potential and suggestions
        """
        potential = {
            "current_efficiency": current_efficiency,
            "theoretical_max": 0.85,  # Practical limit
            "improvement_potential_pct": 0.0,
            "suggestions": [],
        }

        gap = potential["theoretical_max"] - current_efficiency
        potential["improvement_potential_pct"] = gap * 100

        if bottleneck == "memory":
            potential["suggestions"].extend(
                [
                    "Consider tiling to improve cache utilization",
                    "Check memory coalescing",
                    f"Target arithmetic intensity > {self.ridge_point:.1f} FLOPS/byte",
                ]
            )
        elif bottleneck == "compute":
            potential["suggestions"].extend(
                [
                    "Consider FP16/BF16 for 2x throughput",
                    "Check MFMA utilization"
                    if self.specs.compute.mfma_available
                    else "Improve vectorization",
                ]
            )
        elif bottleneck == "occupancy":
            potential["suggestions"].extend(
                [
                    "Reduce register pressure",
                    "Adjust workgroup size",
                ]
            )

        return potential
