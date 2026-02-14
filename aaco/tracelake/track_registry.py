"""
AACO-SIGMA Track Registry

Unified track namespace for Perfetto traces.
Ensures consistent track naming across all AACO components.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import hashlib


class TrackType(Enum):
    """Track type classification."""

    PROCESS = auto()  # Process-level track
    THREAD = auto()  # Thread-level track
    COUNTER = auto()  # Counter track
    ASYNC = auto()  # Async spans
    FLOW = auto()  # Flow arrows
    MARKER = auto()  # Instant markers
    GPU = auto()  # GPU track
    KERNEL = auto()  # Kernel execution track


class TrackCategory(Enum):
    """High-level track categories."""

    INFRASTRUCTURE = "infra"  # Process, thread, CPU
    GPU_ACTIVITY = "gpu"  # GPU kernels, queues
    COUNTERS = "counter"  # Metrics/counters
    MODEL = "model"  # Inference iterations
    MEMORY = "memory"  # Allocations, transfers
    CAPSULE = "capsule"  # Measurement capsule
    NOISE = "noise"  # Interference/noise
    PROFILER = "profiler"  # Profiler overhead
    CUSTOM = "custom"  # User-defined


@dataclass
class TrackDefinition:
    """Definition of a track in the registry."""

    name: str
    track_type: TrackType
    category: TrackCategory
    description: str = ""
    unit: str = ""  # For counter tracks
    color: str = ""  # Track color hint
    parent_track: Optional[str] = None  # For hierarchical tracks
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def track_id(self) -> int:
        """Compute stable track ID from name."""
        return int(hashlib.sha256(self.name.encode()).hexdigest()[:8], 16) % (2**31)

    def to_perfetto_track(self) -> Dict[str, Any]:
        """Convert to Perfetto track descriptor."""
        track = {
            "trackId": self.track_id,
            "name": self.name,
            "category": self.category.value,
        }
        if self.parent_track:
            track["parentTrackId"] = int(
                hashlib.sha256(self.parent_track.encode()).hexdigest()[:8], 16
            ) % (2**31)
        return track


class StandardTracks:
    """Registry of standard AACO track definitions."""

    # ============================================================
    # Infrastructure Tracks
    # ============================================================

    MAIN_PROCESS = TrackDefinition(
        name="AACO Process",
        track_type=TrackType.PROCESS,
        category=TrackCategory.INFRASTRUCTURE,
        description="Main benchmark process",
    )

    MAIN_THREAD = TrackDefinition(
        name="Main Thread",
        track_type=TrackType.THREAD,
        category=TrackCategory.INFRASTRUCTURE,
        description="Main execution thread",
        parent_track="AACO Process",
    )

    CPU_AFFINITY = TrackDefinition(
        name="CPU Affinity",
        track_type=TrackType.MARKER,
        category=TrackCategory.INFRASTRUCTURE,
        description="CPU core assignments",
    )

    # ============================================================
    # GPU Tracks
    # ============================================================

    GPU_QUEUE = TrackDefinition(
        name="GPU Queue",
        track_type=TrackType.GPU,
        category=TrackCategory.GPU_ACTIVITY,
        description="GPU command queue",
        color="#4CAF50",  # Green
    )

    GPU_KERNEL = TrackDefinition(
        name="GPU Kernels",
        track_type=TrackType.KERNEL,
        category=TrackCategory.GPU_ACTIVITY,
        description="GPU kernel executions",
        color="#2196F3",  # Blue
    )

    GPU_COPY = TrackDefinition(
        name="GPU Memcpy",
        track_type=TrackType.GPU,
        category=TrackCategory.GPU_ACTIVITY,
        description="GPU memory transfers",
        color="#FF9800",  # Orange
    )

    HIP_API = TrackDefinition(
        name="HIP API",
        track_type=TrackType.ASYNC,
        category=TrackCategory.GPU_ACTIVITY,
        description="HIP runtime API calls",
        color="#9C27B0",  # Purple
    )

    # ============================================================
    # Counter Tracks
    # ============================================================

    GPU_UTILIZATION = TrackDefinition(
        name="GPU Utilization",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="GPU compute utilization",
        unit="%",
    )

    GPU_MEMORY = TrackDefinition(
        name="GPU Memory",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="GPU memory usage",
        unit="MB",
    )

    GPU_CLOCK = TrackDefinition(
        name="GPU Clock",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="GPU clock frequency",
        unit="MHz",
    )

    GPU_TEMPERATURE = TrackDefinition(
        name="GPU Temperature",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="GPU temperature",
        unit="°C",
    )

    GPU_POWER = TrackDefinition(
        name="GPU Power",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="GPU power consumption",
        unit="W",
    )

    CPU_UTILIZATION = TrackDefinition(
        name="CPU Utilization",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="CPU utilization",
        unit="%",
    )

    MEMORY_PRESSURE = TrackDefinition(
        name="Memory Pressure",
        track_type=TrackType.COUNTER,
        category=TrackCategory.COUNTERS,
        description="Memory pressure (PSI)",
        unit="%",
    )

    # ============================================================
    # Model/Inference Tracks
    # ============================================================

    INFERENCE_ITERATION = TrackDefinition(
        name="Inference",
        track_type=TrackType.ASYNC,
        category=TrackCategory.MODEL,
        description="Inference iterations",
        color="#E91E63",  # Pink
    )

    INFERENCE_PREFILL = TrackDefinition(
        name="Prefill Phase",
        track_type=TrackType.ASYNC,
        category=TrackCategory.MODEL,
        description="LLM prefill phase",
        parent_track="Inference",
        color="#F44336",  # Red
    )

    INFERENCE_DECODE = TrackDefinition(
        name="Decode Phase",
        track_type=TrackType.ASYNC,
        category=TrackCategory.MODEL,
        description="LLM decode phase",
        parent_track="Inference",
        color="#3F51B5",  # Indigo
    )

    # ============================================================
    # Memory Tracks
    # ============================================================

    HOST_ALLOC = TrackDefinition(
        name="Host Allocation",
        track_type=TrackType.MARKER,
        category=TrackCategory.MEMORY,
        description="Host memory allocations",
    )

    DEVICE_ALLOC = TrackDefinition(
        name="Device Allocation",
        track_type=TrackType.MARKER,
        category=TrackCategory.MEMORY,
        description="Device memory allocations",
    )

    MEMORY_TRANSFER = TrackDefinition(
        name="Memory Transfer",
        track_type=TrackType.ASYNC,
        category=TrackCategory.MEMORY,
        description="Host-device memory transfers",
    )

    # ============================================================
    # Capsule/Measurement Tracks
    # ============================================================

    CAPSULE = TrackDefinition(
        name="Measurement Capsule",
        track_type=TrackType.ASYNC,
        category=TrackCategory.CAPSULE,
        description="Capsule lifecycle",
        color="#795548",  # Brown
    )

    CAPSULE_HEALTH = TrackDefinition(
        name="Capsule Health",
        track_type=TrackType.COUNTER,
        category=TrackCategory.CAPSULE,
        description="Capsule health score",
        unit="%",
    )

    WARMUP = TrackDefinition(
        name="Warmup",
        track_type=TrackType.ASYNC,
        category=TrackCategory.CAPSULE,
        description="Warmup phase",
        color="#9E9E9E",  # Grey
    )

    MEASUREMENT = TrackDefinition(
        name="Measurement",
        track_type=TrackType.ASYNC,
        category=TrackCategory.CAPSULE,
        description="Measurement phase",
        color="#4CAF50",  # Green
    )

    # ============================================================
    # Noise/Interference Tracks
    # ============================================================

    NOISE_EVENT = TrackDefinition(
        name="Noise Events",
        track_type=TrackType.MARKER,
        category=TrackCategory.NOISE,
        description="Detected noise events",
        color="#F44336",  # Red
    )

    IRQ_STORM = TrackDefinition(
        name="IRQ Storm",
        track_type=TrackType.MARKER,
        category=TrackCategory.NOISE,
        description="IRQ storm detection",
    )

    CONTEXT_SWITCH = TrackDefinition(
        name="Context Switches",
        track_type=TrackType.COUNTER,
        category=TrackCategory.NOISE,
        description="Context switch rate",
    )

    THROTTLE = TrackDefinition(
        name="Throttle Events",
        track_type=TrackType.MARKER,
        category=TrackCategory.NOISE,
        description="GPU/CPU throttle events",
        color="#FF5722",  # Deep Orange
    )


class TrackRegistry:
    """
    Registry for managing track definitions.

    Provides:
    - Standard track definitions
    - Custom track registration
    - Track lookup by name/ID
    - Track namespace management
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._tracks: Dict[str, TrackDefinition] = {}
        self._track_by_id: Dict[int, TrackDefinition] = {}

        # Register all standard tracks
        self._register_standard_tracks()

    def _register_standard_tracks(self) -> None:
        """Register all standard AACO tracks."""
        standard_tracks = [
            StandardTracks.MAIN_PROCESS,
            StandardTracks.MAIN_THREAD,
            StandardTracks.CPU_AFFINITY,
            StandardTracks.GPU_QUEUE,
            StandardTracks.GPU_KERNEL,
            StandardTracks.GPU_COPY,
            StandardTracks.HIP_API,
            StandardTracks.GPU_UTILIZATION,
            StandardTracks.GPU_MEMORY,
            StandardTracks.GPU_CLOCK,
            StandardTracks.GPU_TEMPERATURE,
            StandardTracks.GPU_POWER,
            StandardTracks.CPU_UTILIZATION,
            StandardTracks.MEMORY_PRESSURE,
            StandardTracks.INFERENCE_ITERATION,
            StandardTracks.INFERENCE_PREFILL,
            StandardTracks.INFERENCE_DECODE,
            StandardTracks.HOST_ALLOC,
            StandardTracks.DEVICE_ALLOC,
            StandardTracks.MEMORY_TRANSFER,
            StandardTracks.CAPSULE,
            StandardTracks.CAPSULE_HEALTH,
            StandardTracks.WARMUP,
            StandardTracks.MEASUREMENT,
            StandardTracks.NOISE_EVENT,
            StandardTracks.IRQ_STORM,
            StandardTracks.CONTEXT_SWITCH,
            StandardTracks.THROTTLE,
        ]

        for track in standard_tracks:
            self.register(track)

    def register(self, track: TrackDefinition) -> int:
        """Register a track definition. Returns track ID."""
        self._tracks[track.name] = track
        self._track_by_id[track.track_id] = track
        return track.track_id

    def register_custom(
        self,
        name: str,
        track_type: TrackType,
        category: TrackCategory,
        description: str = "",
        unit: str = "",
        color: str = "",
        parent_track: Optional[str] = None,
    ) -> int:
        """Register a custom track. Returns track ID."""
        track = TrackDefinition(
            name=name,
            track_type=track_type,
            category=category,
            description=description,
            unit=unit,
            color=color,
            parent_track=parent_track,
        )
        return self.register(track)

    def get(self, name: str) -> Optional[TrackDefinition]:
        """Get track by name."""
        return self._tracks.get(name)

    def get_by_id(self, track_id: int) -> Optional[TrackDefinition]:
        """Get track by ID."""
        return self._track_by_id.get(track_id)

    def get_track_id(self, name: str) -> Optional[int]:
        """Get track ID by name."""
        track = self._tracks.get(name)
        return track.track_id if track else None

    def get_tracks_by_category(self, category: TrackCategory) -> List[TrackDefinition]:
        """Get all tracks in a category."""
        return [t for t in self._tracks.values() if t.category == category]

    def get_tracks_by_type(self, track_type: TrackType) -> List[TrackDefinition]:
        """Get all tracks of a type."""
        return [t for t in self._tracks.values() if t.track_type == track_type]

    def get_counter_tracks(self) -> List[TrackDefinition]:
        """Get all counter tracks."""
        return self.get_tracks_by_type(TrackType.COUNTER)

    def get_all_tracks(self) -> List[TrackDefinition]:
        """Get all registered tracks."""
        return list(self._tracks.values())

    def export_track_metadata(self) -> Dict[str, Any]:
        """Export track metadata for Perfetto."""
        return {
            "trackDefinitions": [t.to_perfetto_track() for t in self._tracks.values()],
            "categories": [c.value for c in TrackCategory],
            "trackCount": len(self._tracks),
        }

    # ============================================================
    # GPU-specific track factories
    # ============================================================

    def get_gpu_kernel_track(self, gpu_id: int = 0) -> TrackDefinition:
        """Get or create GPU kernel track for specific GPU."""
        name = f"GPU {gpu_id} Kernels"
        if name not in self._tracks:
            self.register_custom(
                name=name,
                track_type=TrackType.KERNEL,
                category=TrackCategory.GPU_ACTIVITY,
                description=f"GPU {gpu_id} kernel executions",
                color="#2196F3",
            )
        return self._tracks[name]

    def get_gpu_counter_track(
        self, gpu_id: int, counter_name: str, unit: str = ""
    ) -> TrackDefinition:
        """Get or create GPU counter track."""
        name = f"GPU {gpu_id} {counter_name}"
        if name not in self._tracks:
            self.register_custom(
                name=name,
                track_type=TrackType.COUNTER,
                category=TrackCategory.COUNTERS,
                description=f"GPU {gpu_id} {counter_name}",
                unit=unit,
            )
        return self._tracks[name]

    # ============================================================
    # Kernel-specific track factories
    # ============================================================

    def get_kernel_family_track(self, family_name: str) -> TrackDefinition:
        """Get or create track for kernel family."""
        name = f"Kernel: {family_name}"
        if name not in self._tracks:
            self.register_custom(
                name=name,
                track_type=TrackType.KERNEL,
                category=TrackCategory.GPU_ACTIVITY,
                description=f"Kernels in family: {family_name}",
            )
        return self._tracks[name]
