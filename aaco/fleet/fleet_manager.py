"""
AACO-SIGMA Fleet Manager

Manages fleet of GPU nodes for distributed profiling.
Handles node registration, discovery, and coordination.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
import time


class NodeStatus(Enum):
    """Status of a fleet node."""

    ONLINE = auto()  # Available for work
    BUSY = auto()  # Currently executing job
    OFFLINE = auto()  # Not reachable
    MAINTENANCE = auto()  # Under maintenance
    ERROR = auto()  # Error state


@dataclass
class GPUInfo:
    """GPU information on a node."""

    index: int = 0
    name: str = ""
    gfx_version: str = ""
    memory_gb: float = 0.0
    driver_version: str = ""
    compute_units: int = 0


@dataclass
class GPUNode:
    """A GPU node in the fleet."""

    # Identity
    node_id: str
    hostname: str

    # Connection
    address: str = ""
    port: int = 8080

    # Status
    status: NodeStatus = NodeStatus.OFFLINE
    last_seen: float = 0.0

    # GPUs on this node
    gpus: List[GPUInfo] = field(default_factory=list)

    # Capabilities
    rocm_version: str = ""
    python_version: str = ""

    # Performance baseline
    baseline_tflops: Dict[str, float] = field(default_factory=dict)

    # Tags for grouping
    tags: List[str] = field(default_factory=list)

    # Metadata
    registered_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return self.status == NodeStatus.ONLINE

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


@dataclass
class FleetConfig:
    """Fleet configuration."""

    # Discovery
    discovery_enabled: bool = True
    discovery_port: int = 8080
    heartbeat_interval_s: int = 30
    offline_threshold_s: int = 90

    # Work distribution
    max_concurrent_jobs: int = 10
    prefer_idle_nodes: bool = True

    # Security
    require_auth: bool = False
    api_key: Optional[str] = None


class FleetManager:
    """
    Manages a fleet of GPU nodes.

    Responsibilities:
    - Node registration and discovery
    - Status tracking
    - Work distribution
    - Fleet-wide coordination
    """

    def __init__(self, config: Optional[FleetConfig] = None):
        self.config = config or FleetConfig()
        self._nodes: Dict[str, GPUNode] = {}
        self._status_callbacks: List[Callable] = []

    def register_node(self, node: GPUNode) -> bool:
        """
        Register a new node with the fleet.

        Args:
            node: Node to register

        Returns:
            True if registration successful
        """
        if node.node_id in self._nodes:
            # Update existing
            existing = self._nodes[node.node_id]
            existing.status = node.status
            existing.last_seen = time.time()
            existing.gpus = node.gpus
            return True

        node.registered_at = time.time()
        node.last_seen = time.time()
        node.status = NodeStatus.ONLINE

        self._nodes[node.node_id] = node
        self._notify_status_change(node, NodeStatus.ONLINE)

        return True

    def unregister_node(self, node_id: str) -> bool:
        """
        Remove a node from the fleet.
        """
        if node_id in self._nodes:
            node = self._nodes.pop(node_id)
            self._notify_status_change(node, NodeStatus.OFFLINE)
            return True
        return False

    def heartbeat(self, node_id: str, status: NodeStatus = NodeStatus.ONLINE) -> bool:
        """
        Receive heartbeat from a node.
        """
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]
        old_status = node.status

        node.last_seen = time.time()
        node.status = status

        if old_status != status:
            self._notify_status_change(node, status)

        return True

    def update_status(self) -> None:
        """
        Update status of all nodes based on heartbeat timeout.
        """
        current_time = time.time()
        threshold = self.config.offline_threshold_s

        for node in self._nodes.values():
            if node.status == NodeStatus.ONLINE:
                if current_time - node.last_seen > threshold:
                    node.status = NodeStatus.OFFLINE
                    self._notify_status_change(node, NodeStatus.OFFLINE)

    def get_node(self, node_id: str) -> Optional[GPUNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[GPUNode]:
        """Get all registered nodes."""
        return list(self._nodes.values())

    def get_online_nodes(self) -> List[GPUNode]:
        """Get all online nodes."""
        return [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]

    def get_available_nodes(self) -> List[GPUNode]:
        """Get nodes available for work."""
        return [n for n in self._nodes.values() if n.is_available]

    def get_nodes_by_gpu(self, gfx_version: str) -> List[GPUNode]:
        """Get nodes with specific GPU type."""
        result = []
        for node in self._nodes.values():
            if any(gpu.gfx_version == gfx_version for gpu in node.gpus):
                result.append(node)
        return result

    def get_nodes_by_tag(self, tag: str) -> List[GPUNode]:
        """Get nodes with specific tag."""
        return [n for n in self._nodes.values() if tag in n.tags]

    def select_node(self, requirements: Optional[Dict[str, Any]] = None) -> Optional[GPUNode]:
        """
        Select best node for a job.

        Args:
            requirements: Optional job requirements

        Returns:
            Selected node or None
        """
        available = self.get_available_nodes()

        if not available:
            return None

        # Apply requirements filter
        if requirements:
            gpu_type = requirements.get("gfx_version")
            if gpu_type:
                available = [n for n in available if any(g.gfx_version == gpu_type for g in n.gpus)]

            min_memory = requirements.get("min_memory_gb")
            if min_memory:
                available = [n for n in available if any(g.memory_gb >= min_memory for g in n.gpus)]

            tags = requirements.get("tags", [])
            if tags:
                available = [n for n in available if all(tag in n.tags for tag in tags)]

        if not available:
            return None

        # Prefer idle nodes
        if self.config.prefer_idle_nodes:
            idle = [n for n in available if n.status == NodeStatus.ONLINE]
            if idle:
                available = idle

        # Return first available (could be more sophisticated)
        return available[0]

    def set_node_maintenance(self, node_id: str, maintenance: bool) -> bool:
        """
        Set node maintenance mode.
        """
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]
        if maintenance:
            node.status = NodeStatus.MAINTENANCE
        else:
            node.status = NodeStatus.ONLINE

        self._notify_status_change(node, node.status)
        return True

    def get_fleet_summary(self) -> Dict[str, Any]:
        """
        Get summary of fleet status.
        """
        nodes = list(self._nodes.values())

        summary = {
            "total_nodes": len(nodes),
            "online_nodes": sum(1 for n in nodes if n.status == NodeStatus.ONLINE),
            "busy_nodes": sum(1 for n in nodes if n.status == NodeStatus.BUSY),
            "offline_nodes": sum(1 for n in nodes if n.status == NodeStatus.OFFLINE),
            "maintenance_nodes": sum(1 for n in nodes if n.status == NodeStatus.MAINTENANCE),
            "total_gpus": sum(n.gpu_count for n in nodes),
            "available_gpus": sum(n.gpu_count for n in nodes if n.is_available),
            "gpu_types": {},
        }

        # Count GPU types
        for node in nodes:
            for gpu in node.gpus:
                gfx = gpu.gfx_version
                if gfx not in summary["gpu_types"]:
                    summary["gpu_types"][gfx] = 0
                summary["gpu_types"][gfx] += 1

        return summary

    def register_status_callback(self, callback: Callable) -> None:
        """Register callback for status changes."""
        self._status_callbacks.append(callback)

    def _notify_status_change(self, node: GPUNode, new_status: NodeStatus) -> None:
        """Notify callbacks of status change."""
        for callback in self._status_callbacks:
            try:
                callback(node, new_status)
            except Exception:
                pass  # Log error in production
