"""
AACO-SIGMA CPU Topology and NUMA Awareness

Provides detailed CPU topology discovery and pinning capabilities
for deterministic performance measurements.
"""

import os
import re
import json
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Set, Tuple, Any
from collections import defaultdict


@dataclass
class CoreInfo:
    """Information about a single CPU core."""
    core_id: int                    # Logical CPU ID
    physical_core_id: int           # Physical core ID
    socket_id: int                  # Socket/package ID
    numa_node: int                  # NUMA node
    sibling_id: Optional[int]       # SMT sibling (if HT enabled)
    
    # Cache topology
    l1d_cache_id: int = 0
    l1i_cache_id: int = 0
    l2_cache_id: int = 0
    l3_cache_id: int = 0
    
    # Frequency info
    base_freq_khz: int = 0
    max_freq_khz: int = 0
    current_freq_khz: int = 0
    
    # Online status
    is_online: bool = True
    is_isolated: bool = False  # isolcpus kernel param


@dataclass
class NUMANode:
    """Information about a NUMA node."""
    node_id: int
    cores: List[int] = field(default_factory=list)
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    distance_map: Dict[int, int] = field(default_factory=dict)  # node_id -> distance
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoreSet:
    """A set of cores for workload pinning."""
    cores: List[int]
    name: str = ""
    numa_nodes: Set[int] = field(default_factory=set)
    is_exclusive: bool = False
    smt_policy: str = "include"  # include, exclude, primary_only
    
    def to_cpuset_str(self) -> str:
        """Convert to cpuset format string."""
        if not self.cores:
            return ""
        
        # Group consecutive cores into ranges
        sorted_cores = sorted(self.cores)
        ranges = []
        start = sorted_cores[0]
        end = start
        
        for core in sorted_cores[1:]:
            if core == end + 1:
                end = core
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = core
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ",".join(ranges)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['numa_nodes'] = list(self.numa_nodes)
        return d


class TopologyPolicy(Enum):
    """Policies for core selection."""
    COMPACT = auto()      # Minimize NUMA nodes, maximize cache sharing
    SPREAD = auto()       # Spread across NUMA nodes
    NUMA_LOCAL = auto()   # All cores from one NUMA node
    SOCKET_LOCAL = auto() # All cores from one socket
    NO_SMT = auto()       # Avoid SMT siblings
    SMT_PAIRS = auto()    # Use SMT pairs together


class CPUTopology:
    """
    CPU topology discovery and management.
    
    Discovers:
    - Physical vs logical cores
    - NUMA topology
    - Cache hierarchy
    - SMT/HyperThreading siblings
    - Isolated cores (isolcpus)
    """
    
    def __init__(self):
        self.cores: Dict[int, CoreInfo] = {}
        self.numa_nodes: Dict[int, NUMANode] = {}
        self.sockets: Dict[int, List[int]] = defaultdict(list)
        self.smt_siblings: Dict[int, int] = {}  # core -> sibling
        self.physical_to_logical: Dict[Tuple[int, int], List[int]] = {}  # (socket, phys_core) -> [logical]
        
        self._isolated_cores: Set[int] = set()
        self._online_cores: Set[int] = set()
        
        self._discover()
    
    def _discover(self) -> None:
        """Discover CPU topology from sysfs."""
        cpu_base = Path("/sys/devices/system/cpu")
        
        if not cpu_base.exists():
            # Fallback for non-Linux
            self._fallback_discovery()
            return
        
        # Find all CPU directories
        cpu_dirs = sorted(cpu_base.glob("cpu[0-9]*"))
        
        # Get isolated cores from kernel cmdline
        self._get_isolated_cores()
        
        for cpu_dir in cpu_dirs:
            cpu_id = int(cpu_dir.name[3:])
            
            # Check if online
            online_file = cpu_dir / "online"
            if online_file.exists():
                try:
                    is_online = online_file.read_text().strip() == "1"
                except:
                    is_online = True
            else:
                is_online = True  # cpu0 doesn't have online file
            
            if not is_online:
                continue
            
            self._online_cores.add(cpu_id)
            
            # Get topology info
            topo_dir = cpu_dir / "topology"
            
            physical_core = cpu_id
            socket_id = 0
            
            if (topo_dir / "core_id").exists():
                try:
                    physical_core = int((topo_dir / "core_id").read_text().strip())
                except:
                    pass
            
            if (topo_dir / "physical_package_id").exists():
                try:
                    socket_id = int((topo_dir / "physical_package_id").read_text().strip())
                except:
                    pass
            
            # Get SMT siblings
            sibling_id = None
            if (topo_dir / "thread_siblings_list").exists():
                try:
                    siblings_str = (topo_dir / "thread_siblings_list").read_text().strip()
                    siblings = self._parse_cpu_list(siblings_str)
                    for sib in siblings:
                        if sib != cpu_id:
                            sibling_id = sib
                            break
                except:
                    pass
            
            # Get NUMA node
            numa_node = 0
            node_links = list(cpu_dir.glob("node*"))
            if node_links:
                numa_node = int(node_links[0].name[4:])
            
            # Get frequency info
            freq_dir = cpu_dir / "cpufreq"
            base_freq = max_freq = current_freq = 0
            
            if (freq_dir / "base_frequency").exists():
                try:
                    base_freq = int((freq_dir / "base_frequency").read_text().strip())
                except:
                    pass
            
            if (freq_dir / "cpuinfo_max_freq").exists():
                try:
                    max_freq = int((freq_dir / "cpuinfo_max_freq").read_text().strip())
                except:
                    pass
            
            if (freq_dir / "scaling_cur_freq").exists():
                try:
                    current_freq = int((freq_dir / "scaling_cur_freq").read_text().strip())
                except:
                    pass
            
            # Get cache info
            l1d_id = l1i_id = l2_id = l3_id = 0
            cache_dir = cpu_dir / "cache"
            if cache_dir.exists():
                for index_dir in cache_dir.glob("index*"):
                    try:
                        level = int((index_dir / "level").read_text().strip())
                        cache_type = (index_dir / "type").read_text().strip().lower()
                        cache_id = int((index_dir / "id").read_text().strip())
                        
                        if level == 1 and cache_type == "data":
                            l1d_id = cache_id
                        elif level == 1 and cache_type == "instruction":
                            l1i_id = cache_id
                        elif level == 2:
                            l2_id = cache_id
                        elif level == 3:
                            l3_id = cache_id
                    except:
                        pass
            
            # Create CoreInfo
            core_info = CoreInfo(
                core_id=cpu_id,
                physical_core_id=physical_core,
                socket_id=socket_id,
                numa_node=numa_node,
                sibling_id=sibling_id,
                l1d_cache_id=l1d_id,
                l1i_cache_id=l1i_id,
                l2_cache_id=l2_id,
                l3_cache_id=l3_id,
                base_freq_khz=base_freq,
                max_freq_khz=max_freq,
                current_freq_khz=current_freq,
                is_online=is_online,
                is_isolated=cpu_id in self._isolated_cores
            )
            
            self.cores[cpu_id] = core_info
            self.sockets[socket_id].append(cpu_id)
            
            # Track SMT siblings
            if sibling_id is not None:
                self.smt_siblings[cpu_id] = sibling_id
            
            # Track physical to logical mapping
            key = (socket_id, physical_core)
            if key not in self.physical_to_logical:
                self.physical_to_logical[key] = []
            self.physical_to_logical[key].append(cpu_id)
        
        # Discover NUMA topology
        self._discover_numa()
    
    def _get_isolated_cores(self) -> None:
        """Get isolated cores from kernel cmdline."""
        try:
            cmdline = Path("/proc/cmdline").read_text()
            match = re.search(r'isolcpus=([^\s]+)', cmdline)
            if match:
                self._isolated_cores = set(self._parse_cpu_list(match.group(1)))
        except:
            pass
    
    def _parse_cpu_list(self, cpu_str: str) -> List[int]:
        """Parse CPU list string (e.g., '0-3,5,7-9')."""
        cpus = []
        for part in cpu_str.split(','):
            if '-' in part:
                start, end = part.split('-')
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        return cpus
    
    def _discover_numa(self) -> None:
        """Discover NUMA topology."""
        numa_base = Path("/sys/devices/system/node")
        
        if not numa_base.exists():
            # Single NUMA node fallback
            node = NUMANode(
                node_id=0,
                cores=list(self.cores.keys()),
                distance_map={0: 10}
            )
            self.numa_nodes[0] = node
            return
        
        for node_dir in numa_base.glob("node[0-9]*"):
            node_id = int(node_dir.name[4:])
            
            # Get cores in this node
            cpulist_file = node_dir / "cpulist"
            node_cores = []
            if cpulist_file.exists():
                try:
                    cpu_str = cpulist_file.read_text().strip()
                    node_cores = self._parse_cpu_list(cpu_str)
                except:
                    pass
            
            # Get memory info
            meminfo_file = node_dir / "meminfo"
            mem_total = mem_free = 0
            if meminfo_file.exists():
                try:
                    for line in meminfo_file.read_text().split('\n'):
                        if 'MemTotal' in line:
                            mem_total = int(line.split()[3]) // 1024  # KB to MB
                        elif 'MemFree' in line:
                            mem_free = int(line.split()[3]) // 1024
                except:
                    pass
            
            # Get distance map
            distance_file = node_dir / "distance"
            distance_map = {}
            if distance_file.exists():
                try:
                    distances = distance_file.read_text().strip().split()
                    for i, dist in enumerate(distances):
                        distance_map[i] = int(dist)
                except:
                    pass
            
            self.numa_nodes[node_id] = NUMANode(
                node_id=node_id,
                cores=node_cores,
                memory_total_mb=mem_total,
                memory_free_mb=mem_free,
                distance_map=distance_map
            )
    
    def _fallback_discovery(self) -> None:
        """Fallback discovery for non-Linux systems."""
        cpu_count = os.cpu_count() or 1
        
        for i in range(cpu_count):
            self.cores[i] = CoreInfo(
                core_id=i,
                physical_core_id=i // 2,  # Assume 2-way SMT
                socket_id=0,
                numa_node=0,
                sibling_id=i + 1 if i % 2 == 0 else i - 1,
                is_online=True
            )
            self.sockets[0].append(i)
        
        self.numa_nodes[0] = NUMANode(
            node_id=0,
            cores=list(range(cpu_count)),
            distance_map={0: 10}
        )
    
    @property
    def num_cores(self) -> int:
        """Total number of online cores."""
        return len(self.cores)
    
    @property
    def num_physical_cores(self) -> int:
        """Number of physical cores (excluding SMT)."""
        return len(self.physical_to_logical)
    
    @property
    def num_sockets(self) -> int:
        """Number of CPU sockets."""
        return len(self.sockets)
    
    @property
    def num_numa_nodes(self) -> int:
        """Number of NUMA nodes."""
        return len(self.numa_nodes)
    
    @property
    def has_smt(self) -> bool:
        """Whether SMT (HyperThreading) is enabled."""
        return len(self.smt_siblings) > 0
    
    def get_physical_cores(self) -> List[int]:
        """Get one logical core per physical core (no SMT siblings)."""
        physical_cores = []
        seen_physical = set()
        
        for core_id, core_info in self.cores.items():
            key = (core_info.socket_id, core_info.physical_core_id)
            if key not in seen_physical:
                seen_physical.add(key)
                physical_cores.append(core_id)
        
        return sorted(physical_cores)
    
    def get_numa_cores(self, node_id: int) -> List[int]:
        """Get all cores in a NUMA node."""
        if node_id in self.numa_nodes:
            return self.numa_nodes[node_id].cores
        return []
    
    def get_socket_cores(self, socket_id: int) -> List[int]:
        """Get all cores in a socket."""
        if socket_id in self.sockets:
            return self.sockets[socket_id]
        return []
    
    def get_non_isolated_cores(self) -> List[int]:
        """Get cores not in isolcpus."""
        return [c for c in self.cores.keys() if c not in self._isolated_cores]
    
    def select_cores(self, count: int, policy: TopologyPolicy = TopologyPolicy.COMPACT,
                     numa_node: Optional[int] = None,
                     exclude_smt: bool = False) -> CoreSet:
        """
        Select cores based on policy.
        
        Args:
            count: Number of cores to select
            policy: Selection policy
            numa_node: Preferred NUMA node (optional)
            exclude_smt: Exclude SMT siblings
        
        Returns:
            CoreSet with selected cores
        """
        available = list(self.cores.keys())
        
        # Filter by NUMA node if specified
        if numa_node is not None and numa_node in self.numa_nodes:
            available = [c for c in available if c in self.numa_nodes[numa_node].cores]
        
        # Filter out SMT siblings if requested
        if exclude_smt:
            physical = self.get_physical_cores()
            available = [c for c in available if c in physical]
        
        # Filter out isolated cores
        available = [c for c in available if c not in self._isolated_cores]
        
        if len(available) < count:
            count = len(available)
        
        selected = []
        
        if policy == TopologyPolicy.COMPACT:
            # Select cores that share L3 cache
            selected = self._select_compact(available, count)
        
        elif policy == TopologyPolicy.SPREAD:
            # Spread across NUMA nodes equally
            selected = self._select_spread(available, count)
        
        elif policy == TopologyPolicy.NUMA_LOCAL:
            # Select from one NUMA node
            if numa_node is None:
                numa_node = 0
            numa_cores = self.get_numa_cores(numa_node)
            available = [c for c in available if c in numa_cores]
            selected = available[:count]
        
        elif policy == TopologyPolicy.SOCKET_LOCAL:
            # Select from one socket
            socket_id = 0
            socket_cores = self.get_socket_cores(socket_id)
            available = [c for c in available if c in socket_cores]
            selected = available[:count]
        
        elif policy == TopologyPolicy.NO_SMT:
            # Select physical cores only
            physical = self.get_physical_cores()
            available = [c for c in available if c in physical]
            selected = available[:count]
        
        elif policy == TopologyPolicy.SMT_PAIRS:
            # Select complete SMT pairs
            selected = self._select_smt_pairs(available, count)
        
        else:
            selected = available[:count]
        
        # Determine NUMA nodes covered
        numa_nodes = set()
        for core_id in selected:
            if core_id in self.cores:
                numa_nodes.add(self.cores[core_id].numa_node)
        
        return CoreSet(
            cores=selected,
            name=f"{policy.name}_{count}cores",
            numa_nodes=numa_nodes,
            smt_policy="exclude" if exclude_smt else "include"
        )
    
    def _select_compact(self, available: List[int], count: int) -> List[int]:
        """Select cores sharing L3 cache (compact)."""
        # Group by L3 cache
        l3_groups: Dict[int, List[int]] = defaultdict(list)
        for core_id in available:
            if core_id in self.cores:
                l3_id = self.cores[core_id].l3_cache_id
                l3_groups[l3_id].append(core_id)
        
        # Select from largest L3 group first
        selected = []
        for l3_id in sorted(l3_groups.keys(), key=lambda x: -len(l3_groups[x])):
            for core_id in l3_groups[l3_id]:
                if len(selected) >= count:
                    break
                selected.append(core_id)
            if len(selected) >= count:
                break
        
        return selected
    
    def _select_spread(self, available: List[int], count: int) -> List[int]:
        """Select cores spread across NUMA nodes."""
        # Group by NUMA node
        numa_groups: Dict[int, List[int]] = defaultdict(list)
        for core_id in available:
            if core_id in self.cores:
                numa_node = self.cores[core_id].numa_node
                numa_groups[numa_node].append(core_id)
        
        # Round-robin selection
        selected = []
        node_ids = list(numa_groups.keys())
        node_indices = {n: 0 for n in node_ids}
        
        while len(selected) < count:
            for node_id in node_ids:
                if len(selected) >= count:
                    break
                idx = node_indices[node_id]
                if idx < len(numa_groups[node_id]):
                    selected.append(numa_groups[node_id][idx])
                    node_indices[node_id] += 1
        
        return selected
    
    def _select_smt_pairs(self, available: List[int], count: int) -> List[int]:
        """Select complete SMT pairs."""
        selected = []
        selected_set = set()
        
        for core_id in available:
            if core_id in selected_set:
                continue
            if len(selected) >= count:
                break
            
            selected.append(core_id)
            selected_set.add(core_id)
            
            # Add SMT sibling if available
            if core_id in self.smt_siblings:
                sibling = self.smt_siblings[core_id]
                if sibling in available and sibling not in selected_set:
                    selected.append(sibling)
                    selected_set.add(sibling)
        
        return selected[:count]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export topology as dictionary."""
        return {
            "num_cores": self.num_cores,
            "num_physical_cores": self.num_physical_cores,
            "num_sockets": self.num_sockets,
            "num_numa_nodes": self.num_numa_nodes,
            "has_smt": self.has_smt,
            "cores": {str(k): asdict(v) for k, v in self.cores.items()},
            "numa_nodes": {str(k): v.to_dict() for k, v in self.numa_nodes.items()},
            "sockets": {str(k): v for k, v in self.sockets.items()},
            "isolated_cores": list(self._isolated_cores),
        }
    
    def save(self, path: Path) -> None:
        """Save topology to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def pin_to_cores(cores: List[int], pid: Optional[int] = None) -> bool:
    """
    Pin a process to specific CPU cores.
    
    Args:
        cores: List of CPU core IDs
        pid: Process ID (default: current process)
    
    Returns:
        True if successful
    """
    if pid is None:
        pid = os.getpid()
    
    try:
        os.sched_setaffinity(pid, set(cores))
        return True
    except (PermissionError, OSError):
        return False


def isolate_cores(cores: List[int], use_cgroup: bool = True) -> bool:
    """
    Isolate cores for exclusive use.
    
    This is a best-effort operation that:
    1. Uses cgroup cpuset if available
    2. Falls back to CPU affinity
    
    Args:
        cores: List of CPU core IDs
        use_cgroup: Whether to use cgroup isolation
    
    Returns:
        True if isolation was successful
    """
    if use_cgroup:
        # Try cgroup isolation
        cgroup_path = Path("/sys/fs/cgroup/aaco_isolated")
        try:
            cgroup_path.mkdir(parents=True, exist_ok=True)
            
            # Set cpuset
            cpuset_file = cgroup_path / "cpuset.cpus"
            if cpuset_file.exists():
                cores_str = ",".join(str(c) for c in cores)
                cpuset_file.write_text(cores_str)
            
            # Move current process
            procs_file = cgroup_path / "cgroup.procs"
            if procs_file.exists():
                procs_file.write_text(str(os.getpid()))
            
            return True
        except (PermissionError, OSError):
            pass
    
    # Fallback to affinity
    return pin_to_cores(cores)


def get_topology() -> CPUTopology:
    """Get CPU topology singleton."""
    if not hasattr(get_topology, '_instance'):
        get_topology._instance = CPUTopology()
    return get_topology._instance
