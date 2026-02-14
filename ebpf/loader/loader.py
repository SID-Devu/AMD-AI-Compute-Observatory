"""
AACO eBPF Program Loader

Loads and manages eBPF programs for low-level GPU/system profiling.
"""

import os
import sys
import ctypes
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import struct


class ProgramType(Enum):
    """eBPF program types."""
    KPROBE = auto()
    TRACEPOINT = auto()
    RAW_TRACEPOINT = auto()
    PERF_EVENT = auto()


@dataclass
class EBPFProgram:
    """Represents a loaded eBPF program."""
    
    name: str
    prog_type: ProgramType
    fd: int = -1
    attached: bool = False
    attach_point: str = ""


@dataclass
class EBPFMap:
    """Represents an eBPF map."""
    
    name: str
    fd: int = -1
    key_size: int = 0
    value_size: int = 0
    max_entries: int = 0


class EBPFLoader:
    """
    Loads and manages eBPF programs for AACO profiling.
    
    Supports:
    - Loading compiled eBPF programs (.o files)
    - Attaching to kprobes, tracepoints
    - Reading from eBPF maps
    - Perf event ring buffer
    """
    
    def __init__(self):
        self.programs: Dict[str, EBPFProgram] = {}
        self.maps: Dict[str, EBPFMap] = {}
        self._libbpf = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the eBPF subsystem.
        
        Returns:
            True if initialization successful
        """
        # Check if running on Linux
        if sys.platform != "linux":
            print("Warning: eBPF only supported on Linux")
            return False
        
        # Check for root permissions
        if os.geteuid() != 0:
            print("Warning: eBPF requires root permissions")
            return False
        
        # Try to load libbpf
        try:
            self._libbpf = ctypes.CDLL("libbpf.so.1", mode=ctypes.RTLD_GLOBAL)
            self._initialized = True
            return True
        except OSError:
            try:
                self._libbpf = ctypes.CDLL("libbpf.so.0", mode=ctypes.RTLD_GLOBAL)
                self._initialized = True
                return True
            except OSError:
                print("Warning: libbpf not found")
                return False
    
    def load_program(
        self,
        path: str,
        prog_type: ProgramType = ProgramType.KPROBE,
    ) -> Optional[EBPFProgram]:
        """
        Load an eBPF program from object file.
        
        Args:
            path: Path to compiled .o file
            prog_type: Type of eBPF program
            
        Returns:
            Loaded program or None
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        if not os.path.exists(path):
            print(f"Error: Program file not found: {path}")
            return None
        
        name = os.path.splitext(os.path.basename(path))[0]
        
        # Placeholder for actual loading logic
        # In real implementation, would use libbpf APIs
        prog = EBPFProgram(
            name=name,
            prog_type=prog_type,
            fd=-1,  # Would be set by bpf() syscall
        )
        
        self.programs[name] = prog
        return prog
    
    def attach_kprobe(
        self,
        prog: EBPFProgram,
        function: str,
        is_return: bool = False,
    ) -> bool:
        """
        Attach program to a kernel function (kprobe).
        
        Args:
            prog: Program to attach
            function: Kernel function name
            is_return: If true, attach to function return (kretprobe)
            
        Returns:
            True if successful
        """
        if prog.attached:
            return True
        
        attach_point = f"{'r' if is_return else ''}:{function}"
        prog.attach_point = attach_point
        prog.attached = True  # In real impl, would actually attach
        
        return True
    
    def attach_tracepoint(
        self,
        prog: EBPFProgram,
        category: str,
        name: str,
    ) -> bool:
        """
        Attach program to a tracepoint.
        
        Args:
            prog: Program to attach
            category: Tracepoint category (e.g., "sched")
            name: Tracepoint name (e.g., "sched_switch")
            
        Returns:
            True if successful
        """
        if prog.attached:
            return True
        
        attach_point = f"{category}:{name}"
        prog.attach_point = attach_point
        prog.attached = True
        
        return True
    
    def create_map(
        self,
        name: str,
        key_size: int,
        value_size: int,
        max_entries: int,
    ) -> Optional[EBPFMap]:
        """
        Create an eBPF map.
        
        Args:
            name: Map name
            key_size: Size of key in bytes
            value_size: Size of value in bytes
            max_entries: Maximum entries
            
        Returns:
            Created map or None
        """
        ebpf_map = EBPFMap(
            name=name,
            fd=-1,  # Would be set by bpf() syscall
            key_size=key_size,
            value_size=value_size,
            max_entries=max_entries,
        )
        
        self.maps[name] = ebpf_map
        return ebpf_map
    
    def read_map(
        self,
        map_obj: EBPFMap,
        key: bytes,
    ) -> Optional[bytes]:
        """
        Read value from eBPF map.
        
        Args:
            map_obj: Map to read from
            key: Key as bytes
            
        Returns:
            Value as bytes or None
        """
        if map_obj.fd < 0:
            return None
        
        # Placeholder - would use BPF_MAP_LOOKUP_ELEM
        return None
    
    def update_map(
        self,
        map_obj: EBPFMap,
        key: bytes,
        value: bytes,
    ) -> bool:
        """
        Update value in eBPF map.
        
        Args:
            map_obj: Map to update
            key: Key as bytes
            value: Value as bytes
            
        Returns:
            True if successful
        """
        if map_obj.fd < 0:
            return False
        
        # Placeholder - would use BPF_MAP_UPDATE_ELEM
        return True
    
    def detach_all(self) -> None:
        """Detach all attached programs."""
        for prog in self.programs.values():
            if prog.attached:
                prog.attached = False
                prog.attach_point = ""
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        self.detach_all()
        self.programs.clear()
        self.maps.clear()


# Convenience functions
def load_aaco_programs(loader: EBPFLoader) -> bool:
    """Load all AACO eBPF programs."""
    prog_dir = os.path.dirname(__file__)
    programs_dir = os.path.join(prog_dir, "..", "programs")
    
    if not os.path.exists(programs_dir):
        return False
    
    for filename in os.listdir(programs_dir):
        if filename.endswith(".o"):
            path = os.path.join(programs_dir, filename)
            loader.load_program(path)
    
    return True
