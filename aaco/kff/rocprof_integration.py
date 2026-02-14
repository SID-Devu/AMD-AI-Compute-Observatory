# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Dual-Mode Rocprof Integration

Integrates both rocprofv1 and rocprofv2 (rocprof-sys) for:
- Activity tracing (async GPU events)
- Counter collection (PMC sampling)
- Kernel duration extraction
- Hardware counter analysis
"""

import json
import subprocess
import tempfile
import os
import csv
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RocprofMode(Enum):
    """Rocprof operation modes."""
    TRACE = "trace"           # Activity tracing only
    COUNTERS = "counters"     # PMC counter collection
    COMBINED = "combined"     # Both trace and counters


class RocprofVersion(Enum):
    """Rocprof version variants."""
    V1 = "rocprof"
    V2 = "rocprofv2"
    SYS = "rocprof-sys"


@dataclass
class KernelTrace:
    """Single GPU kernel trace entry."""
    kernel_name: str = ""
    grid_size: Tuple[int, int, int] = (0, 0, 0)
    workgroup_size: Tuple[int, int, int] = (0, 0, 0)
    start_ns: int = 0
    end_ns: int = 0
    duration_ns: int = 0
    queue_id: int = 0
    device_id: int = 0
    
    # Counter values (when available)
    counters: Dict[str, int] = field(default_factory=dict)


@dataclass
class CounterConfig:
    """Configuration for counter collection."""
    # Memory counters
    memory_counters: List[str] = field(default_factory=lambda: [
        "FETCH_SIZE",        # Bytes fetched
        "WRITE_SIZE",        # Bytes written
        "L2CacheHit",        # L2 cache hits
        "MemUnitBusy",       # Memory unit utilization
    ])
    
    # Compute counters
    compute_counters: List[str] = field(default_factory=lambda: [
        "VALUInsts",         # Vector ALU instructions
        "SALUInsts",         # Scalar ALU instructions
        "VFetchInsts",       # Vector fetch instructions
        "LDSInsts",          # LDS instructions
    ])
    
    # Occupancy counters
    occupancy_counters: List[str] = field(default_factory=lambda: [
        "Wavefronts",        # Wavefronts launched
        "VALUUtilization",   # VALU busy percentage
        "MemUnitStalled",    # Memory stall cycles
    ])
    
    def get_all_counters(self) -> List[str]:
        """Get all configured counters."""
        return (
            self.memory_counters + 
            self.compute_counters + 
            self.occupancy_counters
        )


@dataclass
class RocprofResult:
    """Result of rocprof collection."""
    success: bool = False
    error_message: str = ""
    
    # Traces
    kernel_traces: List[KernelTrace] = field(default_factory=list)
    
    # Aggregated statistics
    total_kernels: int = 0
    total_gpu_time_ns: int = 0
    unique_kernel_names: int = 0
    
    # Per-kernel aggregates
    kernel_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Raw output files
    trace_file: str = ""
    counter_file: str = ""


class DualModeRocprof:
    """
    Dual-mode rocprof integration supporting both versions.
    
    Features:
    - Automatic version detection
    - Activity tracing
    - Counter collection
    - Kernel extraction and aggregation
    """
    
    # Counter groups for gfx11 architecture
    GFX11_COUNTER_GROUPS = {
        'memory': [
            'FETCH_SIZE', 'WRITE_SIZE', 
            'TA_TA_BUSY', 'TCP_TCC_READ_REQ_sum',
        ],
        'compute': [
            'SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_SALU',
            'SQ_INSTS_LDS', 'SQ_INSTS_SMEM',
        ],
        'cache': [
            'TCC_HIT_sum', 'TCC_MISS_sum',
            'TCC_EA_RDREQ_32B_sum', 'TCC_EA_WRREQ_sum',
        ],
    }
    
    def __init__(
        self,
        rocprof_path: Optional[str] = None,
        preferred_version: RocprofVersion = RocprofVersion.V1,
    ):
        """
        Initialize dual-mode rocprof.
        
        Args:
            rocprof_path: Custom path to rocprof binary
            preferred_version: Preferred rocprof version
        """
        self._rocprof_path = rocprof_path or self._detect_rocprof()
        self._version = preferred_version
        self._counter_config = CounterConfig()
        self._temp_dir = None
    
    def _detect_rocprof(self) -> str:
        """Detect available rocprof installation."""
        candidates = [
            "rocprof",
            "rocprofv2",
            "rocprof-sys",
            "/opt/rocm/bin/rocprof",
            "/opt/rocm/bin/rocprofv2",
        ]
        
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    logger.info(f"Found rocprof: {candidate}")
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        logger.warning("No rocprof found, using stub mode")
        return "rocprof"
    
    def collect_trace(
        self,
        command: List[str],
        output_dir: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> RocprofResult:
        """
        Collect GPU activity trace.
        
        Args:
            command: Command to profile [executable, args...]
            output_dir: Directory for output files
            timeout_seconds: Timeout for profiling
            
        Returns:
            RocprofResult with kernel traces
        """
        result = RocprofResult()
        
        self._temp_dir = output_dir or tempfile.mkdtemp(prefix="aaco_rocprof_")
        output_file = os.path.join(self._temp_dir, "trace.csv")
        
        # Build rocprof command for tracing
        rocprof_cmd = self._build_trace_command(command, output_file)
        
        try:
            logger.info(f"Running rocprof trace: {' '.join(rocprof_cmd)}")
            
            proc = subprocess.run(
                rocprof_cmd,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=self._temp_dir,
            )
            
            if proc.returncode != 0:
                result.error_message = proc.stderr.decode()[:500]
                logger.error(f"Rocprof failed: {result.error_message}")
                return result
            
            # Parse trace output
            result = self._parse_trace_output(output_file)
            result.trace_file = output_file
            result.success = True
            
        except subprocess.TimeoutExpired:
            result.error_message = f"Timeout after {timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)
        
        return result
    
    def collect_counters(
        self,
        command: List[str],
        counter_groups: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> RocprofResult:
        """
        Collect hardware counter values.
        
        Args:
            command: Command to profile
            counter_groups: Counter groups to collect ('memory', 'compute', 'cache')
            output_dir: Directory for output files
            timeout_seconds: Timeout for profiling
            
        Returns:
            RocprofResult with counter values
        """
        result = RocprofResult()
        
        self._temp_dir = output_dir or tempfile.mkdtemp(prefix="aaco_rocprof_")
        output_file = os.path.join(self._temp_dir, "counters.csv")
        input_file = os.path.join(self._temp_dir, "input.txt")
        
        # Generate counter input file
        counters = self._get_counters_for_groups(counter_groups)
        self._write_counter_input(input_file, counters)
        
        # Build rocprof command for counters
        rocprof_cmd = self._build_counter_command(command, input_file, output_file)
        
        try:
            logger.info(f"Running rocprof counters: {' '.join(rocprof_cmd)}")
            
            proc = subprocess.run(
                rocprof_cmd,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=self._temp_dir,
            )
            
            if proc.returncode != 0:
                result.error_message = proc.stderr.decode()[:500]
                return result
            
            # Parse counter output
            result = self._parse_counter_output(output_file)
            result.counter_file = output_file
            result.success = True
            
        except subprocess.TimeoutExpired:
            result.error_message = f"Timeout after {timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)
        
        return result
    
    def _build_trace_command(
        self,
        command: List[str],
        output_file: str,
    ) -> List[str]:
        """Build rocprof trace command."""
        if self._version == RocprofVersion.V2:
            return [
                self._rocprof_path,
                "-d", os.path.dirname(output_file),
                "-o", output_file,
                "--hip-trace",
                "--hsa-trace",
                "--kernel-trace",
            ] + command
        else:
            # rocprofv1
            return [
                self._rocprof_path,
                "-o", output_file,
                "--hip-trace",
                "--hsa-trace",
                "--timestamp", "on",
            ] + command
    
    def _build_counter_command(
        self,
        command: List[str],
        input_file: str,
        output_file: str,
    ) -> List[str]:
        """Build rocprof counter collection command."""
        return [
            self._rocprof_path,
            "-i", input_file,
            "-o", output_file,
            "--timestamp", "on",
        ] + command
    
    def _get_counters_for_groups(
        self,
        groups: Optional[List[str]] = None,
    ) -> List[str]:
        """Get counter list for specified groups."""
        groups = groups or ['memory', 'compute']
        
        counters = []
        for group in groups:
            if group in self.GFX11_COUNTER_GROUPS:
                counters.extend(self.GFX11_COUNTER_GROUPS[group])
        
        return counters
    
    def _write_counter_input(
        self,
        filepath: str,
        counters: List[str],
    ) -> None:
        """Write counter input file for rocprof."""
        # Split into groups of 4 (hardware limit)
        with open(filepath, 'w') as f:
            for i in range(0, len(counters), 4):
                group = counters[i:i+4]
                f.write(f"pmc: {' '.join(group)}\n")
    
    def _parse_trace_output(self, filepath: str) -> RocprofResult:
        """Parse rocprof trace CSV output."""
        result = RocprofResult()
        
        if not os.path.exists(filepath):
            result.error_message = f"Trace file not found: {filepath}"
            return result
        
        kernel_map = {}
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Extract kernel info
                    kernel_name = row.get('Name', row.get('KernelName', ''))
                    if not kernel_name:
                        continue
                    
                    trace = KernelTrace(
                        kernel_name=kernel_name,
                        start_ns=int(row.get('Start', row.get('BeginNs', 0))),
                        end_ns=int(row.get('End', row.get('EndNs', 0))),
                        duration_ns=int(row.get('DurationNs', 0)),
                        device_id=int(row.get('DeviceId', row.get('gpu-id', 0))),
                    )
                    
                    # Calculate duration if not provided
                    if trace.duration_ns == 0 and trace.end_ns > trace.start_ns:
                        trace.duration_ns = trace.end_ns - trace.start_ns
                    
                    # Parse grid size
                    grid_str = row.get('grd', row.get('WorkGroupSize', ''))
                    if grid_str:
                        trace.grid_size = self._parse_dim_string(grid_str)
                    
                    result.kernel_traces.append(trace)
                    
                    # Aggregate by kernel name
                    if kernel_name not in kernel_map:
                        kernel_map[kernel_name] = {
                            'count': 0,
                            'durations': [],
                            'total_ns': 0,
                        }
                    kernel_map[kernel_name]['count'] += 1
                    kernel_map[kernel_name]['durations'].append(trace.duration_ns)
                    kernel_map[kernel_name]['total_ns'] += trace.duration_ns
            
            result.total_kernels = len(result.kernel_traces)
            result.total_gpu_time_ns = sum(t.duration_ns for t in result.kernel_traces)
            result.unique_kernel_names = len(kernel_map)
            result.kernel_stats = kernel_map
            
        except Exception as e:
            result.error_message = f"Parse error: {e}"
        
        return result
    
    def _parse_counter_output(self, filepath: str) -> RocprofResult:
        """Parse rocprof counter CSV output."""
        result = RocprofResult()
        
        if not os.path.exists(filepath):
            result.error_message = f"Counter file not found: {filepath}"
            return result
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                
                for row in reader:
                    kernel_name = row.get('Name', row.get('KernelName', ''))
                    if not kernel_name:
                        continue
                    
                    trace = KernelTrace(
                        kernel_name=kernel_name,
                        duration_ns=int(float(row.get('DurationNs', 0))),
                    )
                    
                    # Extract all counter values
                    for header in headers:
                        if header not in ['Name', 'KernelName', 'DurationNs', 'Start', 'End']:
                            try:
                                trace.counters[header] = int(float(row.get(header, 0)))
                            except (ValueError, TypeError):
                                pass
                    
                    result.kernel_traces.append(trace)
            
            result.total_kernels = len(result.kernel_traces)
            
        except Exception as e:
            result.error_message = f"Parse error: {e}"
        
        return result
    
    def _parse_dim_string(self, dim_str: str) -> Tuple[int, int, int]:
        """Parse dimension string like '256x1x1' or '(256, 1, 1)'."""
        match = re.findall(r'\d+', dim_str)
        if len(match) >= 3:
            return (int(match[0]), int(match[1]), int(match[2]))
        return (0, 0, 0)
    
    def get_kernel_durations(self) -> Dict[str, List[int]]:
        """Get durations grouped by kernel name from last collection."""
        if not hasattr(self, '_last_result') or not self._last_result:
            return {}
        
        durations = {}
        for trace in self._last_result.kernel_traces:
            if trace.kernel_name not in durations:
                durations[trace.kernel_name] = []
            durations[trace.kernel_name].append(trace.duration_ns)
        
        return durations


def create_rocprof_collector(
    mode: RocprofMode = RocprofMode.COMBINED,
    preferred_version: RocprofVersion = RocprofVersion.V1,
) -> DualModeRocprof:
    """
    Factory function to create rocprof collector.
    
    Args:
        mode: Collection mode (trace, counters, or combined)
        preferred_version: Preferred rocprof version
        
    Returns:
        Configured DualModeRocprof instance
    """
    return DualModeRocprof(preferred_version=preferred_version)
