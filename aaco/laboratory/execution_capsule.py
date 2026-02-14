# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Execution Capsule

Reproducible execution environments that capture complete system state.
Produces capsule_manifest.json and noise_signature.json for every run.
"""

import json
import hashlib
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class HardwareFingerprint:
    """Hardware configuration fingerprint."""
    cpu_model: str = ""
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    memory_total_gb: float = 0.0
    numa_nodes: int = 1
    
    # GPU info
    gpu_count: int = 0
    gpu_models: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    gpu_compute_units: List[int] = field(default_factory=list)
    
    # Storage
    storage_type: str = ""  # SSD, NVMe, HDD
    
    def fingerprint_hash(self) -> str:
        """Generate hash of hardware configuration."""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class SoftwareFingerprint:
    """Software configuration fingerprint."""
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""
    
    # ROCm
    rocm_version: str = ""
    hip_version: str = ""
    migraphx_version: str = ""
    
    # Python
    python_version: str = ""
    onnxruntime_version: str = ""
    
    # Driver
    gpu_driver_version: str = ""
    
    # Git info
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    
    def fingerprint_hash(self) -> str:
        """Generate hash of software configuration."""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class RuntimeState:
    """Runtime state at measurement time."""
    # System state
    load_average: List[float] = field(default_factory=list)
    memory_used_pct: float = 0.0
    swap_used_pct: float = 0.0
    
    # CPU state
    cpu_governor: str = ""
    cpu_freq_current_mhz: float = 0.0
    cpu_temp_celsius: float = 0.0
    
    # GPU state
    gpu_clocks_mhz: List[float] = field(default_factory=list)
    gpu_memory_used_pct: List[float] = field(default_factory=list)
    gpu_temps_celsius: List[float] = field(default_factory=list)
    gpu_power_watts: List[float] = field(default_factory=list)
    gpu_perf_level: List[str] = field(default_factory=list)
    
    # Process state
    process_nice: int = 0
    process_affinity: List[int] = field(default_factory=list)
    process_memory_mb: float = 0.0


@dataclass
class CapsuleManifest:
    """Complete execution capsule manifest."""
    # Identity
    capsule_id: str = ""
    session_id: str = ""
    created_at: str = ""
    
    # Fingerprints
    hardware: HardwareFingerprint = field(default_factory=HardwareFingerprint)
    software: SoftwareFingerprint = field(default_factory=SoftwareFingerprint)
    runtime_start: RuntimeState = field(default_factory=RuntimeState)
    runtime_end: RuntimeState = field(default_factory=RuntimeState)
    
    # Isolation settings
    isolation_enabled: bool = False
    isolated_cores: List[int] = field(default_factory=list)
    cgroup_name: str = ""
    
    # Measurement config
    warmup_iterations: int = 0
    measure_iterations: int = 0
    batch_size: int = 1
    
    # Reproducibility contract
    deterministic_mode: bool = False
    random_seed: Optional[int] = None
    
    # Composite fingerprint
    environment_hash: str = ""
    
    def compute_environment_hash(self) -> str:
        """Compute combined environment hash."""
        combined = (
            self.hardware.fingerprint_hash() +
            self.software.fingerprint_hash()
        )
        return hashlib.sha256(combined.encode()).hexdigest()[:32]


class ExecutionCapsule:
    """
    Manages reproducible execution environments.
    
    Captures complete system state before and after measurements
    to ensure reproducibility and detect environment drift.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize execution capsule."""
        self.session_id = session_id or str(uuid.uuid4())
        self.capsule_id = str(uuid.uuid4())
        self.manifest = CapsuleManifest(
            capsule_id=self.capsule_id,
            session_id=self.session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._start_time = None
        self._end_time = None
    
    def capture_pre_execution_state(self) -> CapsuleManifest:
        """Capture complete system state before execution."""
        self._start_time = time.time()
        
        # Capture hardware fingerprint
        self.manifest.hardware = self._capture_hardware()
        
        # Capture software fingerprint
        self.manifest.software = self._capture_software()
        
        # Capture runtime state
        self.manifest.runtime_start = self._capture_runtime_state()
        
        # Compute environment hash
        self.manifest.environment_hash = self.manifest.compute_environment_hash()
        
        logger.info(f"Capsule {self.capsule_id[:8]} pre-execution state captured")
        return self.manifest
    
    def capture_post_execution_state(self) -> CapsuleManifest:
        """Capture system state after execution."""
        self._end_time = time.time()
        
        # Capture post-execution runtime state
        self.manifest.runtime_end = self._capture_runtime_state()
        
        logger.info(f"Capsule {self.capsule_id[:8]} post-execution state captured")
        return self.manifest
    
    def _capture_hardware(self) -> HardwareFingerprint:
        """Capture hardware configuration."""
        hw = HardwareFingerprint()
        
        # CPU info
        hw.cpu_model = platform.processor() or "Unknown"
        hw.cpu_count = platform.os.cpu_count() or 0
        
        try:
            # Get CPU frequency
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        hw.cpu_model = line.split(':')[1].strip()
                    elif 'cpu MHz' in line:
                        hw.cpu_freq_mhz = float(line.split(':')[1].strip())
                        break
        except Exception:
            pass
        
        # Memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        hw.memory_total_gb = kb / 1024 / 1024
                        break
        except Exception:
            pass
        
        # GPU info via rocm-smi
        hw.gpu_count, hw.gpu_models, hw.gpu_memory_gb = self._get_gpu_info()
        
        return hw
    
    def _capture_software(self) -> SoftwareFingerprint:
        """Capture software configuration."""
        sw = SoftwareFingerprint()
        
        # OS info
        sw.os_name = platform.system()
        sw.os_version = platform.release()
        sw.kernel_version = platform.version()
        sw.python_version = platform.python_version()
        
        # ROCm version
        sw.rocm_version = self._get_rocm_version()
        
        # Git info
        sw.git_commit, sw.git_branch, sw.git_dirty = self._get_git_info()
        
        # Package versions
        try:
            import onnxruntime
            sw.onnxruntime_version = onnxruntime.__version__
        except ImportError:
            pass
        
        return sw
    
    def _capture_runtime_state(self) -> RuntimeState:
        """Capture current runtime state."""
        state = RuntimeState()
        
        # Load average
        try:
            with open('/proc/loadavg', 'r') as f:
                parts = f.read().split()
                state.load_average = [float(parts[0]), float(parts[1]), float(parts[2])]
        except Exception:
            pass
        
        # Memory usage
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_total = mem_available = 0
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1])
                    elif line.startswith('MemAvailable:'):
                        mem_available = int(line.split()[1])
                if mem_total > 0:
                    state.memory_used_pct = 100.0 * (mem_total - mem_available) / mem_total
        except Exception:
            pass
        
        # CPU governor
        try:
            gov_path = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
            if gov_path.exists():
                state.cpu_governor = gov_path.read_text().strip()
        except Exception:
            pass
        
        # GPU state via rocm-smi
        gpu_state = self._get_gpu_runtime_state()
        state.gpu_clocks_mhz = gpu_state.get('clocks', [])
        state.gpu_temps_celsius = gpu_state.get('temps', [])
        state.gpu_power_watts = gpu_state.get('power', [])
        state.gpu_memory_used_pct = gpu_state.get('mem_used_pct', [])
        state.gpu_perf_level = gpu_state.get('perf_level', [])
        
        # Process state
        import os
        state.process_nice = os.nice(0)
        if hasattr(os, 'sched_getaffinity'):
            state.process_affinity = list(os.sched_getaffinity(0))
        
        return state
    
    def _get_gpu_info(self) -> tuple:
        """Get GPU information via rocm-smi."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname', '--showmeminfo', 'vram', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                gpu_count = len([k for k in data.keys() if k.startswith('card')])
                models = []
                memory_gb = []
                
                for i in range(gpu_count):
                    card_key = f'card{i}'
                    if card_key in data:
                        card = data[card_key]
                        models.append(card.get('Card series', 'Unknown'))
                        vram = card.get('VRAM Total Memory (B)', 0)
                        memory_gb.append(int(vram) / 1024**3 if vram else 0)
                
                return gpu_count, models, memory_gb
        except Exception:
            pass
        
        return 0, [], []
    
    def _get_gpu_runtime_state(self) -> Dict[str, List]:
        """Get current GPU runtime state."""
        state = {'clocks': [], 'temps': [], 'power': [], 'mem_used_pct': [], 'perf_level': []}
        
        try:
            result = subprocess.run(
                ['rocm-smi', '--showclocks', '--showtemp', '--showpower', 
                 '--showmeminfo', 'vram', '--showperflevel', '--json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for key in sorted(data.keys()):
                    if key.startswith('card'):
                        card = data[key]
                        # Parse clock
                        sclk = card.get('GPU Clock Level', '0 MHz')
                        if isinstance(sclk, str):
                            state['clocks'].append(float(sclk.split()[0]) if sclk else 0)
                        
                        # Parse temp
                        temp = card.get('Temperature (Sensor edge) (C)', 0)
                        state['temps'].append(float(temp) if temp else 0)
                        
                        # Parse power
                        power = card.get('Average Graphics Package Power (W)', 0)
                        state['power'].append(float(power) if power else 0)
                        
                        # Parse perf level
                        perf = card.get('Performance Level', 'unknown')
                        state['perf_level'].append(perf)
        except Exception:
            pass
        
        return state
    
    def _get_rocm_version(self) -> str:
        """Get ROCm version."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showversion'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'ROCm' in line:
                        return line.strip()
        except Exception:
            pass
        
        # Try reading from file
        try:
            version_file = Path('/opt/rocm/.info/version')
            if version_file.exists():
                return version_file.read_text().strip()
        except Exception:
            pass
        
        return "unknown"
    
    def _get_git_info(self) -> tuple:
        """Get git repository information."""
        try:
            commit = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            
            branch = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            
            dirty = subprocess.run(
                ['git', 'diff', '--quiet'],
                capture_output=True, timeout=5
            ).returncode != 0
            
            return commit[:12] if commit else "", branch, dirty
        except Exception:
            return "", "", False
    
    def detect_environment_drift(
        self,
        reference_manifest: CapsuleManifest
    ) -> Dict[str, Any]:
        """
        Detect drift from a reference environment.
        
        Args:
            reference_manifest: Previously captured manifest to compare against
            
        Returns:
            Dict with drift analysis
        """
        drift = {
            'has_drift': False,
            'hardware_match': True,
            'software_match': True,
            'differences': [],
        }
        
        # Compare hardware
        if (self.manifest.hardware.fingerprint_hash() != 
            reference_manifest.hardware.fingerprint_hash()):
            drift['hardware_match'] = False
            drift['has_drift'] = True
            drift['differences'].append({
                'component': 'hardware',
                'current': self.manifest.hardware.fingerprint_hash(),
                'reference': reference_manifest.hardware.fingerprint_hash(),
            })
        
        # Compare software
        if (self.manifest.software.fingerprint_hash() != 
            reference_manifest.software.fingerprint_hash()):
            drift['software_match'] = False
            drift['has_drift'] = True
            
            # Detailed software diff
            if self.manifest.software.rocm_version != reference_manifest.software.rocm_version:
                drift['differences'].append({
                    'component': 'rocm_version',
                    'current': self.manifest.software.rocm_version,
                    'reference': reference_manifest.software.rocm_version,
                })
            
            if self.manifest.software.git_commit != reference_manifest.software.git_commit:
                drift['differences'].append({
                    'component': 'git_commit',
                    'current': self.manifest.software.git_commit,
                    'reference': reference_manifest.software.git_commit,
                })
        
        return drift
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save capsule manifest to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Save main manifest
        manifest_path = output_dir / 'capsule_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(asdict(self.manifest), f, indent=2)
        files['manifest'] = manifest_path
        
        # Save noise signature (if we have runtime states)
        if self.manifest.runtime_start and self.manifest.runtime_end:
            noise_path = output_dir / 'noise_signature.json'
            noise_data = {
                'pre_execution': {
                    'load_average': self.manifest.runtime_start.load_average,
                    'memory_used_pct': self.manifest.runtime_start.memory_used_pct,
                    'gpu_temps': self.manifest.runtime_start.gpu_temps_celsius,
                    'gpu_clocks': self.manifest.runtime_start.gpu_clocks_mhz,
                },
                'post_execution': {
                    'load_average': self.manifest.runtime_end.load_average,
                    'memory_used_pct': self.manifest.runtime_end.memory_used_pct,
                    'gpu_temps': self.manifest.runtime_end.gpu_temps_celsius,
                    'gpu_clocks': self.manifest.runtime_end.gpu_clocks_mhz,
                },
                'drift_detected': self._detect_runtime_drift(),
            }
            with open(noise_path, 'w') as f:
                json.dump(noise_data, f, indent=2)
            files['noise_signature'] = noise_path
        
        return files
    
    def _detect_runtime_drift(self) -> bool:
        """Detect if runtime state drifted during execution."""
        start = self.manifest.runtime_start
        end = self.manifest.runtime_end
        
        # Check for significant temp increase
        if start.gpu_temps_celsius and end.gpu_temps_celsius:
            temp_delta = max(
                abs(e - s) 
                for s, e in zip(start.gpu_temps_celsius, end.gpu_temps_celsius)
            )
            if temp_delta > 10:  # More than 10C increase
                return True
        
        # Check for clock throttling
        if start.gpu_clocks_mhz and end.gpu_clocks_mhz:
            for s, e in zip(start.gpu_clocks_mhz, end.gpu_clocks_mhz):
                if s > 0 and e < s * 0.9:  # More than 10% drop
                    return True
        
        return False
    
    @classmethod
    def load(cls, manifest_path: Path) -> 'ExecutionCapsule':
        """Load capsule from manifest file."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        capsule = cls(session_id=data.get('session_id'))
        capsule.capsule_id = data.get('capsule_id', capsule.capsule_id)
        
        # Reconstruct manifest
        capsule.manifest = CapsuleManifest(
            capsule_id=data.get('capsule_id', ''),
            session_id=data.get('session_id', ''),
            created_at=data.get('created_at', ''),
            environment_hash=data.get('environment_hash', ''),
        )
        
        if 'hardware' in data:
            capsule.manifest.hardware = HardwareFingerprint(**data['hardware'])
        if 'software' in data:
            capsule.manifest.software = SoftwareFingerprint(**data['software'])
        
        return capsule
