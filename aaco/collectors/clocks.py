"""
Clock and Governor Monitor
Reads and monitors CPU/GPU clock frequencies and governor settings.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from aaco.core.utils import read_proc_file, run_command

logger = logging.getLogger(__name__)


class ClockMonitor:
    """
    Monitor CPU and GPU clock frequencies and power management settings.
    """
    
    def __init__(self):
        self._cpu_freq_path = Path("/sys/devices/system/cpu")
        self._gpu_freq_path = Path("/sys/class/drm")
    
    def get_cpu_governor(self, cpu_id: int = 0) -> str:
        """Get CPU frequency governor."""
        path = self._cpu_freq_path / f"cpu{cpu_id}/cpufreq/scaling_governor"
        content = read_proc_file(str(path))
        return content.strip() if content else "unknown"
    
    def get_all_cpu_governors(self) -> Dict[int, str]:
        """Get governors for all CPUs."""
        governors = {}
        
        for cpu_dir in sorted(self._cpu_freq_path.glob("cpu[0-9]*")):
            try:
                cpu_id = int(cpu_dir.name[3:])
                governors[cpu_id] = self.get_cpu_governor(cpu_id)
            except (ValueError, OSError):
                continue
        
        return governors
    
    def get_cpu_frequency(self, cpu_id: int = 0) -> Dict[str, float]:
        """Get current, min, and max CPU frequency in MHz."""
        base = self._cpu_freq_path / f"cpu{cpu_id}/cpufreq"
        
        freq = {"current": 0.0, "min": 0.0, "max": 0.0}
        
        for key, filename in [
            ("current", "scaling_cur_freq"),
            ("min", "scaling_min_freq"),
            ("max", "scaling_max_freq"),
        ]:
            content = read_proc_file(str(base / filename))
            if content:
                try:
                    # Frequency in kHz, convert to MHz
                    freq[key] = float(content.strip()) / 1000
                except ValueError:
                    pass
        
        return freq
    
    def get_gpu_clock_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU clock information via sysfs or rocm-smi."""
        info = {
            "gfx_clock_mhz": 0.0,
            "mem_clock_mhz": 0.0,
            "gfx_clock_min": 0.0,
            "gfx_clock_max": 0.0,
        }
        
        # Try sysfs first
        card_path = self._gpu_freq_path / f"card{device_id}/device"
        
        # Current GFX clock
        content = read_proc_file(str(card_path / "pp_dpm_sclk"))
        if content:
            for line in content.split("\n"):
                if "*" in line:  # Current level marked with *
                    try:
                        match = line.split()
                        for part in match:
                            if "Mhz" in part or part.isdigit():
                                info["gfx_clock_mhz"] = float(part.replace("Mhz", ""))
                                break
                    except (ValueError, IndexError):
                        pass
        
        # Current MEM clock
        content = read_proc_file(str(card_path / "pp_dpm_mclk"))
        if content:
            for line in content.split("\n"):
                if "*" in line:
                    try:
                        match = line.split()
                        for part in match:
                            if "Mhz" in part or part.isdigit():
                                info["mem_clock_mhz"] = float(part.replace("Mhz", ""))
                                break
                    except (ValueError, IndexError):
                        pass
        
        # Fallback to rocm-smi
        if info["gfx_clock_mhz"] == 0:
            result = run_command(["rocm-smi", "-d", str(device_id), "--showclocks"])
            if result:
                for line in result.split("\n"):
                    if "sclk" in line.lower():
                        try:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if "mhz" in p.lower() and i > 0:
                                    info["gfx_clock_mhz"] = float(parts[i-1])
                                    break
                        except (ValueError, IndexError):
                            pass
        
        return info
    
    def get_gpu_power_profile(self, device_id: int = 0) -> str:
        """Get current GPU power profile."""
        card_path = self._gpu_freq_path / f"card{device_id}/device"
        
        content = read_proc_file(str(card_path / "power_dpm_force_performance_level"))
        if content:
            return content.strip()
        
        # Fallback to rocm-smi
        result = run_command(["rocm-smi", "-d", str(device_id), "--showperflevel"])
        if result:
            for line in result.split("\n"):
                if "performance" in line.lower() or "level" in line.lower():
                    return line.strip()
        
        return "unknown"
    
    def get_system_config(self) -> Dict[str, any]:
        """Get complete clock/power configuration snapshot."""
        config = {
            "cpu": {
                "governor": self.get_cpu_governor(0),
                "frequency": self.get_cpu_frequency(0),
                "all_governors": self.get_all_cpu_governors(),
            },
            "gpu": {
                "clocks": self.get_gpu_clock_info(0),
                "power_profile": self.get_gpu_power_profile(0),
            },
        }
        
        # Check if all CPU governors are the same
        governors = list(config["cpu"]["all_governors"].values())
        config["cpu"]["governors_uniform"] = len(set(governors)) <= 1
        
        return config
    
    def validate_performance_mode(self) -> Dict[str, bool]:
        """
        Validate that system is configured for consistent benchmarking.
        Returns dict of checks with pass/fail status.
        """
        checks = {
            "cpu_governor_performance": False,
            "gpu_power_profile_manual": False,
            "governors_uniform": False,
        }
        
        # CPU governor should be "performance"
        governor = self.get_cpu_governor(0)
        checks["cpu_governor_performance"] = governor == "performance"
        
        # All CPUs should have same governor
        all_governors = self.get_all_cpu_governors()
        checks["governors_uniform"] = len(set(all_governors.values())) <= 1
        
        # GPU power profile should be manual/high for consistent clocks
        power_profile = self.get_gpu_power_profile(0)
        checks["gpu_power_profile_manual"] = power_profile in ["manual", "high", "profile_peak"]
        
        return checks
