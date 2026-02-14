"""
ROCm-SMI GPU Telemetry Sampler
Collects GPU metrics (clocks, power, temperature, VRAM, utilization) from rocm-smi.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from aaco.core.schema import GPUEvent
from aaco.core.utils import get_monotonic_ns, run_command

logger = logging.getLogger(__name__)


@dataclass
class GPUSample:
    """Single GPU telemetry sample."""

    t_ns: int
    device_id: int
    gfx_clock_mhz: float
    mem_clock_mhz: float
    power_w: float
    temp_c: float
    vram_used_mb: float
    vram_total_mb: float
    gpu_util_pct: float
    mem_util_pct: float


class ROCmSMISampler:
    """
    Samples GPU telemetry from rocm-smi at configurable intervals.
    Runs in a background thread during workload execution.
    """

    def __init__(
        self,
        interval_ms: int = 500,
        device_id: int = 0,
        t0_ns: int = 0,
    ):
        self.interval_ms = interval_ms
        self.interval_s = interval_ms / 1000.0
        self.device_id = device_id
        self.t0_ns = t0_ns

        self.samples: List[GPUSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Check if rocm-smi is available
        self._available = self._check_rocm_smi()

    def _check_rocm_smi(self) -> bool:
        """Check if rocm-smi is available."""
        result = run_command(["rocm-smi", "--version"])
        return result is not None

    def _get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information from rocm-smi."""
        if not self._available:
            return None

        result = run_command(
            ["rocm-smi", "-d", str(self.device_id), "--showproductname", "--showmeminfo", "vram"]
        )
        if not result:
            return None

        info = {
            "name": "AMD GPU",
            "vram_total_mb": 0,
        }

        for line in result.split("\n"):
            if "Card series" in line or "GPU" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    info["name"] = parts[1].strip()
            elif "Total Memory" in line or "vram Total" in line.lower():
                match = re.search(r"(\d+)", line)
                if match:
                    # Value might be in bytes or MB depending on rocm-smi version
                    val = int(match.group(1))
                    if val > 1000000:  # Likely bytes
                        info["vram_total_mb"] = val // (1024 * 1024)
                    else:
                        info["vram_total_mb"] = val

        return info

    @property
    def available(self) -> bool:
        """Check if ROCm-SMI sampling is available."""
        return self._available

    def start(self) -> None:
        """Start background sampling thread."""
        if not self._available:
            logger.warning("rocm-smi not available, GPU sampling disabled")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.debug("GPU sampler started")

    def stop(self) -> List[GPUSample]:
        """Stop sampling and return collected samples."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.debug(f"GPU sampler stopped, collected {len(self.samples)} samples")
        return self.samples

    def _sample_loop(self) -> None:
        """Main sampling loop."""
        while self._running:
            try:
                sample = self._collect_sample()
                if sample:
                    self.samples.append(sample)
            except Exception as e:
                logger.debug(f"GPU sample collection error: {e}")

            time.sleep(self.interval_s)

    def _collect_sample(self) -> Optional[GPUSample]:
        """Collect a single sample from rocm-smi."""
        t_ns = get_monotonic_ns() - self.t0_ns

        # Get all metrics in one call for efficiency
        result = run_command(
            [
                "rocm-smi",
                "-d",
                str(self.device_id),
                "--showclocks",
                "--showpower",
                "--showtemp",
                "--showmeminfo",
                "vram",
                "--showuse",
                "--json",
            ]
        )

        if not result:
            # Fallback to individual calls
            return self._collect_sample_fallback(t_ns)

        # Try JSON parsing first
        try:
            import json

            data = json.loads(result)
            gpu_key = f"card{self.device_id}"

            if gpu_key in data:
                gpu_data = data[gpu_key]
                return self._parse_json_sample(t_ns, gpu_data)
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback to text parsing
        return self._collect_sample_fallback(t_ns)

    def _parse_json_sample(self, t_ns: int, gpu_data: Dict) -> GPUSample:
        """Parse GPU sample from JSON output."""
        return GPUSample(
            t_ns=t_ns,
            device_id=self.device_id,
            gfx_clock_mhz=float(gpu_data.get("GFX Clock Level", "0").split("(")[0].strip() or 0),
            mem_clock_mhz=float(gpu_data.get("MEM Clock Level", "0").split("(")[0].strip() or 0),
            power_w=float(gpu_data.get("Average Graphics Package Power (W)", 0)),
            temp_c=float(gpu_data.get("Temperature (Sensor edge) (C)", 0)),
            vram_used_mb=float(gpu_data.get("VRAM Total Used Memory (B)", 0)) / (1024**2),
            vram_total_mb=float(gpu_data.get("VRAM Total Memory (B)", 0)) / (1024**2),
            gpu_util_pct=float(gpu_data.get("GPU use (%)", 0)),
            mem_util_pct=float(gpu_data.get("GPU memory use (%)", 0)),
        )

    def _collect_sample_fallback(self, t_ns: int) -> GPUSample:
        """Collect sample using individual rocm-smi calls."""
        gfx_clock = 0.0
        mem_clock = 0.0
        power = 0.0
        temp = 0.0
        vram_used = 0.0
        vram_total = 0.0
        gpu_util = 0.0

        # Clocks
        result = run_command(["rocm-smi", "-d", str(self.device_id), "--showclocks"])
        if result:
            for line in result.split("\n"):
                if "sclk" in line.lower() or "gfx" in line.lower():
                    match = re.search(r"(\d+(?:\.\d+)?)\s*[Mm][Hh]z", line)
                    if match:
                        gfx_clock = float(match.group(1))
                elif "mclk" in line.lower() or "mem" in line.lower():
                    match = re.search(r"(\d+(?:\.\d+)?)\s*[Mm][Hh]z", line)
                    if match:
                        mem_clock = float(match.group(1))

        # Power
        result = run_command(["rocm-smi", "-d", str(self.device_id), "--showpower"])
        if result:
            match = re.search(r"(\d+(?:\.\d+)?)\s*[Ww]", result)
            if match:
                power = float(match.group(1))

        # Temperature
        result = run_command(["rocm-smi", "-d", str(self.device_id), "--showtemp"])
        if result:
            match = re.search(r"(\d+(?:\.\d+)?)\s*[Cc]", result)
            if match:
                temp = float(match.group(1))

        # VRAM
        result = run_command(["rocm-smi", "-d", str(self.device_id), "--showmeminfo", "vram"])
        if result:
            for line in result.split("\n"):
                if "used" in line.lower():
                    match = re.search(r"(\d+)", line)
                    if match:
                        vram_used = float(match.group(1)) / (1024**2)  # Assume bytes
                elif "total" in line.lower():
                    match = re.search(r"(\d+)", line)
                    if match:
                        vram_total = float(match.group(1)) / (1024**2)

        # Utilization
        result = run_command(["rocm-smi", "-d", str(self.device_id), "--showuse"])
        if result:
            match = re.search(r"GPU use\s*\(%\)\s*:\s*(\d+)", result, re.IGNORECASE)
            if match:
                gpu_util = float(match.group(1))

        return GPUSample(
            t_ns=t_ns,
            device_id=self.device_id,
            gfx_clock_mhz=gfx_clock,
            mem_clock_mhz=mem_clock,
            power_w=power,
            temp_c=temp,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            gpu_util_pct=gpu_util,
            mem_util_pct=0.0,
        )

    def to_events(self) -> List[GPUEvent]:
        """Convert samples to GPUEvent schema."""
        return [
            GPUEvent(
                t_ns=s.t_ns,
                gfx_clock_mhz=s.gfx_clock_mhz,
                mem_clock_mhz=s.mem_clock_mhz,
                power_w=s.power_w,
                temp_c=s.temp_c,
                vram_used_mb=s.vram_used_mb,
                gpu_util_pct=s.gpu_util_pct,
            )
            for s in self.samples
        ]

    def to_dataframe(self):
        """Convert samples to pandas DataFrame."""
        import pandas as pd

        if not self.samples:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "t_ns": s.t_ns,
                    "t_ms": s.t_ns / 1_000_000,
                    "gfx_clock_mhz": s.gfx_clock_mhz,
                    "mem_clock_mhz": s.mem_clock_mhz,
                    "power_w": s.power_w,
                    "temp_c": s.temp_c,
                    "vram_used_mb": s.vram_used_mb,
                    "gpu_util_pct": s.gpu_util_pct,
                }
                for s in self.samples
            ]
        )

    def get_summary(self) -> Dict:
        """Get summary statistics of collected samples."""
        if not self.samples:
            return {}

        import numpy as np

        gfx_clocks = [s.gfx_clock_mhz for s in self.samples if s.gfx_clock_mhz > 0]
        powers = [s.power_w for s in self.samples if s.power_w > 0]
        temps = [s.temp_c for s in self.samples if s.temp_c > 0]
        utils = [s.gpu_util_pct for s in self.samples]

        return {
            "sample_count": len(self.samples),
            "gfx_clock_mean": float(np.mean(gfx_clocks)) if gfx_clocks else 0,
            "gfx_clock_min": float(np.min(gfx_clocks)) if gfx_clocks else 0,
            "gfx_clock_max": float(np.max(gfx_clocks)) if gfx_clocks else 0,
            "gfx_clock_std": float(np.std(gfx_clocks)) if gfx_clocks else 0,
            "power_mean": float(np.mean(powers)) if powers else 0,
            "power_max": float(np.max(powers)) if powers else 0,
            "temp_mean": float(np.mean(temps)) if temps else 0,
            "temp_max": float(np.max(temps)) if temps else 0,
            "gpu_util_mean": float(np.mean(utils)),
            "gpu_util_max": float(np.max(utils)),
            "vram_used_max_mb": max(s.vram_used_mb for s in self.samples),
        }
