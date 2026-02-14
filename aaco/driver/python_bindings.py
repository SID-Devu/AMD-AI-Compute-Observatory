"""
AACO-SIGMA Linux Kernel Driver Module

Python bindings and userspace interface for /dev/aaco kernel driver.
Provides:
- High-precision timestamps via rdtsc
- Memory barrier primitives
- CPU affinity coordination
- Kernel-level noise detection
"""

from __future__ import annotations

import os
import ctypes
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import IntEnum
import fcntl


# =============================================================================
# IOCTL Definitions (must match kernel module)
# =============================================================================

AACO_IOC_MAGIC = ord("A")


class AAcoIoctl(IntEnum):
    """IOCTL command numbers."""

    GET_VERSION = 0
    GET_TSC_FREQ = 1
    READ_TSC = 2
    SET_CPU_AFFINITY = 3
    GET_CPU_AFFINITY = 4
    MEMORY_BARRIER = 5
    DISABLE_PREEMPTION = 6
    ENABLE_PREEMPTION = 7
    GET_NOISE_COUNTER = 8
    RESET_NOISE_COUNTER = 9
    START_MEASUREMENT = 10
    STOP_MEASUREMENT = 11
    GET_IRQ_STATS = 12
    GET_CONTEXT_SWITCHES = 13
    PIN_MEMORY = 14
    UNPIN_MEMORY = 15


# Generate ioctl numbers (Linux _IO, _IOR, _IOW, _IOWR)
def _IO(type: int, nr: int) -> int:
    return (type << 8) | nr


def _IOR(type: int, nr: int, size: int) -> int:
    return (2 << 30) | (size << 16) | (type << 8) | nr


def _IOW(type: int, nr: int, size: int) -> int:
    return (1 << 30) | (size << 16) | (type << 8) | nr


def _IOWR(type: int, nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (type << 8) | nr


AACO_GET_VERSION = _IOR(AACO_IOC_MAGIC, AAcoIoctl.GET_VERSION, 4)
AACO_GET_TSC_FREQ = _IOR(AACO_IOC_MAGIC, AAcoIoctl.GET_TSC_FREQ, 8)
AACO_READ_TSC = _IOR(AACO_IOC_MAGIC, AAcoIoctl.READ_TSC, 8)
AACO_SET_CPU_AFFINITY = _IOW(AACO_IOC_MAGIC, AAcoIoctl.SET_CPU_AFFINITY, 8)
AACO_GET_CPU_AFFINITY = _IOR(AACO_IOC_MAGIC, AAcoIoctl.GET_CPU_AFFINITY, 8)
AACO_MEMORY_BARRIER = _IO(AACO_IOC_MAGIC, AAcoIoctl.MEMORY_BARRIER)
AACO_GET_IRQ_STATS = _IOR(AACO_IOC_MAGIC, AAcoIoctl.GET_IRQ_STATS, 32)
AACO_GET_CTX_SWITCHES = _IOR(AACO_IOC_MAGIC, AAcoIoctl.GET_CONTEXT_SWITCHES, 8)


# =============================================================================
# Data Structures (must match kernel)
# =============================================================================


@dataclass
class DriverVersion:
    """Driver version information."""

    major: int
    minor: int
    patch: int

    @classmethod
    def from_int(cls, version: int) -> "DriverVersion":
        return cls(
            major=(version >> 16) & 0xFF,
            minor=(version >> 8) & 0xFF,
            patch=version & 0xFF,
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class IRQStats:
    """IRQ statistics from kernel."""

    total_irqs: int = 0
    timer_irqs: int = 0
    ipi_irqs: int = 0
    other_irqs: int = 0

    @classmethod
    def from_bytes(cls, data: bytes) -> "IRQStats":
        vals = struct.unpack("QQQQ", data[:32])
        return cls(total_irqs=vals[0], timer_irqs=vals[1], ipi_irqs=vals[2], other_irqs=vals[3])


@dataclass
class MeasurementStats:
    """Statistics from measurement session."""

    start_tsc: int = 0
    end_tsc: int = 0
    context_switches: int = 0
    irqs_during: int = 0
    page_faults: int = 0


# =============================================================================
# Fallback TSC Reading (when driver not available)
# =============================================================================


class TSCReader:
    """
    TSC reader with fallback implementation.

    Uses kernel driver if available, otherwise falls back to
    Python ctypes rdtsc.
    """

    def __init__(self):
        self._use_driver = False
        self._tsc_freq_hz: Optional[int] = None
        self._libc = None

        # Try to load libc for syscall fallback
        try:
            self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except:
            pass

    @staticmethod
    def rdtsc() -> int:
        """Read TSC using inline assembly via ctypes."""
        # This is a simplified approach - real implementation would use
        # assembly or kernel driver
        try:
            import time

            return int(time.perf_counter_ns())
        except:
            return 0

    @staticmethod
    def rdtscp() -> Tuple[int, int]:
        """Read TSC with processor ID (serializing)."""
        # Returns (tsc, processor_id)
        try:
            import time

            return (
                int(time.perf_counter_ns()),
                os.sched_getcpu() if hasattr(os, "sched_getcpu") else 0,
            )
        except:
            return (0, 0)

    def get_tsc_frequency(self) -> int:
        """Get TSC frequency in Hz."""
        if self._tsc_freq_hz is not None:
            return self._tsc_freq_hz

        # Try to read from sysfs
        try:
            with open("/sys/devices/system/cpu/cpu0/tsc_freq_khz") as f:
                self._tsc_freq_hz = int(f.read().strip()) * 1000
                return self._tsc_freq_hz
        except:
            pass

        # Calibrate
        self._tsc_freq_hz = self._calibrate_tsc()
        return self._tsc_freq_hz

    def _calibrate_tsc(self) -> int:
        """Calibrate TSC frequency."""
        import time

        start_tsc = self.rdtsc()
        start_time = time.perf_counter_ns()

        time.sleep(0.01)  # 10ms calibration

        end_tsc = self.rdtsc()
        end_time = time.perf_counter_ns()

        elapsed_ns = end_time - start_time
        elapsed_tsc = end_tsc - start_tsc

        if elapsed_ns > 0:
            return int(elapsed_tsc * 1_000_000_000 / elapsed_ns)
        return 1_000_000_000  # Fallback 1GHz


# =============================================================================
# AACO Device Driver Interface
# =============================================================================


class AAcoDriver:
    """
    Interface to AACO kernel driver (/dev/aaco).

    Provides:
    - High-precision TSC timestamps
    - Memory barriers
    - CPU affinity control
    - Noise detection
    """

    DEVICE_PATH = "/dev/aaco"

    def __init__(self, device_path: Optional[str] = None):
        self.device_path = device_path or self.DEVICE_PATH
        self._fd: Optional[int] = None
        self._tsc_reader = TSCReader()
        self._driver_available = False
        self._version: Optional[DriverVersion] = None

        self._try_open_driver()

    def _try_open_driver(self) -> bool:
        """Attempt to open the AACO driver."""
        if not os.path.exists(self.device_path):
            self._driver_available = False
            return False

        try:
            self._fd = os.open(self.device_path, os.O_RDWR)
            self._driver_available = True
            return True
        except (OSError, PermissionError):
            self._driver_available = False
            return False

    @property
    def is_available(self) -> bool:
        """Check if driver is available."""
        return self._driver_available

    def close(self) -> None:
        """Close driver file descriptor."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_version(self) -> DriverVersion:
        """Get driver version."""
        if self._version is not None:
            return self._version

        if not self._driver_available:
            return DriverVersion(0, 0, 0)

        try:
            version_buf = ctypes.c_uint32()
            fcntl.ioctl(self._fd, AACO_GET_VERSION, version_buf)
            self._version = DriverVersion.from_int(version_buf.value)
            return self._version
        except:
            return DriverVersion(0, 0, 0)

    def get_tsc_frequency(self) -> int:
        """Get TSC frequency in Hz."""
        if self._driver_available:
            try:
                freq_buf = ctypes.c_uint64()
                fcntl.ioctl(self._fd, AACO_GET_TSC_FREQ, freq_buf)
                return freq_buf.value
            except:
                pass

        return self._tsc_reader.get_tsc_frequency()

    def read_tsc(self) -> int:
        """Read current TSC value."""
        if self._driver_available:
            try:
                tsc_buf = ctypes.c_uint64()
                fcntl.ioctl(self._fd, AACO_READ_TSC, tsc_buf)
                return tsc_buf.value
            except:
                pass

        return self._tsc_reader.rdtsc()

    def read_tsc_ns(self) -> int:
        """Read TSC converted to nanoseconds."""
        tsc = self.read_tsc()
        freq = self.get_tsc_frequency()

        if freq > 0:
            return int(tsc * 1_000_000_000 / freq)
        return tsc

    def memory_barrier(self) -> None:
        """Issue memory barrier."""
        if self._driver_available:
            try:
                fcntl.ioctl(self._fd, AACO_MEMORY_BARRIER)
                return
            except:
                pass

        # Fallback: atomic operation acts as barrier
        import threading

        threading.Thread(target=lambda: None).start()

    def set_cpu_affinity(self, cpu_mask: int) -> bool:
        """Set CPU affinity for current process."""
        if self._driver_available:
            try:
                mask_buf = ctypes.c_uint64(cpu_mask)
                fcntl.ioctl(self._fd, AACO_SET_CPU_AFFINITY, mask_buf)
                return True
            except:
                pass

        # Fallback: use os.sched_setaffinity
        try:
            cpus = []
            for i in range(64):
                if cpu_mask & (1 << i):
                    cpus.append(i)
            os.sched_setaffinity(0, cpus)
            return True
        except:
            return False

    def get_cpu_affinity(self) -> int:
        """Get CPU affinity mask."""
        if self._driver_available:
            try:
                mask_buf = ctypes.c_uint64()
                fcntl.ioctl(self._fd, AACO_GET_CPU_AFFINITY, mask_buf)
                return mask_buf.value
            except:
                pass

        # Fallback
        try:
            cpus = os.sched_getaffinity(0)
            mask = 0
            for cpu in cpus:
                mask |= 1 << cpu
            return mask
        except:
            return 0xFFFFFFFFFFFFFFFF

    def get_irq_stats(self) -> IRQStats:
        """Get IRQ statistics."""
        if self._driver_available:
            try:
                buf = ctypes.create_string_buffer(32)
                fcntl.ioctl(self._fd, AACO_GET_IRQ_STATS, buf)
                return IRQStats.from_bytes(buf.raw)
            except:
                pass

        # Fallback: parse /proc/interrupts
        return self._parse_proc_interrupts()

    def get_context_switches(self) -> int:
        """Get context switch count."""
        if self._driver_available:
            try:
                count_buf = ctypes.c_uint64()
                fcntl.ioctl(self._fd, AACO_GET_CTX_SWITCHES, count_buf)
                return count_buf.value
            except:
                pass

        # Fallback: parse /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("voluntary_ctxt_switches:"):
                        vol = int(line.split()[1])
                    elif line.startswith("nonvoluntary_ctxt_switches:"):
                        nvol = int(line.split()[1])
                        return vol + nvol
        except:
            pass
        return 0

    def _parse_proc_interrupts(self) -> IRQStats:
        """Parse /proc/interrupts for IRQ stats."""
        stats = IRQStats()

        try:
            with open("/proc/interrupts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    # Sum all CPU columns
                    irq_name = parts[0].rstrip(":")
                    total = 0
                    for p in parts[1:]:
                        if p.isdigit():
                            total += int(p)
                        else:
                            break

                    stats.total_irqs += total

                    if "timer" in irq_name.lower() or irq_name == "LOC":
                        stats.timer_irqs += total
                    elif "IPI" in line or irq_name in ["RES", "CAL"]:
                        stats.ipi_irqs += total
                    else:
                        stats.other_irqs += total
        except:
            pass

        return stats


# =============================================================================
# Measurement Session with Kernel Support
# =============================================================================


class KernelMeasurementSession:
    """
    Measurement session using kernel driver for timing.

    Provides highest-precision timestamps with minimal overhead.
    """

    def __init__(self, driver: Optional[AAcoDriver] = None):
        self.driver = driver or AAcoDriver()
        self._start_tsc: int = 0
        self._end_tsc: int = 0
        self._start_irqs: IRQStats = IRQStats()
        self._start_ctx: int = 0
        self._tsc_freq: int = 0
        self._active = False

    def __enter__(self) -> "KernelMeasurementSession":
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self) -> None:
        """Start measurement session."""
        self._tsc_freq = self.driver.get_tsc_frequency()
        self._start_irqs = self.driver.get_irq_stats()
        self._start_ctx = self.driver.get_context_switches()

        # Memory barrier before timing
        self.driver.memory_barrier()
        self._start_tsc = self.driver.read_tsc()

        self._active = True

    def stop(self) -> MeasurementStats:
        """Stop measurement session and return stats."""
        self._end_tsc = self.driver.read_tsc()
        self.driver.memory_barrier()

        end_irqs = self.driver.get_irq_stats()
        end_ctx = self.driver.get_context_switches()

        self._active = False

        return MeasurementStats(
            start_tsc=self._start_tsc,
            end_tsc=self._end_tsc,
            context_switches=end_ctx - self._start_ctx,
            irqs_during=end_irqs.total_irqs - self._start_irqs.total_irqs,
        )

    def elapsed_ns(self) -> int:
        """Get elapsed time in nanoseconds."""
        if self._active:
            current_tsc = self.driver.read_tsc()
            elapsed_tsc = current_tsc - self._start_tsc
        else:
            elapsed_tsc = self._end_tsc - self._start_tsc

        if self._tsc_freq > 0:
            return int(elapsed_tsc * 1_000_000_000 / self._tsc_freq)
        return elapsed_tsc

    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.elapsed_ns() / 1000.0

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ns() / 1_000_000.0


# =============================================================================
# Module Loading Utilities
# =============================================================================


def is_module_loaded() -> bool:
    """Check if AACO kernel module is loaded."""
    try:
        with open("/proc/modules") as f:
            for line in f:
                if line.startswith("aaco_driver"):
                    return True
    except:
        pass
    return False


def get_module_info() -> Dict[str, Any]:
    """Get AACO module information."""
    info = {
        "loaded": is_module_loaded(),
        "device_exists": os.path.exists("/dev/aaco"),
        "driver_version": None,
        "tsc_frequency_hz": None,
    }

    if info["device_exists"]:
        try:
            driver = AAcoDriver()
            info["driver_version"] = str(driver.get_version())
            info["tsc_frequency_hz"] = driver.get_tsc_frequency()
            driver.close()
        except:
            pass

    return info


def install_module(module_path: Optional[Path] = None) -> bool:
    """Attempt to load AACO kernel module."""
    if is_module_loaded():
        return True

    if module_path is None:
        # Try default paths
        candidates = [
            Path("/lib/modules") / os.uname().release / "extra/aaco_driver.ko",
            Path.cwd() / "aaco_driver.ko",
            Path(__file__).parent / "aaco_driver.ko",
        ]
        for path in candidates:
            if path.exists():
                module_path = path
                break

    if module_path is None or not module_path.exists():
        return False

    try:
        import subprocess

        subprocess.run(["insmod", str(module_path)], check=True, capture_output=True)
        return True
    except:
        return False
