"""
Report Plot Generator
Creates embedded SVG/PNG plots for HTML reports.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None


def check_matplotlib():
    """Check if matplotlib is available."""
    return MATPLOTLIB_AVAILABLE


def create_latency_histogram(
    latencies: List[float],
    title: str = "Latency Distribution",
    percentiles: bool = True,
) -> Optional[str]:
    """
    Create latency histogram and return as base64 PNG.
    
    Args:
        latencies: List of latency values in ms
        title: Plot title
        percentiles: Add P50/P99 lines
        
    Returns:
        Base64 encoded PNG or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE or not latencies:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    arr = np.array(latencies)
    
    # Histogram
    ax.hist(arr, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    
    if percentiles:
        p50 = np.percentile(arr, 50)
        p99 = np.percentile(arr, 99)
        
        ax.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'P50: {p50:.2f}ms')
        ax.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'P99: {p99:.2f}ms')
        ax.legend(loc='upper right')
    
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def create_latency_timeline(
    latencies: List[float],
    title: str = "Latency Over Time",
) -> Optional[str]:
    """
    Create latency timeline plot.
    
    Args:
        latencies: List of latency values in ms (per iteration)
        title: Plot title
        
    Returns:
        Base64 encoded PNG or None.
    """
    if not MATPLOTLIB_AVAILABLE or not latencies:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    iterations = list(range(len(latencies)))
    arr = np.array(latencies)
    
    # Plot line
    ax.plot(iterations, latencies, color='steelblue', linewidth=1, alpha=0.7)
    ax.scatter(iterations, latencies, color='steelblue', s=10, alpha=0.5)
    
    # Add mean line
    mean_val = np.mean(arr)
    ax.axhline(mean_val, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(latencies)*0.02, mean_val*1.02, f'Mean: {mean_val:.2f}ms', fontsize=9, color='gray')
    
    # Highlight spikes (>2 std)
    std_val = np.std(arr)
    spike_threshold = mean_val + 2 * std_val
    spikes = [(i, lat) for i, lat in enumerate(latencies) if lat > spike_threshold]
    
    if spikes:
        spike_x, spike_y = zip(*spikes)
        ax.scatter(spike_x, spike_y, color='red', s=30, marker='x', label='Spike', zorder=5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if spikes:
        ax.legend(loc='upper right')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def create_kernel_bar_chart(
    kernels: List[Dict[str, Any]],
    top_n: int = 10,
    title: str = "Top GPU Kernels",
) -> Optional[str]:
    """
    Create bar chart of top kernels by time.
    
    Args:
        kernels: List of kernel summary dicts with 'kernel_name' and 'total_time_ms'
        top_n: Number of top kernels to show
        title: Plot title
        
    Returns:
        Base64 encoded PNG or None.
    """
    if not MATPLOTLIB_AVAILABLE or not kernels:
        return None
    
    # Sort and take top N
    sorted_kernels = sorted(kernels, key=lambda k: k.get('total_time_ms', 0), reverse=True)[:top_n]
    
    if not sorted_kernels:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [k.get('kernel_name', 'unknown')[-40:] for k in sorted_kernels]  # Truncate names
    times = [k.get('total_time_ms', 0) for k in sorted_kernels]
    
    # Horizontal bar chart
    bars = ax.barh(range(len(names)), times, color='orange', alpha=0.7)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Total Time (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, time in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{time:.2f}ms', va='center', fontsize=8)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def create_kernel_duration_histogram(
    kernels: List[Dict[str, Any]],
    title: str = "Kernel Duration Distribution",
    microkernel_threshold_us: float = 10.0,
) -> Optional[str]:
    """
    Create histogram of kernel durations with microkernel threshold.
    
    Args:
        kernels: List of kernel summary dicts with 'avg_time_us'
        title: Plot title
        microkernel_threshold_us: Threshold for microkernel coloring
        
    Returns:
        Base64 encoded PNG or None.
    """
    if not MATPLOTLIB_AVAILABLE or not kernels:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    durations = [k.get('avg_time_us', 0) for k in kernels]
    
    if not durations:
        return None
    
    # Use log scale bins
    min_dur = max(0.1, min(durations))
    max_dur = max(durations)
    bins = np.logspace(np.log10(min_dur), np.log10(max_dur), 30)
    
    # Color by microkernel status
    micro = [d for d in durations if d < microkernel_threshold_us]
    normal = [d for d in durations if d >= microkernel_threshold_us]
    
    if micro:
        ax.hist(micro, bins=bins, color='red', alpha=0.7, label=f'Microkernel (<{microkernel_threshold_us}μs)')
    if normal:
        ax.hist(normal, bins=bins, color='steelblue', alpha=0.7, label=f'Normal (≥{microkernel_threshold_us}μs)')
    
    ax.axvline(microkernel_threshold_us, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Average Duration (μs)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def create_gpu_timeline(
    samples: List[Dict[str, Any]],
    title: str = "GPU Metrics Over Time",
) -> Optional[str]:
    """
    Create multi-line plot of GPU metrics over time.
    
    Args:
        samples: List of GPU sample dicts with timestamps and metrics
        title: Plot title
        
    Returns:
        Base64 encoded PNG or None.
    """
    if not MATPLOTLIB_AVAILABLE or not samples:
        return None
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Extract time series
    times = [s.get('t_ms', i) for i, s in enumerate(samples)]
    
    # Power
    power = [s.get('power_w', 0) for s in samples]
    axes[0].plot(times, power, color='orange', linewidth=1)
    axes[0].set_ylabel('Power (W)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    # Temperature
    temp = [s.get('temp_c', 0) for s in samples]
    axes[1].plot(times, temp, color='red', linewidth=1)
    axes[1].set_ylabel('Temperature (°C)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Clocks
    sclk = [s.get('gfx_clock_mhz', 0) for s in samples]
    axes[2].plot(times, sclk, color='purple', linewidth=1)
    axes[2].set_ylabel('GFX Clock (MHz)', fontsize=10)
    axes[2].set_xlabel('Time (ms)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def create_bottleneck_radar(
    indicators: Dict[str, float],
    title: str = "Bottleneck Indicators",
) -> Optional[str]:
    """
    Create radar/spider chart of bottleneck indicators.
    
    Args:
        indicators: Dict of indicator name -> value (0-1 scale)
        title: Plot title
        
    Returns:
        Base64 encoded PNG or None.
    """
    if not MATPLOTLIB_AVAILABLE or not indicators:
        return None
    
    # Filter to relevant indicators and normalize to 0-1
    plot_indicators = {
        'Launch Tax': min(1, indicators.get('launch_tax_score', 0) / 100),
        'CPU Usage': min(1, indicators.get('cpu_pct', 0) / 100),
        'GPU Active': indicators.get('gpu_active_ratio', 0),
        'Microkernel': min(1, indicators.get('microkernel_pct', 0) / 100),
        'Memory': min(1, indicators.get('mem_util_pct', 0) / 100),
        'Stability': 1 - min(1, indicators.get('cov_pct', 0) / 50),
    }
    
    categories = list(plot_indicators.keys())
    values = list(plot_indicators.values())
    
    # Complete the circle
    values += values[:1]
    
    # Create angles
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def create_comparison_bar(
    current: Dict[str, float],
    baseline: Dict[str, float],
    metrics: List[str],
    title: str = "Performance Comparison",
) -> Optional[str]:
    """
    Create comparison bar chart.
    
    Args:
        current: Current session metrics
        baseline: Baseline session metrics
        metrics: List of metric names to compare
        title: Plot title
        
    Returns:
        Base64 encoded PNG or None.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    curr_vals = [current.get(m, 0) for m in metrics]
    base_vals = [baseline.get(m, 0) for m in metrics]
    
    ax.bar(x - width/2, base_vals, width, label='Baseline', color='gray', alpha=0.7)
    ax.bar(x + width/2, curr_vals, width, label='Current', color='steelblue', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def embed_plot_in_html(base64_png: str, alt_text: str = "Plot") -> str:
    """Create HTML img tag for embedded base64 PNG."""
    if not base64_png:
        return f'<p class="plot-unavailable">{alt_text} unavailable (matplotlib not installed)</p>'
    
    return f'<img src="data:image/png;base64,{base64_png}" alt="{alt_text}" class="plot-image" />'
