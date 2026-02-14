"""
Report Renderer
Generates HTML and terminal reports from session data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from . import plots

logger = logging.getLogger(__name__)


class ReportRenderer:
    """
    Renders performance reports from session bundles.

    Supports:
    - Terminal (colored text)
    - HTML (Jinja2 templates)
    - JSON (structured data)
    """

    def __init__(self, session_path: Path):
        self.session_path = Path(session_path)
        self.data: Dict[str, Any] = {}
        self._load_session()

    def _load_session(self) -> None:
        """Load all session artifacts."""
        # Load session metadata
        meta_file = self.session_path / "session_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                self.data["metadata"] = json.load(f)

        # Load inference results
        results_file = self.session_path / "inference_results.json"
        if results_file.exists():
            with open(results_file) as f:
                self.data["inference_results"] = json.load(f)

        # Load derived metrics
        metrics_file = self.session_path / "derived_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                self.data["metrics"] = json.load(f)

        # Load bottleneck analysis
        bottleneck_file = self.session_path / "bottleneck.json"
        if bottleneck_file.exists():
            with open(bottleneck_file) as f:
                self.data["bottleneck"] = json.load(f)

        # Load kernel summary if available
        kernel_file = self.session_path / "kernel_summary.json"
        if kernel_file.exists():
            with open(kernel_file) as f:
                self.data["kernel_summary"] = json.load(f)

        # Load GPU samples
        gpu_file = self.session_path / "gpu_samples.json"
        if gpu_file.exists():
            with open(gpu_file) as f:
                self.data["gpu_samples"] = json.load(f)

    def _generate_plots(self) -> Dict[str, str]:
        """
        Generate embedded plots for the report.

        Returns:
            Dict of plot_name -> base64 PNG string
        """
        plots_dict = {}

        # Latency histogram
        results = self.data.get("inference_results", {})
        latencies = results.get("latencies_ms", [])
        if latencies:
            plots_dict["latency_histogram"] = plots.create_latency_histogram(
                latencies, title="Inference Latency Distribution"
            )
            plots_dict["latency_timeline"] = plots.create_latency_timeline(
                latencies, title="Latency Over Iterations"
            )

        # Kernel charts
        kernels = self.data.get("kernel_summary", [])
        if kernels:
            plots_dict["kernel_bar"] = plots.create_kernel_bar_chart(
                kernels, title="Top GPU Kernels by Time"
            )
            plots_dict["kernel_duration_hist"] = plots.create_kernel_duration_histogram(
                kernels, title="Kernel Duration Distribution"
            )

        # GPU timeline
        gpu_samples = self.data.get("gpu_samples", [])
        if gpu_samples:
            plots_dict["gpu_timeline"] = plots.create_gpu_timeline(
                gpu_samples, title="GPU Metrics Over Time"
            )

        # Bottleneck radar
        metrics = self.data.get("metrics", {})
        if metrics:
            plots_dict["bottleneck_radar"] = plots.create_bottleneck_radar(
                metrics, title="Performance Indicators"
            )

        return plots_dict

    def render_terminal(self) -> str:
        """Render report for terminal output."""
        lines = []

        # Header
        lines.append("╔" + "═" * 60 + "╗")
        lines.append("║" + " AMD AI Compute Observatory - Session Report ".center(60) + "║")
        lines.append("╚" + "═" * 60 + "╝")
        lines.append("")

        # Session info
        meta = self.data.get("metadata", {})
        lines.append("═══ Session Information ═══")
        lines.append(f"  Session ID:  {meta.get('session_id', 'N/A')}")
        lines.append(f"  Timestamp:   {meta.get('timestamp', 'N/A')}")
        lines.append(f"  Tag:         {meta.get('tag', 'N/A')}")
        lines.append(f"  Host:        {meta.get('hostname', 'N/A')}")

        gpu = meta.get("gpu", {})
        if gpu:
            lines.append(f"  GPU:         {gpu.get('name', 'N/A')}")
            lines.append(f"  VRAM:        {gpu.get('vram_total_mb', 0)} MB")

        lines.append("")

        # Latency metrics
        metrics = self.data.get("metrics", {})
        lines.append("═══ Performance Metrics ═══")
        lines.append(f"  Warmup Iterations:      {metrics.get('warmup_iterations', 'N/A')}")
        lines.append(f"  Measurement Iterations: {metrics.get('measurement_iterations', 'N/A')}")
        lines.append(f"  Mean Latency:           {metrics.get('measurement_mean_ms', 0):.3f} ms")
        lines.append(f"  Std Deviation:          {metrics.get('measurement_std_ms', 0):.3f} ms")
        lines.append(f"  P99 Latency:            {metrics.get('measurement_p99_ms', 0):.3f} ms")
        lines.append(f"  Throughput:             {metrics.get('throughput_ips', 0):.1f} infer/sec")
        lines.append("")

        # Efficiency metrics
        lines.append("═══ Efficiency Metrics ═══")
        lines.append(f"  GPU Active Ratio:       {metrics.get('gpu_active_ratio', 0):.2%}")
        lines.append(f"  Kernel Amplification:   {metrics.get('kar', 0):.1f}x")
        lines.append(f"  Microkernel %:          {metrics.get('microkernel_pct', 0):.1f}%")
        lines.append("")

        # GPU utilization
        if "gpu_util_mean_pct" in metrics:
            lines.append("═══ GPU Utilization ═══")
            lines.append(f"  GPU Util (mean):        {metrics.get('gpu_util_mean_pct', 0):.1f}%")
            lines.append(f"  GPU Util (max):         {metrics.get('gpu_util_max_pct', 0):.1f}%")
            lines.append(f"  Power (mean):           {metrics.get('power_mean_w', 0):.0f}W")
            lines.append(f"  Temperature (max):      {metrics.get('temp_max_c', 0):.0f}°C")
            lines.append("")

        # Bottleneck analysis
        bottleneck = self.data.get("bottleneck", {})
        if bottleneck:
            lines.append("═══ Bottleneck Analysis ═══")
            lines.append(f"  Primary:     {bottleneck.get('primary', 'N/A').upper()}")
            lines.append(f"  Confidence:  {bottleneck.get('confidence', 0):.0%}")

            secondary = bottleneck.get("secondary", [])
            if secondary:
                lines.append(f"  Secondary:   {', '.join(secondary)}")

            evidence = bottleneck.get("evidence", [])
            if evidence:
                lines.append("\n  Evidence:")
                for ev in evidence[:5]:
                    lines.append(f"    • {ev}")

            recommendations = bottleneck.get("recommendations", [])
            if recommendations:
                lines.append("\n  Recommendations:")
                for rec in recommendations[:3]:
                    lines.append(f"    → {rec}")

        lines.append("")
        lines.append("═" * 62)

        return "\n".join(lines)

    def render_html(self) -> str:
        """Render HTML report."""
        try:
            from jinja2 import Template
        except ImportError:
            return self._render_basic_html()

        # Generate plots
        generated_plots = self._generate_plots()

        template = Template(HTML_TEMPLATE)
        return template.render(
            data=self.data,
            timestamp=datetime.now().isoformat(),
            session_path=str(self.session_path),
            plots=generated_plots,
            embed_plot=plots.embed_plot_in_html,
        )

    def _render_basic_html(self) -> str:
        """Render basic HTML without Jinja2."""
        meta = self.data.get("metadata", {})
        metrics = self.data.get("metrics", {})
        bottleneck = self.data.get("bottleneck", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AACO Report - {meta.get("session_id", "Session")}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #e62020, #1a1a1a); color: white; 
                   padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ margin-top: 0; color: #333; border-bottom: 2px solid #e62020; padding-bottom: 10px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; 
                   border-bottom: 1px solid #eee; }}
        .metric-name {{ color: #666; }}
        .metric-value {{ font-weight: bold; color: #333; }}
        .bottleneck {{ font-size: 24px; font-weight: bold; 
                       color: {"#d32f2f" if bottleneck.get("primary") == "launch_overhead" else "#388e3c"}; }}
        .evidence {{ color: #666; margin: 5px 0; }}
        .recommendation {{ background: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AMD AI Compute Observatory</h1>
        <p>Session: {meta.get("session_id", "N/A")} | {meta.get("timestamp", "N/A")}</p>
    </div>
    
    <div class="card">
        <h2>Performance Summary</h2>
        <div class="metric"><span class="metric-name">Mean Latency</span><span class="metric-value">{metrics.get("measurement_mean_ms", 0):.3f} ms</span></div>
        <div class="metric"><span class="metric-name">Std Deviation</span><span class="metric-value">{metrics.get("measurement_std_ms", 0):.3f} ms</span></div>
        <div class="metric"><span class="metric-name">P99 Latency</span><span class="metric-value">{metrics.get("measurement_p99_ms", 0):.3f} ms</span></div>
        <div class="metric"><span class="metric-name">Throughput</span><span class="metric-value">{metrics.get("throughput_ips", 0):.1f} infer/sec</span></div>
    </div>
    
    <div class="card">
        <h2>Efficiency Metrics</h2>
        <div class="metric"><span class="metric-name">GPU Active Ratio</span><span class="metric-value">{metrics.get("gpu_active_ratio", 0) * 100:.1f}%</span></div>
        <div class="metric"><span class="metric-name">Kernel Amplification</span><span class="metric-value">{metrics.get("kar", 0):.1f}x</span></div>
        <div class="metric"><span class="metric-name">Microkernel %</span><span class="metric-value">{metrics.get("microkernel_pct", 0):.1f}%</span></div>
    </div>
    
    <div class="card">
        <h2>Bottleneck Analysis</h2>
        <p class="bottleneck">{bottleneck.get("primary", "N/A").upper().replace("_", " ")}</p>
        <p>Confidence: {bottleneck.get("confidence", 0) * 100:.0f}%</p>
    </div>
</body>
</html>"""
        return html

    def render_json(self) -> str:
        """Render JSON report."""
        return json.dumps(self.data, indent=2, default=str)


# Full HTML template with Jinja2
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AACO Report - {{ data.metadata.session_id }}</title>
    <style>
        :root {
            --amd-red: #e62020;
            --amd-dark: #1a1a1a;
            --bg-light: #f8f9fa;
            --card-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        * { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: var(--bg-light);
        }
        
        .header {
            background: linear-gradient(135deg, var(--amd-red) 0%, var(--amd-dark) 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 700;
        }
        
        .header .subtitle {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: var(--card-shadow);
        }
        
        .card h2 {
            margin: 0 0 20px 0;
            color: var(--amd-dark);
            font-size: 1.3em;
            border-bottom: 3px solid var(--amd-red);
            padding-bottom: 10px;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric-row:last-child { border-bottom: none; }
        
        .metric-name {
            color: #666;
            font-weight: 500;
        }
        
        .metric-value {
            font-weight: 700;
            color: var(--amd-dark);
            font-family: 'SF Mono', 'Consolas', monospace;
        }
        
        .bottleneck-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.2em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .bottleneck-badge.launch_overhead { background: #ffebee; color: #c62828; }
        .bottleneck-badge.compute_bound { background: #e3f2fd; color: #1565c0; }
        .bottleneck-badge.memory_bound { background: #fff3e0; color: #ef6c00; }
        .bottleneck-badge.balanced { background: #e8f5e9; color: #2e7d32; }
        
        .evidence-list {
            list-style: none;
            padding: 0;
            margin: 15px 0;
        }
        
        .evidence-list li {
            padding: 8px 15px;
            margin: 5px 0;
            background: #f5f5f5;
            border-radius: 5px;
            border-left: 3px solid var(--amd-red);
        }
        
        .recommendation {
            background: linear-gradient(135deg, #fff8e1, #ffecb3);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
        
        .plot-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .plot-unavailable {
            padding: 20px;
            text-align: center;
            color: #999;
            background: #f9f9f9;
            border-radius: 8px;
            font-style: italic;
        }
        
        .plot-section {
            margin-top: 30px;
        }
        
        .plot-section h2 {
            display: flex;
            align-items: center;
            gap: 10px;
        }
            border-left: 4px solid #ffc107;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #f5f5f5;
            font-weight: 600;
            color: var(--amd-dark);
        }
        
        .progress-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--amd-red), #ff5252);
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AMD AI Compute Observatory</h1>
        <p class="subtitle">
            Session {{ data.metadata.session_id }} | 
            {{ data.metadata.timestamp }} |
            {{ data.metadata.hostname | default('Unknown Host') }}
        </p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h2>📊 Latency Metrics</h2>
            <div class="metric-row">
                <span class="metric-name">Mean Latency</span>
                <span class="metric-value">{{ "%.3f"|format(data.metrics.measurement_mean_ms|default(0)) }} ms</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Std Deviation</span>
                <span class="metric-value">{{ "%.3f"|format(data.metrics.measurement_std_ms|default(0)) }} ms</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">P99 Latency</span>
                <span class="metric-value">{{ "%.3f"|format(data.metrics.measurement_p99_ms|default(0)) }} ms</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Throughput</span>
                <span class="metric-value">{{ "%.1f"|format(data.metrics.throughput_ips|default(0)) }} infer/sec</span>
            </div>
        </div>
        
        <div class="card">
            <h2>⚡ Efficiency</h2>
            <div class="metric-row">
                <span class="metric-name">GPU Active Ratio</span>
                <span class="metric-value">{{ "%.1f"|format((data.metrics.gpu_active_ratio|default(0))*100) }}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Kernel Amplification</span>
                <span class="metric-value">{{ "%.1f"|format(data.metrics.kar|default(0)) }}x</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Microkernel %</span>
                <span class="metric-value">{{ "%.1f"|format(data.metrics.microkernel_pct|default(0)) }}%</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🖥️ GPU Utilization</h2>
            <div class="metric-row">
                <span class="metric-name">GPU Util (mean)</span>
                <span class="metric-value">{{ "%.1f"|format(data.metrics.gpu_util_mean_pct|default(0)) }}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Power (mean)</span>
                <span class="metric-value">{{ "%.0f"|format(data.metrics.power_mean_w|default(0)) }}W</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">Temperature (max)</span>
                <span class="metric-value">{{ "%.0f"|format(data.metrics.temp_max_c|default(0)) }}°C</span>
            </div>
            <div class="metric-row">
                <span class="metric-name">VRAM (max)</span>
                <span class="metric-value">{{ "%.0f"|format(data.metrics.vram_max_mb|default(0)) }} MB</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 Bottleneck Analysis</h2>
            <p>
                <span class="bottleneck-badge {{ data.bottleneck.primary }}">
                    {{ data.bottleneck.primary | replace('_', ' ') }}
                </span>
            </p>
            <p><strong>Confidence:</strong> {{ "%.0f"|format((data.bottleneck.confidence|default(0))*100) }}%</p>
            
            {% if data.bottleneck.evidence %}
            <h3>Evidence</h3>
            <ul class="evidence-list">
                {% for ev in data.bottleneck.evidence[:5] %}
                <li>{{ ev }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            {% if data.bottleneck.recommendations %}
            <h3>Recommendations</h3>
            {% for rec in data.bottleneck.recommendations[:3] %}
            <div class="recommendation">→ {{ rec }}</div>
            {% endfor %}
            {% endif %}
        </div>
    </div>
    
    {% if data.kernel_summary %}
    <div class="card">
        <h2>🔥 Top Kernels</h2>
        <table>
            <thead>
                <tr>
                    <th>Kernel</th>
                    <th>Calls</th>
                    <th>Total (ms)</th>
                    <th>Avg (μs)</th>
                    <th>% Total</th>
                </tr>
            </thead>
            <tbody>
                {% for k in data.kernel_summary[:10] %}
                <tr>
                    <td style="font-family: monospace; font-size: 0.85em;">{{ k.kernel_name[:50] }}{% if k.kernel_name|length > 50 %}...{% endif %}</td>
                    <td>{{ k.calls }}</td>
                    <td>{{ "%.2f"|format(k.total_time_ms) }}</td>
                    <td>{{ "%.1f"|format(k.avg_time_us) }}</td>
                    <td>
                        <div class="progress-bar" style="width: 100px; display: inline-block; vertical-align: middle;">
                            <div class="progress-fill" style="width: {{ k.pct_total }}%;"></div>
                        </div>
                        {{ "%.1f"|format(k.pct_total) }}%
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    <!-- Embedded Plots Section -->
    {% if plots %}
    <div class="plot-section">
        {% if plots.latency_histogram %}
        <div class="card">
            <h2>📈 Latency Distribution</h2>
            {{ embed_plot(plots.latency_histogram, "Latency Histogram") }}
        </div>
        {% endif %}
        
        <div class="grid">
            {% if plots.latency_timeline %}
            <div class="card">
                <h2>📉 Latency Timeline</h2>
                {{ embed_plot(plots.latency_timeline, "Latency Timeline") }}
            </div>
            {% endif %}
            
            {% if plots.kernel_bar %}
            <div class="card">
                <h2>🔥 Kernel Time Distribution</h2>
                {{ embed_plot(plots.kernel_bar, "Kernel Bar Chart") }}
            </div>
            {% endif %}
        </div>
        
        <div class="grid">
            {% if plots.kernel_duration_hist %}
            <div class="card">
                <h2>⏱️ Kernel Duration Distribution</h2>
                {{ embed_plot(plots.kernel_duration_hist, "Kernel Duration Histogram") }}
            </div>
            {% endif %}
            
            {% if plots.gpu_timeline %}
            <div class="card">
                <h2>🖥️ GPU Metrics Timeline</h2>
                {{ embed_plot(plots.gpu_timeline, "GPU Timeline") }}
            </div>
            {% endif %}
        </div>
        
        {% if plots.bottleneck_radar %}
        <div class="card" style="max-width: 600px; margin: 20px auto;">
            <h2>🎯 Performance Radar</h2>
            {{ embed_plot(plots.bottleneck_radar, "Bottleneck Radar") }}
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    <div class="footer">
        <p>Generated by AMD AI Compute Observatory (AACO) at {{ timestamp }}</p>
        <p>Session: {{ session_path }}</p>
    </div>
</body>
</html>
"""
