"""
AACO Streamlit Dashboard
Interactive performance observability dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Check for streamlit availability
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    import pandas as pd
    import numpy as np

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None


def check_dependencies():
    """Check if dashboard dependencies are available."""
    missing = []
    if not STREAMLIT_AVAILABLE:
        missing.append("streamlit")
    if not PANDAS_AVAILABLE:
        missing.append("pandas")
    if not PLOTLY_AVAILABLE:
        missing.append("plotly")
    return missing


class SessionLoader:
    """Loads and parses AACO session data."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []

        if not self.sessions_dir.exists():
            return sessions

        for session_path in self.sessions_dir.glob("*/*/session.json"):
            try:
                with open(session_path) as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data.get("session_id", "unknown"),
                        "path": str(session_path.parent),
                        "model": data.get("workload", {}).get("model_name", "unknown"),
                        "backend": data.get("backend", {}).get("name", "unknown"),
                        "created": data.get("created_utc", "unknown"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load {session_path}: {e}")

        return sessions

    def load_session(self, session_path: str) -> Dict[str, Any]:
        """Load full session data."""
        path = Path(session_path)
        data = {
            "path": session_path,
            "session": {},
            "metrics": {},
            "inference": [],
            "bottleneck": {},
            "kernels": [],
        }

        # Load session metadata
        if (path / "session.json").exists():
            with open(path / "session.json") as f:
                data["session"] = json.load(f)

        # Load metrics
        if (path / "derived_metrics.json").exists():
            with open(path / "derived_metrics.json") as f:
                data["metrics"] = json.load(f)

        # Load inference results
        if (path / "inference_results.json").exists():
            with open(path / "inference_results.json") as f:
                data["inference"] = json.load(f)

        # Load bottleneck classification
        if (path / "bottleneck.json").exists():
            with open(path / "bottleneck.json") as f:
                data["bottleneck"] = json.load(f)

        # Load kernel data
        if (path / "rocprof" / "kernel_summary.json").exists():
            with open(path / "rocprof" / "kernel_summary.json") as f:
                data["kernels"] = json.load(f)

        return data


def create_dashboard():
    """Create the Streamlit dashboard."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="AACO Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 AMD AI Compute Observatory")
    st.markdown("*Full-stack performance observability for AMD AI workloads*")

    # Sidebar
    with st.sidebar:
        st.header("📁 Session Browser")

        sessions_dir = st.text_input(
            "Sessions Directory",
            value="./aaco_sessions",
            help="Path to AACO sessions directory",
        )

        loader = SessionLoader(Path(sessions_dir))
        sessions = loader.list_sessions()

        if not sessions:
            st.warning("No sessions found. Run `aaco run` to create sessions.")
            return

        # Session selector
        session_options = {
            f"{s['model']} / {s['backend']} ({s['session_id'][:8]})": s["path"] for s in sessions
        }

        selected_label = st.selectbox("Select Session", options=list(session_options.keys()))

        if selected_label:
            selected_path = session_options[selected_label]
            session_data = loader.load_session(selected_path)

            # Comparison mode
            st.markdown("---")
            st.subheader("📊 Compare")
            compare_enabled = st.checkbox("Enable comparison")
            baseline_path = None

            if compare_enabled:
                baseline_options = {k: v for k, v in session_options.items() if v != selected_path}
                if baseline_options:
                    baseline_label = st.selectbox(
                        "Baseline Session", options=list(baseline_options.keys())
                    )
                    baseline_path = baseline_options.get(baseline_label)

    # Main content
    if not sessions:
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Overview", "🔥 Kernels", "⏱️ Timeline", "🎯 Bottleneck", "📋 Details"]
    )

    with tab1:
        render_overview(session_data)

    with tab2:
        render_kernels(session_data)

    with tab3:
        render_timeline(session_data)

    with tab4:
        render_bottleneck(session_data)

    with tab5:
        render_details(session_data)

    # Comparison view
    if compare_enabled and baseline_path:
        baseline_data = loader.load_session(baseline_path)
        st.markdown("---")
        st.header("🔄 Comparison")
        render_comparison(session_data, baseline_data)


def render_overview(data: Dict[str, Any]):
    """Render overview tab."""
    st.header("Performance Overview")

    data.get("session", {})
    metrics = data.get("metrics", {})

    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latency = metrics.get("latency", {})
        st.metric("P50 Latency", f"{latency.get('p50_ms', 0):.2f} ms", delta=None)

    with col2:
        st.metric("P99 Latency", f"{latency.get('p99_ms', 0):.2f} ms")

    with col3:
        throughput = metrics.get("throughput", {})
        st.metric("Throughput", f"{throughput.get('inferences_per_sec', 0):.1f} inf/s")

    with col4:
        efficiency = metrics.get("efficiency", {})
        st.metric("GPU Active", f"{efficiency.get('gpu_active_ratio', 0) * 100:.1f}%")

    # Latency distribution plot
    inference = data.get("inference", [])
    if inference and PLOTLY_AVAILABLE:
        st.subheader("Latency Distribution")

        latencies = [r.get("latency_ms", 0) for r in inference if r.get("phase") == "measurement"]

        if latencies:
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=latencies,
                    nbinsx=30,
                    name="Latency",
                    marker_color="steelblue",
                )
            )

            # Add percentile lines
            p50 = np.percentile(latencies, 50)
            p99 = np.percentile(latencies, 99)

            fig.add_vline(
                x=p50,
                line_dash="dash",
                line_color="green",
                annotation_text=f"P50: {p50:.2f}ms",
            )
            fig.add_vline(
                x=p99,
                line_dash="dash",
                line_color="red",
                annotation_text=f"P99: {p99:.2f}ms",
            )

            fig.update_layout(
                title="Inference Latency Distribution",
                xaxis_title="Latency (ms)",
                yaxis_title="Count",
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)


def render_kernels(data: Dict[str, Any]):
    """Render kernels tab."""
    st.header("GPU Kernel Analysis")

    kernels = data.get("kernels", [])

    if not kernels:
        st.info("No kernel data available. Run with --profile to capture kernel traces.")
        return

    if not PANDAS_AVAILABLE:
        st.error("pandas required for kernel analysis")
        return

    # Summary metrics
    total_calls = sum(k.get("calls", 0) for k in kernels)
    total_time = sum(k.get("total_time_ms", 0) for k in kernels)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Kernels", f"{total_calls:,}")

    with col2:
        st.metric("Unique Kernels", len(kernels))

    with col3:
        avg_us = (total_time * 1000 / total_calls) if total_calls > 0 else 0
        st.metric("Avg Duration", f"{avg_us:.1f} μs")

    # Top kernels table
    st.subheader("Top Kernels by Time")

    df = pd.DataFrame(kernels)
    if not df.empty:
        df = df.sort_values("total_time_ms", ascending=False).head(15)
        df["pct"] = df["total_time_ms"] / total_time * 100

        st.dataframe(
            df[["kernel_name", "calls", "avg_time_us", "total_time_ms", "pct"]].rename(
                columns={
                    "kernel_name": "Kernel",
                    "calls": "Calls",
                    "avg_time_us": "Avg (μs)",
                    "total_time_ms": "Total (ms)",
                    "pct": "% Time",
                }
            ),
            use_container_width=True,
        )

    # Kernel duration distribution
    if PLOTLY_AVAILABLE and kernels:
        st.subheader("Kernel Duration Distribution")

        # Create histogram of average durations
        durations = [k.get("avg_time_us", 0) for k in kernels]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=durations,
                nbinsx=20,
                marker_color="orange",
            )
        )

        # Add microkernel threshold line
        fig.add_vline(
            x=10,
            line_dash="dash",
            line_color="red",
            annotation_text="Microkernel threshold (10μs)",
        )

        fig.update_layout(
            title="Kernel Average Duration Distribution",
            xaxis_title="Average Duration (μs)",
            yaxis_title="Count",
            xaxis_type="log",
        )

        st.plotly_chart(fig, use_container_width=True)


def render_timeline(data: Dict[str, Any]):
    """Render timeline tab."""
    st.header("Execution Timeline")

    inference = data.get("inference", [])

    if not inference:
        st.info("No timeline data available.")
        return

    if not PLOTLY_AVAILABLE:
        st.error("plotly required for timeline visualization")
        return

    # Latency over time
    st.subheader("Latency Over Time")

    measure_results = [r for r in inference if r.get("phase") == "measurement"]

    if measure_results:
        latencies = [r.get("latency_ms", 0) for r in measure_results]
        iterations = list(range(len(latencies)))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=latencies,
                mode="lines+markers",
                name="Latency",
                line=dict(color="steelblue"),
                marker=dict(size=4),
            )
        )

        # Add mean line
        mean_lat = np.mean(latencies)
        fig.add_hline(
            y=mean_lat,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Mean: {mean_lat:.2f}ms",
        )

        fig.update_layout(
            title="Inference Latency Over Time",
            xaxis_title="Iteration",
            yaxis_title="Latency (ms)",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Detect and highlight spikes
    st.subheader("Latency Spike Analysis")

    if measure_results:
        latencies = [r.get("latency_ms", 0) for r in measure_results]
        median = np.median(latencies)
        threshold = median * 1.5

        spikes = [(i, lat) for i, lat in enumerate(latencies) if lat > threshold]

        if spikes:
            st.warning(f"Detected {len(spikes)} latency spike(s) (>50% above median)")

            for idx, lat in spikes[:5]:
                st.write(
                    f"- Iteration {idx}: {lat:.2f}ms ({(lat / median - 1) * 100:.0f}% above median)"
                )
        else:
            st.success("No significant latency spikes detected")


def render_bottleneck(data: Dict[str, Any]):
    """Render bottleneck analysis tab."""
    st.header("Bottleneck Analysis")

    bottleneck = data.get("bottleneck", {})

    if not bottleneck:
        st.info("No bottleneck analysis available.")
        return

    # Primary bottleneck
    primary = bottleneck.get("primary", "unknown")
    confidence = bottleneck.get("confidence", 0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification")

        color_map = {
            "compute_bound": "🟢",
            "memory_bound": "🟡",
            "launch_overhead": "🔴",
            "cpu_bound": "🟠",
            "balanced": "🟢",
        }

        icon = color_map.get(primary, "⚪")
        st.markdown(f"### {icon} {primary.replace('_', ' ').title()}")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence * 100:.0f}%")

    with col2:
        st.subheader("Evidence")
        evidence = bottleneck.get("evidence", [])

        if evidence:
            for ev in evidence[:5]:
                st.markdown(f"• {ev}")
        else:
            st.info("No specific evidence recorded")

    # Recommendations
    st.subheader("Recommendations")
    recommendations = bottleneck.get("recommendations", [])

    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("No specific optimizations recommended at this time.")

    # Indicator breakdown
    indicators = bottleneck.get("indicators", {})
    if indicators and PLOTLY_AVAILABLE:
        st.subheader("Indicator Breakdown")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=list(indicators.keys()),
                y=list(indicators.values()),
                marker_color="steelblue",
            )
        )

        fig.update_layout(
            title="Bottleneck Indicators",
            xaxis_title="Indicator",
            yaxis_title="Value",
            xaxis_tickangle=45,
        )

        st.plotly_chart(fig, use_container_width=True)


def render_details(data: Dict[str, Any]):
    """Render session details tab."""
    st.header("Session Details")

    session = data.get("session", {})

    # Session info
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Session")
        st.json(
            {
                "session_id": session.get("session_id"),
                "created": session.get("created_utc"),
            }
        )

        st.subheader("Workload")
        st.json(session.get("workload", {}))

    with col2:
        st.subheader("Host")
        st.json(session.get("host", {}))

        st.subheader("GPU")
        st.json(session.get("gpu", {}))

    # Raw metrics
    st.subheader("Full Metrics")
    with st.expander("View raw metrics"):
        st.json(data.get("metrics", {}))


def render_comparison(current: Dict[str, Any], baseline: Dict[str, Any]):
    """Render comparison view."""
    st.subheader("Performance Comparison")

    curr_metrics = current.get("metrics", {})
    base_metrics = baseline.get("metrics", {})

    # Latency comparison
    col1, col2, col3 = st.columns(3)

    curr_lat = curr_metrics.get("latency", {})
    base_lat = base_metrics.get("latency", {})

    with col1:
        curr_p50 = curr_lat.get("p50_ms", 0)
        base_p50 = base_lat.get("p50_ms", 0)
        delta = ((curr_p50 - base_p50) / base_p50 * 100) if base_p50 > 0 else 0

        st.metric(
            "P50 Latency",
            f"{curr_p50:.2f} ms",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",
        )

    with col2:
        curr_p99 = curr_lat.get("p99_ms", 0)
        base_p99 = base_lat.get("p99_ms", 0)
        delta = ((curr_p99 - base_p99) / base_p99 * 100) if base_p99 > 0 else 0

        st.metric(
            "P99 Latency",
            f"{curr_p99:.2f} ms",
            delta=f"{delta:+.1f}%",
            delta_color="inverse",
        )

    with col3:
        curr_tput = curr_metrics.get("throughput", {}).get("inferences_per_sec", 0)
        base_tput = base_metrics.get("throughput", {}).get("inferences_per_sec", 0)
        delta = ((curr_tput - base_tput) / base_tput * 100) if base_tput > 0 else 0

        st.metric("Throughput", f"{curr_tput:.1f} inf/s", delta=f"{delta:+.1f}%")

    # Verdict
    lat_regression = curr_lat.get("p50_ms", 0) > base_lat.get("p50_ms", 0) * 1.05

    if lat_regression:
        st.error("⚠️ REGRESSION DETECTED: Latency increased by >5%")
    else:
        st.success("✅ No regression detected")


def main():
    """Entry point for dashboard."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return

    create_dashboard()


if __name__ == "__main__":
    main()
