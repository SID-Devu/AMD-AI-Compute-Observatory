"""
AACO Command Line Interface
Main entry point for benchmarking, reporting, and analysis.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from aaco.core.session import SessionManager, Session
from aaco.core.schema import SessionMetadata
from aaco.runner.ort_runner import ORTRunner, RunConfig
from aaco.runner.model_registry import ModelRegistry, ModelConfig
from aaco.collectors.sys_sampler import SystemSampler
from aaco.collectors.rocm_smi_sampler import ROCmSMISampler
from aaco.collectors.clocks import ClockMonitor
from aaco.profiler.rocprof_wrap import RocprofWrapper, RocprofConfig
from aaco.profiler.rocprof_parse import RocprofParser
from aaco.analytics.metrics import DerivedMetricsEngine
from aaco.analytics.classify import BottleneckClassifier
from aaco.analytics.diff import RegressionDetector, diff_sessions


# Configure logging
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    AMD AI Compute Observatory (AACO)
    
    Principal-level performance observability for AMD AI workloads.
    Full-stack profiling from kernel launches to inference latency.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--backend", "-b", default="migraphx", 
              type=click.Choice(["migraphx", "rocm", "cuda", "cpu"]),
              help="Execution provider backend")
@click.option("--warmup", "-w", default=10, help="Warmup iterations")
@click.option("--iterations", "-n", default=100, help="Measurement iterations")
@click.option("--batch-size", default=1, help="Inference batch size")
@click.option("--output", "-o", default="./aaco_sessions", help="Output directory")
@click.option("--tag", "-t", default=None, help="Session tag for identification")
@click.option("--profile/--no-profile", default=False, help="Enable rocprof kernel profiling")
@click.option("--telemetry/--no-telemetry", default=True, help="Collect GPU/system telemetry")
@click.pass_context
def run(
    ctx: click.Context,
    model_path: str,
    backend: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    output: str,
    tag: Optional[str],
    profile: bool,
    telemetry: bool,
) -> None:
    """
    Run benchmark session on an ONNX model.
    
    Executes warmup and measurement phases, collecting:
    - Per-iteration latency metrics
    - GPU kernel traces (if --profile)
    - System/GPU telemetry
    """
    logger = logging.getLogger("aaco.cli.run")
    
    click.echo(click.style("\n╔══════════════════════════════════════════════╗", fg="cyan"))
    click.echo(click.style("║   AMD AI Compute Observatory - Benchmark     ║", fg="cyan"))
    click.echo(click.style("╚══════════════════════════════════════════════╝\n", fg="cyan"))
    
    # Initialize session
    session_mgr = SessionManager(base_dir=Path(output))
    session = session_mgr.create_session(tag=tag)
    
    logger.info(f"Session ID: {session.session_id}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Backend: {backend}")
    
    # Save environment
    session.save_environment_lockbox()
    
    # Initialize components
    model_path_obj = Path(model_path)
    run_config = RunConfig(
        backend=backend,
        warmup_iterations=warmup,
        measurement_iterations=iterations,
        batch_size=batch_size,
    )
    
    # Start telemetry collectors
    sys_sampler = None
    gpu_sampler = None
    
    if telemetry:
        logger.info("Starting telemetry collectors...")
        sys_sampler = SystemSampler(interval_ms=100)
        gpu_sampler = ROCmSMISampler(interval_ms=200)
        sys_sampler.start()
        gpu_sampler.start()
    
    # Record clock/governor state
    clock_monitor = ClockMonitor()
    clock_state = clock_monitor.get_clock_summary()
    logger.info(f"Clock state: {clock_state}")
    
    try:
        # Initialize runner
        runner = ORTRunner(model_path_obj, config=run_config)
        
        click.echo(f"\nRunning {warmup} warmup + {iterations} measurement iterations...")
        
        # Run benchmark
        results = runner.run_benchmark()
        
        # Stop telemetry
        if sys_sampler:
            sys_sampler.stop()
        if gpu_sampler:
            gpu_sampler.stop()
        
        # Compute metrics
        logger.info("Computing derived metrics...")
        metrics_engine = DerivedMetricsEngine()
        metrics_engine.add_inference_results(results)
        
        if sys_sampler:
            metrics_engine.add_system_samples(sys_sampler.get_samples())
        if gpu_sampler:
            metrics_engine.add_gpu_samples(gpu_sampler.get_samples())
        
        derived_metrics = metrics_engine.compute()
        
        # Kernel profiling (if enabled)
        kernel_metrics = None
        if profile:
            logger.info("Kernel profiling requested - run separately with `aaco profile`")
        
        # Classify bottlenecks
        classifier = BottleneckClassifier()
        bottleneck = classifier.classify(
            metrics=derived_metrics,
            kernel_metrics=kernel_metrics,
        )
        
        # Save artifacts
        session.save_artifact("inference_results.json", [r.__dict__ for r in results])
        session.save_artifact("derived_metrics.json", metrics_engine.summary_dict())
        session.save_artifact("bottleneck.json", bottleneck.__dict__)
        session.save_artifact("clock_state.json", clock_state)
        
        if sys_sampler:
            session.save_artifact("sys_samples.json", [s.__dict__ for s in sys_sampler.get_samples()])
        if gpu_sampler:
            session.save_artifact("gpu_samples.json", [s.__dict__ for s in gpu_sampler.get_samples()])
        
        # Display results
        click.echo(click.style("\n═══ Results ═══", fg="green", bold=True))
        click.echo(f"Mean Latency:    {derived_metrics.measurement_phase.mean_ms:.3f} ms")
        click.echo(f"Std Dev:         {derived_metrics.measurement_phase.std_ms:.3f} ms")
        click.echo(f"P50/P99:         {derived_metrics.measurement_phase.p50_ms:.3f} / {derived_metrics.measurement_phase.p99_ms:.3f} ms")
        click.echo(f"Throughput:      {derived_metrics.throughput.get('inferences_per_sec', 0):.1f} infer/sec")
        click.echo(f"\nBottleneck:      {bottleneck.primary.upper()} (conf: {bottleneck.confidence:.0%})")
        
        if bottleneck.evidence:
            click.echo("\nEvidence:")
            for ev in bottleneck.evidence[:5]:
                click.echo(f"  • {ev}")
        
        click.echo(click.style(f"\n✓ Session saved: {session.path}", fg="green"))
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        sys.exit(1)
    
    finally:
        # Ensure telemetry is stopped
        if sys_sampler:
            sys_sampler.stop()
        if gpu_sampler:
            gpu_sampler.stop()


@cli.command()
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("current", type=click.Path(exists=True))
@click.option("--threshold", "-t", default=5.0, help="Regression threshold (%)")
@click.option("--output", "-o", default=None, help="Output report path")
def diff(
    baseline: str,
    current: str,
    threshold: float,
    output: Optional[str],
) -> None:
    """
    Compare two sessions for regressions.
    
    Performs statistical comparison and identifies significant changes.
    """
    click.echo(click.style("\n═══ Session Comparison ═══", fg="cyan", bold=True))
    click.echo(f"Baseline: {baseline}")
    click.echo(f"Current:  {current}")
    
    verdict = diff_sessions(
        baseline,
        current,
        thresholds={"latency_regression_pct": threshold},
    )
    
    # Display verdict
    verdict_color = {
        "REGRESSION": "red",
        "IMPROVEMENT": "green",
        "NEUTRAL": "yellow",
    }.get(verdict.verdict, "white")
    
    click.echo(click.style(f"\nVerdict: {verdict.verdict}", fg=verdict_color, bold=True))
    click.echo(f"Confidence: {verdict.confidence:.0%}")
    
    if verdict.p_value is not None:
        click.echo(f"Statistical significance (p-value): {verdict.p_value:.4f}")
    
    click.echo(f"\n{verdict.summary}")
    
    if output:
        with open(output, 'w') as f:
            json.dump(verdict.__dict__, f, indent=2, default=str)
        click.echo(f"\n✓ Report saved: {output}")


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", default="terminal",
              type=click.Choice(["terminal", "html", "json"]),
              help="Output format")
@click.option("--output", "-o", default=None, help="Output file path")
def report(
    session_path: str,
    fmt: str,
    output: Optional[str],
) -> None:
    """
    Generate report from a session bundle.
    
    Supports terminal, HTML, and JSON output formats.
    """
    from aaco.report.render import ReportRenderer
    
    session = Path(session_path)
    renderer = ReportRenderer(session)
    
    if fmt == "terminal":
        report_text = renderer.render_terminal()
        click.echo(report_text)
    elif fmt == "html":
        html = renderer.render_html()
        if output:
            Path(output).write_text(html)
            click.echo(f"✓ HTML report saved: {output}")
        else:
            click.echo(html)
    elif fmt == "json":
        data = renderer.render_json()
        if output:
            Path(output).write_text(data)
            click.echo(f"✓ JSON report saved: {output}")
        else:
            click.echo(data)


@cli.command()
@click.option("--output", "-o", default="./aaco_sessions", help="Sessions directory")
@click.option("--limit", "-n", default=10, help="Number of sessions to list")
def ls(output: str, limit: int) -> None:
    """
    List recent benchmark sessions.
    """
    sessions_dir = Path(output)
    
    if not sessions_dir.exists():
        click.echo("No sessions found.")
        return
    
    sessions = sorted(sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    
    click.echo(click.style("\n═══ Recent Sessions ═══", fg="cyan", bold=True))
    click.echo(f"{'Session ID':<40} {'Date':<20} {'Tag':<15}")
    click.echo("─" * 75)
    
    for session in sessions[:limit]:
        if not session.is_dir():
            continue
        
        meta_file = session / "session_meta.json"
        tag = "-"
        date_str = datetime.fromtimestamp(session.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    tag = meta.get("tag", "-") or "-"
            except:
                pass
        
        click.echo(f"{session.name:<40} {date_str:<20} {tag:<15}")


@cli.command()
def info() -> None:
    """
    Display system information and tool availability.
    """
    import platform
    
    click.echo(click.style("\n═══ AACO System Information ═══", fg="cyan", bold=True))
    
    # Python
    click.echo(f"\nPython:    {platform.python_version()}")
    click.echo(f"Platform:  {platform.platform()}")
    
    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        click.echo(f"ONNXRuntime: {ort.__version__}")
        providers = ort.get_available_providers()
        click.echo(f"  Providers: {', '.join(providers)}")
    except ImportError:
        click.echo("ONNXRuntime: NOT INSTALLED")
    
    # Check rocm-smi
    from aaco.core.utils import run_command
    rocm_version = run_command(["rocm-smi", "--version"])
    if rocm_version:
        click.echo(f"rocm-smi:  Available")
    else:
        click.echo("rocm-smi:  NOT AVAILABLE")
    
    # Check rocprof
    rocprof_version = run_command(["rocprof", "--version"])
    if rocprof_version:
        click.echo(f"rocprof:   Available")
    else:
        click.echo("rocprof:   NOT AVAILABLE")
    
    # GPU info
    gpu_sampler = ROCmSMISampler()
    gpu_info = gpu_sampler._get_gpu_info()
    if gpu_info:
        click.echo(f"\nGPU:       {gpu_info.get('name', 'Unknown')}")
        click.echo(f"  VRAM:    {gpu_info.get('vram_total_mb', 0)} MB")
    else:
        click.echo("\nGPU:       NOT DETECTED")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--backend", "-b", default="migraphx",
              type=click.Choice(["migraphx", "rocm", "cuda", "cpu"]),
              help="Execution provider backend")
@click.option("--batches", "-B", default="1,2,4,8,16,32",
              help="Comma-separated batch sizes to sweep")
@click.option("--warmup", "-w", default=5, help="Warmup iterations per batch size")
@click.option("--iterations", "-n", default=50, help="Measurement iterations per batch size")
@click.option("--output", "-o", default="./aaco_sessions", help="Output directory")
@click.option("--tag", "-t", default=None, help="Session tag for identification")
@click.pass_context
def sweep(
    ctx: click.Context,
    model_path: str,
    backend: str,
    batches: str,
    warmup: int,
    iterations: int,
    output: str,
    tag: Optional[str],
) -> None:
    """
    Run batch size sweep to analyze scaling characteristics.
    
    Runs benchmark at multiple batch sizes and generates scaling analysis:
    - Throughput vs batch size curve
    - Latency vs batch size curve  
    - Saturation point detection
    - Optimal batch size recommendation
    """
    from aaco.analytics.batch_scaler import BatchScalingAnalyzer, BatchPoint
    
    logger = logging.getLogger("aaco.cli.sweep")
    
    click.echo(click.style("\n╔══════════════════════════════════════════════╗", fg="magenta"))
    click.echo(click.style("║   AMD AI Compute Observatory - Batch Sweep   ║", fg="magenta"))
    click.echo(click.style("╚══════════════════════════════════════════════╝\n", fg="magenta"))
    
    # Parse batch sizes
    try:
        batch_sizes = [int(b.strip()) for b in batches.split(",")]
    except ValueError:
        click.echo(click.style("Error: Invalid batch sizes format", fg="red"))
        sys.exit(1)
    
    click.echo(f"Model:      {model_path}")
    click.echo(f"Backend:    {backend}")
    click.echo(f"Batches:    {batch_sizes}")
    click.echo(f"Iterations: {warmup} warmup + {iterations} measurement per batch")
    click.echo("")
    
    # Initialize session
    session_mgr = SessionManager(base_dir=Path(output))
    sweep_tag = tag or f"sweep_{datetime.now().strftime('%H%M%S')}"
    session = session_mgr.create_session(tag=sweep_tag)
    
    logger.info(f"Sweep session: {session.session_id}")
    
    # Collect batch points
    batch_points: List[Dict[str, Any]] = []
    
    model_path_obj = Path(model_path)
    
    click.echo("Running batch sweep...")
    click.echo("─" * 60)
    
    for batch_size in batch_sizes:
        click.echo(f"\n  Batch size {batch_size:>4}:", nl=False)
        
        try:
            run_config = RunConfig(
                backend=backend,
                warmup_iterations=warmup,
                measurement_iterations=iterations,
                batch_size=batch_size,
            )
            
            runner = ORTRunner(model_path_obj, config=run_config)
            results = runner.run_benchmark()
            
            # Extract stats
            latencies = [r.latency_ms for r in results if r.is_measurement]
            if not latencies:
                click.echo(click.style(" FAILED (no data)", fg="red"))
                continue
            
            import numpy as np
            mean_latency_ms = float(np.mean(latencies))
            throughput_ips = (1000.0 / mean_latency_ms) * batch_size
            
            batch_points.append({
                "batch_size": batch_size,
                "mean_latency_ms": mean_latency_ms,
                "std_ms": float(np.std(latencies)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "throughput_ips": throughput_ips,
                "latencies": latencies,
            })
            
            click.echo(click.style(f" {mean_latency_ms:>8.2f}ms  {throughput_ips:>8.1f} infer/s", fg="green"))
            
        except Exception as e:
            logger.error(f"Batch {batch_size} failed: {e}")
            click.echo(click.style(f" ERROR: {e}", fg="red"))
    
    click.echo("\n" + "─" * 60)
    
    if len(batch_points) < 2:
        click.echo(click.style("\nInsufficient data for scaling analysis (need >= 2 batch sizes)", fg="yellow"))
        sys.exit(1)
    
    # Run scaling analysis
    click.echo(click.style("\n═══ Scaling Analysis ═══", fg="cyan", bold=True))
    
    analyzer = BatchScalingAnalyzer()
    
    for bp in batch_points:
        analyzer.add_batch_point(BatchPoint(
            batch_size=bp["batch_size"],
            mean_latency_ms=bp["mean_latency_ms"],
            std_latency_ms=bp["std_ms"],
            throughput_ips=bp["throughput_ips"],
            memory_mb=0,  # Would need actual measurement
            latencies=bp["latencies"],
        ))
    
    analysis = analyzer.analyze()
    
    # Display results
    click.echo(f"\nScaling Efficiency:   {analysis.scaling_efficiency*100:.1f}%")
    click.echo(f"Saturation Detected:  {'Yes' if analysis.saturation_detected else 'No'}")
    
    if analysis.saturation_batch:
        click.echo(f"Saturation Point:     Batch {analysis.saturation_batch}")
    
    click.echo(f"Memory Bound:         {'Yes' if analysis.memory_bound else 'No'}")
    click.echo(f"Optimal Batch Size:   {analysis.optimal_batch}")
    
    # Save artifacts
    session.save_artifact("sweep_points.json", batch_points)
    session.save_artifact("scaling_analysis.json", {
        "scaling_efficiency": analysis.scaling_efficiency,
        "saturation_detected": analysis.saturation_detected,
        "saturation_batch": analysis.saturation_batch,
        "memory_bound": analysis.memory_bound,
        "optimal_batch": analysis.optimal_batch,
        "throughput_curve": analysis.throughput_curve,
        "latency_curve": analysis.latency_curve,
    })
    
    # Generate plot if matplotlib available
    try:
        plot_path = session.path / "scaling_curves.png"
        analyzer.plot_scaling_curves(str(plot_path))
        click.echo(click.style(f"\n✓ Scaling plot saved: {plot_path}", fg="green"))
    except Exception:
        pass
    
    click.echo(click.style(f"✓ Sweep session saved: {session.path}", fg="green"))


@cli.command()
@click.option("--port", "-p", default=8501, help="Streamlit port")
@click.option("--sessions-dir", "-d", default="./aaco_sessions", help="Sessions directory")
def dashboard(port: int, sessions_dir: str) -> None:
    """
    Launch interactive Streamlit dashboard.
    
    Opens a web-based UI for browsing sessions, viewing metrics,
    and comparing benchmark results.
    """
    import subprocess
    
    click.echo(click.style("\n═══ AACO Dashboard ═══", fg="cyan", bold=True))
    click.echo(f"Starting Streamlit dashboard on port {port}...")
    click.echo(f"Sessions directory: {sessions_dir}")
    
    # Check if streamlit is available
    try:
        import streamlit
        click.echo(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        click.echo(click.style("\nError: Streamlit not installed", fg="red"))
        click.echo("Install with: pip install streamlit")
        sys.exit(1)
    
    # Get dashboard app path
    from pathlib import Path
    import aaco
    
    dashboard_path = Path(aaco.__file__).parent / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        click.echo(click.style(f"\nError: Dashboard app not found at {dashboard_path}", fg="red"))
        sys.exit(1)
    
    click.echo(f"\nOpen in browser: http://localhost:{port}")
    click.echo("Press Ctrl+C to stop\n")
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--",
            "--sessions-dir", sessions_dir,
        ])
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
