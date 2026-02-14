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


# ============================================================================
# Advanced AACO-X Commands
# ============================================================================

@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output JSON path")
def chi(session_path: str, output: Optional[str]) -> None:
    """
    Compute Health Index analysis.
    
    Calculates the Compute Health Index (CHI) for a session,
    providing a composite score [0-100] of overall system health.
    """
    from aaco.analytics.feature_store import extract_and_store_features
    from aaco.analytics.chi import CHICalculator, get_chi_badge
    
    session = Path(session_path)
    
    click.echo(click.style("\n═══ Compute Health Index ═══", fg="cyan", bold=True))
    
    # Extract features
    session_features, _ = extract_and_store_features(session)
    
    # Calculate CHI
    calculator = CHICalculator()
    report = calculator.calculate(session_features)
    
    # Display results
    badge = get_chi_badge(report.chi_score)
    click.echo(f"\n{badge}")
    click.echo(f"CHI Score: {report.chi_score:.0f}/100")
    click.echo(f"Rating:    {report.rating.value.upper()}")
    
    click.echo(click.style("\n── Component Breakdown ──", fg="yellow"))
    for comp in report.components:
        bar_len = int(comp.score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        click.echo(f"  {comp.name:<15} [{bar}] {comp.score:.0%}")
    
    if report.strengths:
        click.echo(click.style("\n✓ Strengths:", fg="green"))
        for s in report.strengths:
            click.echo(f"  • {s}")
    
    if report.weaknesses:
        click.echo(click.style("\n⚠ Weaknesses:", fg="yellow"))
        for w in report.weaknesses:
            click.echo(f"  • {w}")
    
    click.echo(f"\n{report.summary}")
    
    if output:
        with open(output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        click.echo(f"\n✓ Report saved: {output}")


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--model-name", "-m", default=None, help="Model name for baseline lookup")
@click.option("--baseline-dir", "-b", default="./aaco_sessions", help="Baseline sessions directory")
@click.option("--min-sessions", default=3, help="Minimum baseline sessions required")
@click.option("--output", "-o", default=None, help="Output JSON path")
def regression(
    session_path: str,
    model_name: Optional[str],
    baseline_dir: str,
    min_sessions: int,
    output: Optional[str],
) -> None:
    """
    Statistical regression analysis against historical baseline.
    
    Compares current session metrics against baseline model derived
    from historical sessions for the same model.
    """
    from aaco.analytics.feature_store import FeatureStore, extract_and_store_features
    from aaco.analytics.regression_guard import RegressionGuard, RegressionSeverity, get_regression_exit_code
    
    session = Path(session_path)
    
    click.echo(click.style("\n═══ Regression Analysis ═══", fg="cyan", bold=True))
    
    # Load/build feature store from baseline directory
    store = FeatureStore(Path(baseline_dir))
    
    # Extract current session features
    current_features, _ = extract_and_store_features(session, store=None)
    
    if model_name:
        current_features.model_name = model_name
    
    click.echo(f"Session:  {session.name}")
    click.echo(f"Model:    {current_features.model_name or '(unknown)'}")
    
    # Run regression analysis
    guard = RegressionGuard(feature_store=store, min_baseline_sessions=min_sessions)
    report = guard.check_regression(current_features)
    
    if report.baseline_session_count < min_sessions:
        click.echo(click.style(f"\n⚠ Insufficient baseline data", fg="yellow"))
        click.echo(f"  Need {min_sessions} sessions, have {report.baseline_session_count}")
        click.echo("  Run more benchmarks to build baseline history")
        return
    
    click.echo(f"Baseline: {report.baseline_session_count} sessions")
    
    # Display verdict
    if report.has_regression:
        severity_color = {
            RegressionSeverity.MILD: "yellow",
            RegressionSeverity.MODERATE: "yellow",
            RegressionSeverity.SEVERE: "red",
            RegressionSeverity.CRITICAL: "red",
        }.get(report.overall_severity, "white")
        
        click.echo(click.style(f"\n⚠ REGRESSION DETECTED", fg=severity_color, bold=True))
        click.echo(f"Severity:   {report.overall_severity.value.upper()}")
        click.echo(f"Confidence: {report.overall_confidence:.0%}")
        
        click.echo(click.style("\n── Top Regressions ──", fg="yellow"))
        for r in report.top_regressions[:5]:
            direction = "↑" if r.z_score > 0 else "↓"
            click.echo(f"  {r.metric_name:<20} {direction} {r.percent_change:+.1f}% (Z={r.z_score:.2f})")
        
        if report.recommendations:
            click.echo(click.style("\n── Recommendations ──", fg="cyan"))
            for rec in report.recommendations[:3]:
                click.echo(f"  • {rec}")
    else:
        click.echo(click.style("\n✓ NO REGRESSION DETECTED", fg="green", bold=True))
    
    click.echo(f"\n{report.summary}")
    
    if output:
        with open(output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        click.echo(f"\n✓ Report saved: {output}")
    
    # Return exit code for CI/CD
    exit_code = get_regression_exit_code(report)
    if exit_code > 2:  # Severe or critical
        sys.exit(exit_code)


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output JSON path")
def analyze(session_path: str, output: Optional[str]) -> None:
    """
    Full bottleneck analysis with recommendations.
    
    Performs comprehensive analysis to identify performance bottlenecks
    and generates evidence-based optimization recommendations.
    """
    from aaco.analytics.feature_store import extract_and_store_features
    from aaco.analytics.recommendation_engine import RecommendationEngine
    from aaco.analytics.chi import CHICalculator
    
    session = Path(session_path)
    
    click.echo(click.style("\n═══ Performance Analysis ═══", fg="cyan", bold=True))
    
    # Extract features
    session_features, iteration_features = extract_and_store_features(session)
    
    # Run recommendation engine
    engine = RecommendationEngine()
    analysis = engine.analyze(session_features, iteration_features)
    
    # Calculate CHI
    chi_calc = CHICalculator()
    chi_report = chi_calc.calculate(session_features)
    
    # Display CHI
    click.echo(f"\nCompute Health Index: {chi_report.chi_score:.0f}/100 ({chi_report.rating.value})")
    
    # Display primary bottleneck
    if analysis.primary_bottleneck:
        bn = analysis.primary_bottleneck.value.replace("_", " ").title()
        click.echo(f"Primary Bottleneck:   {bn}")
    
    # Top drivers
    if analysis.top_drivers:
        click.echo(click.style("\n── Top Performance Drivers ──", fg="yellow"))
        for i, (driver, impact) in enumerate(analysis.top_drivers, 1):
            bar_len = int(impact * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            click.echo(f"  {i}. {driver:<25} [{bar}] {impact:.0%}")
    
    # Recommendations
    if analysis.recommendations:
        click.echo(click.style("\n── Recommendations ──", fg="cyan"))
        for rec in analysis.recommendations[:5]:
            priority_color = {
                "critical": "red",
                "high": "yellow",
                "medium": "white",
                "low": "bright_black",
            }.get(rec.priority.value, "white")
            
            click.echo(click.style(f"\n[{rec.priority.value.upper()}] {rec.title}", fg=priority_color, bold=True))
            click.echo(f"  {rec.description}")
            
            if rec.actions:
                click.echo("  Actions:")
                for action in rec.actions[:3]:
                    click.echo(f"    • {action}")
            
            if rec.expected_improvement:
                click.echo(f"  Expected improvement: {rec.expected_improvement}")
    
    # Summary
    summary = engine.get_summary(analysis)
    click.echo(f"\n{summary}")
    
    if output:
        with open(output, 'w') as f:
            json.dump({
                "chi": chi_report.to_dict(),
                "analysis": analysis.to_dict(),
            }, f, indent=2, default=str)
        click.echo(f"\n✓ Analysis saved: {output}")


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output trace path")
@click.option("--compress", "-z", is_flag=True, help="Compress output (gzip)")
@click.option("--open", "open_ui", is_flag=True, help="Open in Perfetto UI")
def perfetto(
    session_path: str,
    output: Optional[str],
    compress: bool,
    open_ui: bool,
) -> None:
    """
    Export session to Perfetto trace format.
    
    Creates a Perfetto-compatible JSON trace that can be visualized
    in the Perfetto UI (https://ui.perfetto.dev).
    """
    from aaco.analytics.perfetto_export import AACOSessionToPerfetto, open_in_perfetto
    
    session = Path(session_path)
    
    click.echo(click.style("\n═══ Perfetto Export ═══", fg="cyan", bold=True))
    click.echo(f"Session: {session.name}")
    
    # Convert session
    converter = AACOSessionToPerfetto(session)
    builder = converter.convert()
    
    # Determine output path
    if output:
        out_path = Path(output)
    else:
        out_path = session / "trace.perfetto.json"
    
    # Save trace
    builder.save_json(out_path, compress=compress)
    
    event_count = len(builder._events)
    click.echo(f"Events exported: {event_count}")
    click.echo(click.style(f"\n✓ Trace saved: {out_path}", fg="green"))
    
    if open_ui:
        click.echo("\nOpening Perfetto UI...")
        open_in_perfetto(out_path)
    else:
        click.echo("\nTo visualize:")
        click.echo(f"  1. Open https://ui.perfetto.dev")
        click.echo(f"  2. Drag and drop: {out_path}")


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--store-dir", "-s", default="./aaco_feature_store", help="Feature store directory")
@click.option("--export", "-e", default=None, help="Export features to JSON")
def features(
    session_path: str,
    store_dir: str,
    export: Optional[str],
) -> None:
    """
    Extract and store session features.
    
    Extracts performance features from a session and stores them
    in the feature store for baseline comparison and trend analysis.
    """
    from aaco.analytics.feature_store import FeatureStore, extract_and_store_features
    
    session = Path(session_path)
    store_path = Path(store_dir)
    
    click.echo(click.style("\n═══ Feature Extraction ═══", fg="cyan", bold=True))
    click.echo(f"Session: {session.name}")
    
    # Load or create feature store
    store = FeatureStore(store_path)
    
    # Extract features
    session_features, iteration_features = extract_and_store_features(session, store)
    
    # Display key features
    click.echo(click.style("\n── Session Features ──", fg="yellow"))
    click.echo(f"  Latency (mean):    {session_features.latency_mean_ms:.3f} ms")
    click.echo(f"  Latency (std):     {session_features.latency_std_ms:.3f} ms")
    click.echo(f"  Throughput:        {session_features.throughput_its:.1f} infer/s")
    click.echo(f"  CV%:               {session_features.cv_pct:.1f}%")
    click.echo(f"  Spike ratio:       {session_features.spike_ratio:.1%}")
    click.echo(f"  KAR:               {session_features.kar:.3f}")
    click.echo(f"  CHI Score:         {session_features.chi_score:.0f}/100")
    
    click.echo(f"\n  Iterations:        {len(iteration_features)}")
    click.echo(f"  Spikes detected:   {session_features.spike_count}")
    
    # Save feature store
    store.save()
    click.echo(click.style(f"\n✓ Features saved to: {store_path}", fg="green"))
    
    if export:
        store.export_json(Path(export))
        click.echo(f"✓ Exported to: {export}")


# =============================================================================
# AACO-Λ Advanced Commands
# =============================================================================

@cli.command()
@click.option("--gpu-id", "-g", default=0, help="GPU ID to calibrate")
@click.option("--output", "-o", default=None, help="Output JSON path")
@click.option("--force", "-f", is_flag=True, help="Force recalibration")
def calibrate(gpu_id: int, output: Optional[str], force: bool) -> None:
    """
    Calibrate hardware envelope for GPU.
    
    Measures peak capabilities (bandwidth, compute, launch overhead)
    to establish baseline for efficiency calculations.
    """
    from aaco.calibration.envelope import HardwareEnvelopeCalibrator, load_or_calibrate
    
    click.echo(click.style("\n═══ Hardware Envelope Calibration ═══", fg="cyan", bold=True))
    click.echo(f"GPU ID: {gpu_id}")
    
    if output:
        out_path = Path(output)
    else:
        out_path = Path.home() / ".aaco" / f"hardware_envelope_gpu{gpu_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not force and out_path.exists():
        envelope = load_or_calibrate(out_path, gpu_id, max_age_hours=24)
        click.echo(f"Using cached envelope (age: {(time.time() - envelope.timestamp_utc) / 3600:.1f}h)")
    else:
        calibrator = HardwareEnvelopeCalibrator(gpu_id)
        envelope = calibrator.calibrate()
        envelope.save(out_path)
    
    click.echo(click.style("\n── Hardware Envelope ──", fg="yellow"))
    click.echo(f"  GPU:                {envelope.gpu_name}")
    click.echo(f"  Memory:             {envelope.memory_total_gb:.1f} GB")
    click.echo(f"  Compute Units:      {envelope.compute_units}")
    click.echo(f"\n  Peak BW:            {envelope.bandwidth.peak_bandwidth_gbps:.0f} GB/s")
    click.echo(f"  Read BW:            {envelope.bandwidth.read_bandwidth_gbps:.0f} GB/s ({envelope.bandwidth.read_efficiency:.0%})")
    click.echo(f"  Peak Compute:       {envelope.compute.peak_tflops_fp32:.1f} TFLOPS (FP32)")
    click.echo(f"  GEMM Throughput:    {envelope.compute.gemm_tflops_fp32:.1f} TFLOPS ({envelope.compute.fp32_efficiency:.0%})")
    click.echo(f"  Launch Overhead:    {envelope.launch.launch_p50_us:.1f} μs (p50)")
    
    if envelope.calibration_warnings:
        click.echo(click.style("\n⚠ Warnings:", fg="yellow"))
        for w in envelope.calibration_warnings:
            click.echo(f"  • {w}")
    
    click.echo(click.style(f"\n✓ Envelope saved: {out_path}", fg="green"))


@cli.command(name="root-cause")
@click.argument("metrics_json", type=click.Path(exists=True), required=False)
@click.option("--memory-bw", "-m", type=float, help="Memory BW utilization (0-1)")
@click.option("--compute", "-c", type=float, help="Compute utilization (0-1)")
@click.option("--launch-pct", "-l", type=float, help="Launch overhead percentage")
@click.option("--occupancy", "-o", type=float, help="GPU occupancy (0-1)")
@click.option("--output", default=None, help="Output JSON path")
def root_cause(
    metrics_json: Optional[str],
    memory_bw: Optional[float],
    compute: Optional[float],
    launch_pct: Optional[float],
    occupancy: Optional[float],
    output: Optional[str],
) -> None:
    """
    Analyze root cause of performance issues.
    
    Uses Bayesian inference to rank potential performance bottlenecks
    based on provided metrics or a metrics JSON file.
    """
    from aaco.analytics.root_cause import BayesianRootCauseAnalyzer, explain_root_cause
    
    click.echo(click.style("\n═══ Root Cause Analysis ═══", fg="cyan", bold=True))
    
    # Collect metrics
    metrics: Dict[str, float] = {}
    
    if metrics_json:
        with open(metrics_json) as f:
            metrics = json.load(f)
        click.echo(f"Loaded metrics from: {metrics_json}")
    
    # Override with CLI options
    if memory_bw is not None:
        metrics["memory_bandwidth_utilization"] = memory_bw
    if compute is not None:
        metrics["compute_utilization"] = compute
    if launch_pct is not None:
        metrics["launch_overhead_pct"] = launch_pct
    if occupancy is not None:
        metrics["occupancy"] = occupancy
    
    if not metrics:
        click.echo(click.style("No metrics provided. Use --help for usage.", fg="red"))
        return
    
    click.echo(f"\nMetrics: {len(metrics)} observations")
    
    # Run analysis
    analyzer = BayesianRootCauseAnalyzer()
    analyzer.add_observations(metrics)
    ranking = analyzer.analyze()
    
    # Display results
    explanation = explain_root_cause(ranking)
    click.echo("\n" + explanation)
    
    if output:
        ranking.save(Path(output))
        click.echo(f"\n✓ Analysis saved: {output}")


@cli.command()
@click.option("--db-path", "-d", default=None, help="Warehouse database path")
@click.option("--model", "-m", default=None, help="Filter by model name")
@click.option("--backend", "-b", default=None, help="Filter by backend")
@click.option("--metric", default="latency_p50_ms", help="Metric to query")
@click.option("--days", default=7, help="Days of history")
@click.option("--action", type=click.Choice(["stats", "trend", "compare", "sessions"]),
              default="stats", help="Action to perform")
def warehouse(
    db_path: Optional[str],
    model: Optional[str],
    backend: Optional[str],
    metric: str,
    days: int,
    action: str,
) -> None:
    """
    Query the fleet-scale performance warehouse.
    
    Actions:
      stats    - Get aggregate statistics
      trend    - Show trend over time
      compare  - Compare backends
      sessions - List recent sessions
    """
    from aaco.warehouse.store import FleetWarehouse, get_default_warehouse
    
    click.echo(click.style("\n═══ Fleet Warehouse ═══", fg="cyan", bold=True))
    
    if db_path:
        wh = FleetWarehouse(Path(db_path))
    else:
        wh = get_default_warehouse()
    
    stats = wh.get_stats()
    click.echo(f"Database: {wh.db_path}")
    click.echo(f"Sessions: {stats['sessions']}, Results: {stats['results']}")
    
    if action == "stats":
        metric_stats = wh.get_metric_stats(metric, model, backend, days)
        click.echo(click.style(f"\n── {metric} Statistics ({days}d) ──", fg="yellow"))
        click.echo(f"  Mean:  {metric_stats['mean']:.3f}")
        click.echo(f"  Min:   {metric_stats['min']:.3f}")
        click.echo(f"  Max:   {metric_stats['max']:.3f}")
        click.echo(f"  Count: {metric_stats['count']}")
    
    elif action == "trend":
        trend = wh.get_trend(metric, model, backend, None, days)
        click.echo(click.style(f"\n── {metric} Trend ({days}d) ──", fg="yellow"))
        if trend:
            for point in trend[-10:]:
                ts = datetime.fromtimestamp(point.timestamp).strftime("%Y-%m-%d %H:%M")
                click.echo(f"  {ts}  {point.value:.3f}")
        else:
            click.echo("  No data found")
    
    elif action == "compare":
        if not model:
            click.echo(click.style("--model required for compare", fg="red"))
            return
        comparison = wh.compare_backends(metric, model, days)
        click.echo(click.style(f"\n── {metric} by Backend ({model}) ──", fg="yellow"))
        for b, s in comparison.items():
            click.echo(f"  {b:<15} mean={s['mean']:.3f} (n={s['count']})")
    
    elif action == "sessions":
        sessions = wh.list_sessions(model, backend, None, days, limit=10)
        click.echo(click.style(f"\n── Recent Sessions ({days}d) ──", fg="yellow"))
        for s in sessions:
            ts = datetime.fromtimestamp(s.timestamp_utc).strftime("%Y-%m-%d %H:%M")
            click.echo(f"  {ts}  {s.model_name:<20} {s.backend:<12} {s.gpu_name}")


@cli.command()
@click.option("--level", "-l", 
              type=click.Choice(["none", "basic", "standard", "strict"]),
              default="standard", help="Isolation level")
@click.option("--governor", "-g", default="performance", help="CPU governor")
@click.option("--gpu-clocks", default="high", help="GPU clock policy")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def capsule(level: str, governor: str, gpu_clocks: str, quiet: bool) -> None:
    """
    Enter a measurement capsule for deterministic benchmarking.
    
    Sets up cgroups isolation, CPU governor, and GPU clocks
    for reproducible performance measurements.
    """
    from aaco.isolation.capsule import (
        MeasurementCapsule, CapsulePolicy, IsolationLevel, GovernorPolicy, GPUClockPolicy
    )
    
    if not quiet:
        click.echo(click.style("\n═══ Measurement Capsule ═══", fg="cyan", bold=True))
    
    # Map string to enum
    level_map = {
        "none": IsolationLevel.NONE,
        "basic": IsolationLevel.BASIC,
        "standard": IsolationLevel.STANDARD,
        "strict": IsolationLevel.STRICT,
    }
    
    gov_map = {
        "performance": GovernorPolicy.PERFORMANCE,
        "powersave": GovernorPolicy.POWERSAVE,
        "ondemand": GovernorPolicy.ONDEMAND,
    }
    
    clock_map = {
        "high": GPUClockPolicy.HIGH,
        "low": GPUClockPolicy.LOW,
        "auto": GPUClockPolicy.AUTO,
    }
    
    policy = CapsulePolicy(
        isolation_level=level_map.get(level, IsolationLevel.STANDARD),
        cpu_governor=gov_map.get(governor, GovernorPolicy.PERFORMANCE),
        gpu_clock_policy=clock_map.get(gpu_clocks, GPUClockPolicy.HIGH),
    )
    
    capsule_obj = MeasurementCapsule(policy)
    manifest = capsule_obj.enter()
    
    if not quiet:
        click.echo(f"Capsule ID:      {manifest.capsule_id}")
        click.echo(f"Isolation:       {policy.isolation_level.value}")
        click.echo(f"CPU Governor:    {policy.cpu_governor.value}")
        click.echo(f"GPU Clocks:      {policy.gpu_clock_policy.value}")
        click.echo(f"CPUs:            {manifest.cpu_cores_used}")
        
        if manifest.isolation_warnings:
            click.echo(click.style("\n⚠ Warnings:", fg="yellow"))
            for w in manifest.isolation_warnings:
                click.echo(f"  • {w}")
        
        if manifest.isolation_verified:
            click.echo(click.style("\n✓ Isolation verified", fg="green"))
        else:
            click.echo(click.style("\n⚠ Isolation not fully verified", fg="yellow"))
    
    # Save manifest
    manifest_path = Path(f".aaco_capsule_{manifest.capsule_id}.json")
    manifest.save(manifest_path)
    
    click.echo(f"\nManifest: {manifest_path}")
    click.echo("\nRun your benchmark, then 'aaco uncapsule' to restore state")


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True), required=False)
def uncapsule(manifest_path: Optional[str]) -> None:
    """
    Exit measurement capsule and restore original state.
    """
    from aaco.isolation.capsule import MeasurementCapsule, CapsuleManifest
    
    click.echo(click.style("\n═══ Exit Measurement Capsule ═══", fg="cyan", bold=True))
    
    # Find manifest
    if manifest_path:
        manifest_file = Path(manifest_path)
    else:
        # Find most recent capsule manifest
        manifests = list(Path(".").glob(".aaco_capsule_*.json"))
        if not manifests:
            click.echo(click.style("No capsule manifest found", fg="red"))
            return
        manifest_file = max(manifests, key=lambda p: p.stat().st_mtime)
    
    click.echo(f"Manifest: {manifest_file}")
    
    # Load and restore
    with open(manifest_file) as f:
        data = json.load(f)
    
    click.echo("Restoring original state...")
    
    # Simple restoration - set governors back to original
    original_governors = data.get("cpu_governors_before", {})
    for cpu_id, gov in original_governors.items():
        gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor")
        if gov_path.exists():
            try:
                gov_path.write_text(gov)
            except:
                pass
    
    # Cleanup manifest
    manifest_file.unlink()
    
    click.echo(click.style("✓ Capsule exited, state restored", fg="green"))


def main() -> None:
    """Main entry point."""
    import time  # For calibrate command
    cli(obj={})


if __name__ == "__main__":
    main()
