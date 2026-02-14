"""
AACO-SIGMA Forensic Bundle Exporter

Exports forensic bundles to various formats for sharing and analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, BinaryIO
from enum import Enum, auto
import json
import time
import os
import gzip
import tarfile
import io
import base64

from .bundle import (
    ForensicBundle,
    BundleMetadata,
    BundleSection,
    BundleVersion,
)


class ExportFormat(Enum):
    """Export format options."""
    JSON = "json"           # Plain JSON
    JSON_GZ = "json.gz"     # Gzipped JSON
    TARBALL = "tar.gz"      # Tar archive with separate files
    BUNDLE = "aaco"         # Custom AACO bundle format
    HTML = "html"           # HTML report with embedded data


@dataclass
class ExportResult:
    """Result of export operation."""
    
    success: bool = False
    format: ExportFormat = ExportFormat.JSON
    output_path: str = ""
    output_size_bytes: int = 0
    export_time_s: float = 0.0
    error_message: str = ""


class BundleExporter:
    """
    Exports forensic bundles to various formats.
    
    Supported formats:
    - JSON: Human-readable, good for debugging
    - JSON.GZ: Compressed JSON, good for storage
    - TARBALL: Archive with separate files per section
    - BUNDLE: Custom binary format
    - HTML: Self-contained report
    """
    
    def __init__(self):
        self.exporters = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.JSON_GZ: self._export_json_gz,
            ExportFormat.TARBALL: self._export_tarball,
            ExportFormat.BUNDLE: self._export_bundle,
            ExportFormat.HTML: self._export_html,
        }
    
    def export(
        self,
        bundle: ForensicBundle,
        output_path: str,
        format: ExportFormat = ExportFormat.JSON,
    ) -> ExportResult:
        """
        Export bundle to specified format.
        
        Args:
            bundle: Bundle to export
            output_path: Output file path
            format: Export format
            
        Returns:
            Export result
        """
        start_time = time.time()
        result = ExportResult(format=format, output_path=output_path)
        
        try:
            # Validate bundle first
            is_valid, errors = bundle.validate()
            if not is_valid:
                result.error_message = f"Invalid bundle: {'; '.join(errors)}"
                return result
            
            # Export
            if format in self.exporters:
                self.exporters[format](bundle, output_path)
                result.success = True
                result.output_size_bytes = os.path.getsize(output_path)
            else:
                result.error_message = f"Unsupported format: {format}"
        
        except Exception as e:
            result.error_message = str(e)
        
        result.export_time_s = time.time() - start_time
        return result
    
    def _bundle_to_full_dict(self, bundle: ForensicBundle) -> Dict[str, Any]:
        """Convert bundle to complete dictionary."""
        data = {
            "format_version": BundleVersion.CURRENT,
            "metadata": {
                "bundle_id": bundle.metadata.bundle_id,
                "name": bundle.metadata.name,
                "description": bundle.metadata.description,
                "format_version": bundle.metadata.format_version,
                "workload_id": bundle.metadata.workload_id,
                "model_name": bundle.metadata.model_name,
                "created_at": bundle.metadata.created_at,
                "capture_duration_s": bundle.metadata.capture_duration_s,
                "sections": bundle.metadata.sections,
                "tags": bundle.metadata.tags,
                "labels": bundle.metadata.labels,
                "created_by": bundle.metadata.created_by,
                "checksum": bundle.metadata.checksum,
            },
            "environment": {
                "hostname": bundle.environment.hostname,
                "os_name": bundle.environment.os_name,
                "os_version": bundle.environment.os_version,
                "kernel_version": bundle.environment.kernel_version,
                "cpu_model": bundle.environment.cpu_model,
                "cpu_cores": bundle.environment.cpu_cores,
                "memory_gb": bundle.environment.memory_gb,
                "gpu_count": bundle.environment.gpu_count,
                "gpu_models": bundle.environment.gpu_models,
                "gpu_driver_version": bundle.environment.gpu_driver_version,
                "rocm_version": bundle.environment.rocm_version,
                "hip_version": bundle.environment.hip_version,
                "python_version": bundle.environment.python_version,
                "pytorch_version": bundle.environment.pytorch_version,
                "custom_env": bundle.environment.custom_env,
            },
            "configuration": bundle.configuration,
            "traces": [
                {
                    "trace_id": t.trace_id,
                    "trace_type": t.trace_type,
                    "source_file": t.source_file,
                    "event_count": t.event_count,
                    "events": t.events,
                }
                for t in bundle.traces
            ],
            "counters": {
                "kernel_counters": bundle.counters.kernel_counters,
                "aggregate_counters": bundle.counters.aggregate_counters,
                "counter_descriptions": bundle.counters.counter_descriptions,
            },
            "metrics": {
                "kernel_metrics": bundle.metrics.kernel_metrics,
                "latency_ms": bundle.metrics.latency_ms,
                "throughput": bundle.metrics.throughput,
                "memory_peak_mb": bundle.metrics.memory_peak_mb,
                "compute_time_ms": bundle.metrics.compute_time_ms,
                "memory_time_ms": bundle.metrics.memory_time_ms,
                "overhead_time_ms": bundle.metrics.overhead_time_ms,
            },
            "graph_json": bundle.graph_json,
            "ir_data": bundle.ir_data,
            "logs": bundle.logs,
        }
        
        # Handle artifacts separately (binary data)
        if bundle.artifacts:
            data["artifacts"] = {
                name: base64.b64encode(data).decode()
                for name, data in bundle.artifacts.items()
            }
        
        return data
    
    def _export_json(self, bundle: ForensicBundle, output_path: str) -> None:
        """Export as plain JSON."""
        data = self._bundle_to_full_dict(bundle)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _export_json_gz(self, bundle: ForensicBundle, output_path: str) -> None:
        """Export as gzipped JSON."""
        data = self._bundle_to_full_dict(bundle)
        
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    
    def _export_tarball(self, bundle: ForensicBundle, output_path: str) -> None:
        """Export as tarball with separate files per section."""
        with tarfile.open(output_path, "w:gz") as tar:
            # Metadata
            self._add_json_to_tar(tar, "metadata.json", {
                "bundle_id": bundle.metadata.bundle_id,
                "name": bundle.metadata.name,
                "created_at": bundle.metadata.created_at,
                "sections": bundle.metadata.sections,
                "format_version": bundle.metadata.format_version,
            })
            
            # Environment
            self._add_json_to_tar(tar, "environment.json", {
                "hostname": bundle.environment.hostname,
                "gpu_models": bundle.environment.gpu_models,
                "rocm_version": bundle.environment.rocm_version,
            })
            
            # Configuration
            if bundle.configuration:
                self._add_json_to_tar(tar, "configuration.json", bundle.configuration)
            
            # Traces (each as separate file)
            for i, trace in enumerate(bundle.traces):
                self._add_json_to_tar(tar, f"traces/trace_{i}.json", {
                    "trace_id": trace.trace_id,
                    "trace_type": trace.trace_type,
                    "event_count": trace.event_count,
                    "events": trace.events,
                })
            
            # Counters
            if bundle.counters.kernel_counters:
                self._add_json_to_tar(tar, "counters.json", {
                    "kernel_counters": bundle.counters.kernel_counters,
                    "aggregate_counters": bundle.counters.aggregate_counters,
                })
            
            # Metrics
            self._add_json_to_tar(tar, "metrics.json", {
                "latency_ms": bundle.metrics.latency_ms,
                "throughput": bundle.metrics.throughput,
                "kernel_metrics": bundle.metrics.kernel_metrics,
            })
            
            # IR data
            for name, content in bundle.ir_data.items():
                safe_name = name.replace("/", "_").replace("\\", "_")
                self._add_text_to_tar(tar, f"ir/{safe_name}", content)
            
            # Logs
            if bundle.logs:
                self._add_text_to_tar(tar, "logs.txt", "\n".join(bundle.logs))
            
            # Artifacts
            for name, data in bundle.artifacts.items():
                self._add_bytes_to_tar(tar, f"artifacts/{name}", data)
    
    def _add_json_to_tar(
        self,
        tar: tarfile.TarFile,
        name: str,
        data: Dict[str, Any],
    ) -> None:
        """Add JSON file to tarball."""
        content = json.dumps(data, indent=2).encode("utf-8")
        self._add_bytes_to_tar(tar, name, content)
    
    def _add_text_to_tar(
        self,
        tar: tarfile.TarFile,
        name: str,
        text: str,
    ) -> None:
        """Add text file to tarball."""
        self._add_bytes_to_tar(tar, name, text.encode("utf-8"))
    
    def _add_bytes_to_tar(
        self,
        tar: tarfile.TarFile,
        name: str,
        data: bytes,
    ) -> None:
        """Add bytes to tarball."""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = time.time()
        tar.addfile(info, io.BytesIO(data))
    
    def _export_bundle(self, bundle: ForensicBundle, output_path: str) -> None:
        """Export as AACO bundle format."""
        # Custom binary format with header
        data = self._bundle_to_full_dict(bundle)
        json_bytes = json.dumps(data).encode("utf-8")
        compressed = gzip.compress(json_bytes)
        
        with open(output_path, "wb") as f:
            # Magic header
            f.write(b"AACO")
            # Version (2 bytes)
            f.write(b"\x01\x00")
            # Data length (4 bytes)
            f.write(len(compressed).to_bytes(4, "little"))
            # Compressed data
            f.write(compressed)
    
    def _export_html(self, bundle: ForensicBundle, output_path: str) -> None:
        """Export as self-contained HTML report."""
        data = self._bundle_to_full_dict(bundle)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AACO Forensic Bundle: {bundle.metadata.name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #e4002b; }}
        h2 {{ color: #333; border-bottom: 2px solid #e4002b; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; }}
        .metric-value {{ font-size: 24px; color: #e4002b; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #e4002b; color: white; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; overflow-x: auto; }}
        .tag {{ background: #e4002b; color: white; padding: 2px 8px; border-radius: 3px; margin: 2px; }}
    </style>
</head>
<body>
    <h1>AACO Forensic Bundle</h1>
    
    <div class="section">
        <h2>Metadata</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Bundle ID</td><td>{bundle.metadata.bundle_id}</td></tr>
            <tr><td>Name</td><td>{bundle.metadata.name}</td></tr>
            <tr><td>Workload ID</td><td>{bundle.metadata.workload_id}</td></tr>
            <tr><td>Created</td><td>{time.ctime(bundle.metadata.created_at)}</td></tr>
            <tr><td>Duration</td><td>{bundle.metadata.capture_duration_s:.2f}s</td></tr>
        </table>
        <p>Tags: {''.join(f'<span class="tag">{t}</span>' for t in bundle.metadata.tags)}</p>
    </div>
    
    <div class="section">
        <h2>Environment</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Hostname</td><td>{bundle.environment.hostname}</td></tr>
            <tr><td>OS</td><td>{bundle.environment.os_name} {bundle.environment.os_version}</td></tr>
            <tr><td>GPUs</td><td>{', '.join(bundle.environment.gpu_models) or 'N/A'}</td></tr>
            <tr><td>ROCm Version</td><td>{bundle.environment.rocm_version or 'N/A'}</td></tr>
            <tr><td>PyTorch Version</td><td>{bundle.environment.pytorch_version or 'N/A'}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Metrics Summary</h2>
        <div class="metric">
            <div class="metric-value">{bundle.metrics.latency_ms:.2f} ms</div>
            <div class="metric-label">Latency</div>
        </div>
        <div class="metric">
            <div class="metric-value">{bundle.metrics.throughput:.2f}</div>
            <div class="metric-label">Throughput</div>
        </div>
        <div class="metric">
            <div class="metric-value">{bundle.metrics.memory_peak_mb:.1f} MB</div>
            <div class="metric-label">Peak Memory</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Traces</h2>
        <p>Total traces: {len(bundle.traces)}</p>
        {''.join(f'<p>Trace {t.trace_id}: {t.event_count} events ({t.trace_type})</p>' for t in bundle.traces)}
    </div>
    
    <div class="section">
        <h2>Logs</h2>
        <pre>{chr(10).join(bundle.logs[:100])}</pre>
    </div>
    
    <script>
        // Embedded data for interactive use
        const bundleData = {json.dumps(data)};
    </script>
</body>
</html>"""
        
        with open(output_path, "w") as f:
            f.write(html)
    
    @staticmethod
    def load_json(path: str) -> ForensicBundle:
        """Load bundle from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        return ForensicBundle.from_dict(data)
    
    @staticmethod
    def load_json_gz(path: str) -> ForensicBundle:
        """Load bundle from gzipped JSON."""
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        
        return ForensicBundle.from_dict(data)
    
    @staticmethod
    def load_bundle(path: str) -> ForensicBundle:
        """Load bundle from AACO format."""
        with open(path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != b"AACO":
                raise ValueError("Invalid AACO bundle file")
            
            # Skip version
            f.read(2)
            
            # Read length
            length = int.from_bytes(f.read(4), "little")
            
            # Read and decompress
            compressed = f.read(length)
            json_bytes = gzip.decompress(compressed)
            data = json.loads(json_bytes)
        
        return ForensicBundle.from_dict(data)
