# Tracelake Module

Unified trace storage and cross-layer correlation.

## Storage Format

All trace data is stored in Apache Parquet format for efficient columnar access.

## Schema Overview

| Table | Contents |
|-------|----------|
| `graph_nodes.parquet` | ONNX graph node metadata |
| `graph_edges.parquet` | ONNX graph edge connectivity |
| `kernels.parquet` | HIP kernel execution records |
| `gpu_telemetry.parquet` | GPU time-series metrics |
| `system_telemetry.parquet` | System time-series metrics |
| `attribution.parquet` | Graph-to-kernel mapping |

## Correlation

Tracelake provides cross-layer correlation via timestamp alignment and operation
attribution, enabling queries that span from ONNX operations to kernel executions.

## Usage

```python
from aaco.tracelake import TraceLake

lake = TraceLake.load("sessions/latest")

# Query kernel data
kernels = lake.query("SELECT * FROM kernels WHERE duration_ns > 1000000")

# Get graph-to-kernel mapping
attribution = lake.get_attribution()
```
