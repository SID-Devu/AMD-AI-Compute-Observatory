# API Reference: Core Module

::: aaco.core
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Observatory

The main entry point for AACO functionality.

```python
from aaco import Observatory

obs = Observatory(config=None)
```

### Methods

#### profile()

Run a profiling session.

```python
def profile(
    self,
    model: str | Path,
    iterations: int = 100,
    warmup: int = 10,
    lab_mode: bool = False,
    config: dict = None,
    **kwargs
) -> Session:
    """
    Profile a model and return session data.
    
    Args:
        model: Path to ONNX model or model identifier
        iterations: Number of measured iterations
        warmup: Number of warmup iterations
        lab_mode: Enable laboratory mode
        config: Additional configuration
        
    Returns:
        Session object containing profiling data
        
    Example:
        >>> obs = Observatory()
        >>> session = obs.profile("model.onnx", iterations=100)
    """
```

#### analyze()

Analyze a profiling session.

```python
def analyze(
    self,
    session: Session | str | Path,
    config: dict = None
) -> Analysis:
    """
    Analyze profiling session data.
    
    Args:
        session: Session object or path to session
        config: Analysis configuration
        
    Returns:
        Analysis object with results
    """
```

#### report()

Generate a report from session data.

```python
def report(
    self,
    session: Session | str | Path,
    format: str = "html",
    output: str | Path = None,
    **kwargs
) -> Path:
    """
    Generate a report from session data.
    
    Args:
        session: Session to report on
        format: Output format ('html', 'json', 'markdown', 'pdf')
        output: Output file path
        
    Returns:
        Path to generated report
    """
```

#### compare()

Compare two sessions.

```python
def compare(
    self,
    baseline: Session | str | Path,
    current: Session | str | Path,
    config: dict = None
) -> Comparison:
    """
    Compare two profiling sessions.
    
    Args:
        baseline: Baseline session
        current: Current session to compare
        config: Comparison configuration
        
    Returns:
        Comparison results
    """
```

## Session

Represents a complete profiling session.

```python
@dataclass
class Session:
    """
    A profiling session containing all collected data.
    
    Attributes:
        id: Unique session identifier
        model: Model information
        config: Session configuration
        metrics: Collected metrics
        traces: Execution traces
        metadata: Session metadata
    """
    
    id: str
    model: ModelInfo
    config: SessionConfig
    metrics: Metrics
    traces: TraceData
    metadata: dict
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `latency` | `Series` | Latency measurements |
| `throughput` | `float` | Throughput (inferences/sec) |
| `memory_usage` | `dict` | Memory statistics |
| `gpu_utilization` | `float` | GPU utilization % |

## Analysis

Analysis results from a session.

```python
@dataclass
class Analysis:
    """
    Analysis results containing insights and metrics.
    
    Attributes:
        session: Source session
        statistics: Statistical summary
        bottleneck: Bottleneck classification
        heu: Hardware envelope utilization
    """
```

### Methods

#### root_cause()

```python
def root_cause(self, min_confidence: float = 0.7) -> RootCauseResult:
    """
    Perform Bayesian root cause analysis.
    
    Returns:
        RootCauseResult with ranked causes
    """
```

#### classify_bottleneck()

```python
def classify_bottleneck(self) -> BottleneckResult:
    """
    Classify the performance bottleneck.
    
    Returns:
        BottleneckResult with category and evidence
    """
```

## Config

Configuration management.

```python
from aaco import Config

# Load default config
config = Config.default()

# Load from file
config = Config.from_file("aaco.yaml")

# Load from dict
config = Config.from_dict({...})
```

## Exceptions

```python
from aaco.core.exceptions import (
    AACOError,           # Base exception
    ProfileError,        # Profiling errors
    AnalysisError,       # Analysis errors
    ConfigError,         # Configuration errors
    ModelError,          # Model loading errors
    HardwareError,       # Hardware access errors
)
```
