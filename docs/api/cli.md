# API Reference: CLI

The AACO command-line interface.

## Installation

The CLI is installed automatically with the package:

```bash
pip install aaco
aaco --help
```

## Commands

### aaco profile

Run a profiling session.

```bash
aaco profile [OPTIONS] --model MODEL

Options:
  --model, -m PATH          Path to ONNX model (required)
  --output, -o PATH         Output directory [default: ./sessions]
  --iterations, -n INT      Number of iterations [default: 100]
  --warmup, -w INT          Warmup iterations [default: 10]
  --lab-mode, -l            Enable laboratory mode
  --config, -c PATH         Configuration file
  --device, -d INT          GPU device ID [default: 0]
  --input-shapes TEXT       Input shapes as JSON
  --collectors TEXT         Collectors to use (comma-separated)
  --verbose, -v             Verbose output
  --help                    Show help message

Examples:
  # Basic profile
  aaco profile -m model.onnx
  
  # Lab mode with custom iterations
  aaco profile -m model.onnx -l -n 500 -w 50
  
  # Specific collectors
  aaco profile -m model.onnx --collectors timing,counters,traces
```

### aaco analyze

Analyze a profiling session.

```bash
aaco analyze [OPTIONS] --session SESSION

Options:
  --session, -s PATH        Session directory (required)
  --output, -o PATH         Output file
  --format, -f TEXT         Output format [default: json]
  --config, -c PATH         Analysis configuration
  --verbose, -v             Verbose output
  --help                    Show help message

Examples:
  # Basic analysis
  aaco analyze -s sessions/20260214_120000
  
  # Output to file
  aaco analyze -s sessions/latest -o analysis.json
```

### aaco report

Generate reports from sessions.

```bash
aaco report [OPTIONS] --session SESSION

Options:
  --session, -s PATH        Session directory (required)
  --output, -o PATH         Output file
  --format, -f TEXT         Format: html, json, markdown, pdf [default: html]
  --template TEXT           Custom template
  --open                    Open report after generation
  --help                    Show help message

Examples:
  # HTML report
  aaco report -s sessions/latest -f html -o report.html
  
  # JSON report
  aaco report -s sessions/latest -f json
```

### aaco dashboard

Launch interactive dashboard.

```bash
aaco dashboard [OPTIONS]

Options:
  --session, -s PATH        Session to display
  --port, -p INT            Port number [default: 8501]
  --host TEXT               Host address [default: localhost]
  --no-browser              Don't open browser
  --help                    Show help message

Examples:
  # Launch dashboard
  aaco dashboard -s sessions/latest
  
  # Custom port
  aaco dashboard -s sessions/latest -p 8080
```

### aaco compare

Compare profiling sessions.

```bash
aaco compare [OPTIONS] --baseline BASELINE --current CURRENT

Options:
  --baseline, -b PATH       Baseline session (required)
  --current, -c PATH        Current session (required)
  --output, -o PATH         Output file
  --threshold FLOAT         Regression threshold [default: 0.05]
  --format, -f TEXT         Output format [default: table]
  --help                    Show help message

Examples:
  # Compare sessions
  aaco compare -b sessions/baseline -c sessions/current
  
  # With custom threshold
  aaco compare -b baseline -c current --threshold 0.03
```

### aaco doctor

System diagnostics.

```bash
aaco doctor [OPTIONS]

Options:
  --fix                     Attempt to fix issues
  --verbose, -v             Verbose output
  --help                    Show help message

Examples:
  aaco doctor
  aaco doctor --fix
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AACO_CONFIG` | Path to config file |
| `AACO_LOG_LEVEL` | Log level |
| `AACO_SESSION_DIR` | Default session directory |
| `AACO_OUTPUT_DIR` | Default output directory |

### Config File

```yaml
# ~/.config/aaco/config.yaml
profiling:
  default_iterations: 100
  default_warmup: 10
  
reporting:
  default_format: html
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Model error |
| 4 | Hardware error |
| 5 | Regression detected (compare) |
