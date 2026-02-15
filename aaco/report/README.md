# Report Module

Report generation in HTML and JSON formats.

## Output Formats

| Format | Use Case |
|--------|----------|
| HTML | Interactive visual reports for human review |
| JSON | Structured data for programmatic analysis |
| Markdown | Text-based summaries |

## Report Sections

| Section | Contents |
|---------|----------|
| Summary | Key metrics, verdict, recommendations |
| Metrics | Detailed performance metrics tables |
| Bottleneck | Classification and evidence signals |
| Timeline | Kernel execution timeline visualization |
| Telemetry | GPU and system telemetry charts |

## Usage

```python
from aaco.report import ReportGenerator

generator = ReportGenerator(session_path="sessions/latest")
generator.generate(format="html", output="report.html")
```

## CLI

```bash
aaco report --session sessions/latest --format html --output report.html
aaco report --session sessions/latest --format json --output metrics.json
```
