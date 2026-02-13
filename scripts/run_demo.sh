#!/bin/bash
# AACO Demo Script
# Demonstrates full AACO workflow

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     AMD AI Compute Observatory - Demo Script             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if model is provided
MODEL_PATH="${1:-models/resnet50.onnx}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Usage: ./scripts/run_demo.sh <model_path>"
    exit 1
fi

# Create output directory
OUTPUT_DIR="./demo_sessions"
mkdir -p "$OUTPUT_DIR"

echo "═══ Step 1: System Information ═══"
aaco info
echo ""

echo "═══ Step 2: Lock GPU Clocks (requires sudo) ═══"
if command -v rocm-smi &> /dev/null; then
    echo "Setting GPU to performance mode..."
    rocm-smi --setperflevel high || echo "Could not set performance level (may need sudo)"
fi
echo ""

echo "═══ Step 3: Run Baseline Benchmark ═══"
aaco run "$MODEL_PATH" \
    --backend migraphx \
    --warmup 10 \
    --iterations 100 \
    --tag "baseline" \
    --output "$OUTPUT_DIR" \
    --telemetry

BASELINE_SESSION=$(ls -td "$OUTPUT_DIR"/*/ | head -1)
echo "Baseline session: $BASELINE_SESSION"
echo ""

echo "═══ Step 4: Generate Report ═══"
aaco report "$BASELINE_SESSION" --format terminal
echo ""

echo "═══ Step 5: Generate HTML Report ═══"
HTML_REPORT="${BASELINE_SESSION}/report.html"
aaco report "$BASELINE_SESSION" --format html --output "$HTML_REPORT"
echo "HTML report saved to: $HTML_REPORT"
echo ""

echo "═══ Step 6: Run Comparison Benchmark ═══"
aaco run "$MODEL_PATH" \
    --backend migraphx \
    --warmup 10 \
    --iterations 100 \
    --tag "comparison" \
    --output "$OUTPUT_DIR" \
    --telemetry

CURRENT_SESSION=$(ls -td "$OUTPUT_DIR"/*/ | head -1)
echo "Comparison session: $CURRENT_SESSION"
echo ""

echo "═══ Step 7: Regression Analysis ═══"
aaco diff "$BASELINE_SESSION" "$CURRENT_SESSION"
echo ""

echo "═══ Step 8: List Sessions ═══"
aaco ls --output "$OUTPUT_DIR"
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Demo Complete!                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Sessions saved in: $OUTPUT_DIR"
echo "HTML Report: $HTML_REPORT"
echo ""
echo "Next steps:"
echo "  - View HTML report in browser"
echo "  - Run with --profile for kernel analysis"
echo "  - Compare different backends with aaco diff"
