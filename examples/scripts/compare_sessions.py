#!/usr/bin/env python3
"""
AACO - Compare Sessions Script
==============================
Example script for comparing two profiling sessions.

Usage:
    python compare_sessions.py --baseline session1 --current session2

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Compare two AACO profiling sessions"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline session"
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current session"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for comparison report"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "html"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression threshold (default: 5%%)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    
    if not baseline_path.exists():
        print(f"Error: Baseline session not found: {baseline_path}")
        sys.exit(1)
    
    if not current_path.exists():
        print(f"Error: Current session not found: {current_path}")
        sys.exit(1)
    
    # Import AACO
    try:
        from aaco.core import Observatory
        from aaco.analytics import DriftDetector
    except ImportError:
        print("Error: AACO not installed. Run: pip install aaco")
        sys.exit(1)
    
    # Create observatory and compare
    obs = Observatory()
    
    print(f"Comparing sessions:")
    print(f"  Baseline: {baseline_path}")
    print(f"  Current:  {current_path}")
    print("-" * 50)
    
    comparison = obs.compare(
        baseline=str(baseline_path),
        current=str(current_path)
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("Comparison Results:")
    print("=" * 50)
    
    has_regression = False
    for metric, result in comparison.items():
        change = result.get("change_percent", 0)
        status = "✓" if abs(change) < args.threshold * 100 else "✗"
        if change > args.threshold * 100:
            has_regression = True
            status = "⚠ REGRESSION"
        
        print(f"{metric}: {result.get('baseline', 'N/A'):.2f} -> {result.get('current', 'N/A'):.2f} ({change:+.1f}%) {status}")
    
    # Exit code based on regression
    if has_regression:
        print(f"\n⚠ Performance regression detected (threshold: {args.threshold*100:.1f}%)")
        sys.exit(1)
    else:
        print(f"\n✓ No significant regression (threshold: {args.threshold*100:.1f}%)")
        sys.exit(0)


if __name__ == "__main__":
    main()
