#!/usr/bin/env python3
"""
AACO - Profile Model Script
===========================
Example script demonstrating how to profile an ONNX model using AACO.

Usage:
    python profile_model.py --model path/to/model.onnx --output ./sessions

© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Profile an ONNX model using AMD AI Compute Observatory"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sessions",
        help="Output directory for profiling session"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of profiling iterations"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Import AACO
    try:
        from aaco.core import Observatory
    except ImportError:
        print("Error: AACO not installed. Run: pip install aaco")
        sys.exit(1)
    
    # Create observatory
    config = args.config if args.config else None
    obs = Observatory(config=config)
    
    print(f"Profiling model: {model_path}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")
    print("-" * 50)
    
    # Run profiling
    session = obs.profile(
        model=str(model_path),
        iterations=args.iterations,
        warmup=args.warmup,
        output=args.output
    )
    
    print(f"\nSession completed: {session.id}")
    print(f"Output saved to: {session.path}")
    
    # Basic analysis
    analysis = obs.analyze(session)
    print("\n" + "=" * 50)
    print("Quick Summary:")
    print("=" * 50)
    print(analysis.summary())


if __name__ == "__main__":
    main()
