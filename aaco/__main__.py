"""
AACO CLI Entry Point

This module allows running AACO as:
    python -m aaco [command] [options]

Or as a standalone executable:
    aaco.exe [command] [options]
"""

from aaco.cli import cli

if __name__ == "__main__":
    cli()
