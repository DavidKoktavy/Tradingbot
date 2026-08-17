"""
Namespace package providing the `python -m trading_agent` entry point.

The implementation lives in the top-level packages (app/, broker/, risk/,
...) so that intra-project imports stay short and unambiguous. This
package exists purely to give the CLI the invocation the operator
documentation specifies.
"""

from app.cli import build_parser, main

__all__ = ["main", "build_parser"]
