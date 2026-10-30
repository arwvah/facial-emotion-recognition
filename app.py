"""Compatibility launcher. Implementation: expression_lab.app."""

import importlib
import sys

implementation = importlib.import_module("expression_lab.app")
if __name__ == "__main__" and hasattr(implementation, "main"):
    raise SystemExit(implementation.main())
sys.modules[__name__] = implementation
