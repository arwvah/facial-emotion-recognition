"""Compatibility launcher. Implementation: expression_lab.evaluation."""

import importlib
import sys

implementation = importlib.import_module("expression_lab.evaluation")
if __name__ == "__main__" and hasattr(implementation, "main"):
    raise SystemExit(implementation.main())
sys.modules[__name__] = implementation
