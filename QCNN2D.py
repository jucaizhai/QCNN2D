"""Import wrapper for QCNN2D.PY.

Python import machinery does not reliably pick up uppercase .PY files as source
modules, so this file loads the actual implementation from QCNN2D.PY.
"""

from __future__ import annotations

from pathlib import Path
import runpy

_module_path = Path(__file__).with_suffix(".PY")
_namespace = runpy.run_path(str(_module_path))
globals().update(_namespace)
