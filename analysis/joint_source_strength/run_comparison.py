"""Shared frozen IDW/Nelder--Mead primitives for the joint-strength experiment.

The original campaign carried a duplicate full data-size benchmark here. Only
these unchanged primitives are used by joint.py; the public entry point is
joint.py, not that unrelated historical benchmark.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.datasize_idw_comparison.run_comparison import (  # noqa: E402,F401
    IX, IY, digest, idw, nelder_mead,
)
