"""
Thin wrapper that imports run_pipeline from sight.

Search order:
 1. Same directory as this file (plugin dir — normal deployment)
 2. ~/git/geo (development / host environment)
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEO_REPO = os.path.expanduser('~/git/geo')

for _path in [_HERE, _GEO_REPO]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from sight import run_pipeline  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Cannot import sight. "
        "Place sight.py alongside this file ({}/) "
        "or ensure ~/git/geo is on the Python path. "
        "Original error: {}".format(_HERE, e)
    )
