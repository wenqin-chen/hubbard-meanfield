"""Defensive ``sys.path`` fallback for environments without editable install.

Ensures the project root is on ``sys.path`` so the top-level modules
(``hubbard_meanfield``, ``hubbard_nesting``,
``hubbard_unrestricted_meanfield``) are importable when the package has
not been pip-installed (CI environments, fresh checkouts).

Pytest auto-discovers ``conftest.py`` at the project root. If tests later
move into a ``tests/`` subdirectory, update the path formula here to add
one extra ``.parent`` — the location and formula must move together.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

for path in (_PROJECT_ROOT,):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
