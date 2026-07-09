"""Helper for building ``TriangularParams`` objects from (n, α, A, h_z)
cell parameters explicitly.

Motivation: directly calling ``TriangularParams(...)`` with a subset
of kwargs and defaulting the rest lets the dataclass defaults
*silently* hide an omitted parameter — e.g. a driver that forgets to
forward ``easy_axis_A`` silently runs SCF at A=0, and one that forgets
``alpha_rashba`` silently runs SCF at α=0 (where seeds collapse to the
paramagnet). Such silent-default bugs cost real compute time.

This helper makes the cell parameters explicit at the call site so
that future code reads ``params_for_cell(alpha=1.4, A=0.3)`` rather
than ``TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.3)`` —
omissions become visible immediately.

Drivers SHOULD use ``params_for_cell`` going forward; existing code
using ``TriangularParams`` directly continues to work bit-identically
(this helper is purely additive).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hubbard_nesting import TriangularParams  # noqa: E402


def params_for_cell(
    *,
    alpha: float,
    A: float = 0.0,
    h_z: float = 0.0,
    t: float = 1.0,
    t2: float = 0.0,
    t3: float = 0.0,
    beta: float = 200.0,
) -> TriangularParams:
    """Return a fully-populated ``TriangularParams`` for a (α, A, h_z) cell.

    All sweep-relevant fields are explicit kwargs (alpha_rashba via
    ``alpha``, easy_axis_A via ``A``, Zeeman ``h_z``). Defaults match
    the original ``TriangularParams`` defaults but are now visible at
    the call site.

    Parameters
    ----------
    alpha : Rashba SOC strength (forwarded to ``alpha_rashba``). REQUIRED
        (keyword-only) to force callers to think about it; a silently
        defaulted α=0 is the omission this helper is designed to prevent.
    A : easy-axis anisotropy strength (forwarded to ``easy_axis_A``).
        Default 0.0 (no anisotropy).
    h_z : uniform Zeeman field. Default 0.0.
    t, t2, t3 : hopping amplitudes (nearest, next-nearest, third-nearest).
        Defaults are 1.0, 0.0, 0.0.
    beta : inverse temperature (default 200.0).

    Examples
    --------
    >>> params = params_for_cell(alpha=1.4, A=0.3)  # example parameter point
    >>> params.alpha_rashba
    1.4
    >>> params.easy_axis_A
    0.3
    >>> params.h_z
    0.0
    """
    return TriangularParams(
        t=float(t),
        t2=float(t2),
        t3=float(t3),
        beta=float(beta),
        alpha_rashba=float(alpha),
        h_z=float(h_z),
        easy_axis_A=float(A),
    )


__all__ = ["params_for_cell"]
