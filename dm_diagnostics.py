"""Strong-coupling DM/J diagnostic.

For a triangular Hubbard model with nearest-neighbor hopping ``t`` and
Rashba spin-orbit coupling ``alpha``, the standard second-order
strong-coupling (large ``U``) expansion gives Heisenberg + DM:

.. math::

   J = \\frac{4 t^2}{U},
   \\qquad
   |D| = \\frac{4 t \\alpha}{U},
   \\qquad
   \\frac{|D|}{J} = \\frac{\\alpha}{t}.

The natural DM wavelength is

.. math::

   \\lambda_{DM} = 2\\pi a \\frac{J}{|D|} = \\frac{2\\pi t a}{\\alpha},

where ``a`` is the lattice constant.

Regime classifier:

- ``lambda_DM / a > 10``: weak-DM regime, SkX unlikely (the DM
  wavelength does not fit any cell we can afford).
- ``1 < lambda_DM / a < 10``: SkX regime (Wang-2020 compatible).
- ``lambda_DM / a < 1``: strong-DM regime; atomic-scale chirality,
  frustrated, beyond a triple-Q SkX description.

The diagnostic is purely algebraic; it does not run SCF and does not
depend on any cluster geometry beyond the lattice constant choice
(``a = 1`` throughout this project).
"""

from __future__ import annotations

import math


_LATTICE_CONSTANT_A = 1.0


def strong_coupling_DM_ratio(
    t: float,
    U: float,
    alpha: float,
) -> dict[str, float | str]:
    """Return ``{D_over_J, lambda_DM_over_a, regime, J, D_abs}``.

    Parameters
    ----------
    t : float
        Nearest-neighbor hopping (positive).
    U : float
        Hubbard on-site interaction (positive; we are in the
        strong-coupling regime).
    alpha : float
        Rashba spin-orbit coupling (in units of ``t``). ``alpha = 0``
        gives ``D = 0`` and an undefined wavelength (returned as
        ``float("inf")``).

    Returns
    -------
    dict with keys:

    ``D_over_J`` : float
        Dimensionless ratio ``|D|/J = alpha/t`` (independent of ``U``
        at this order).
    ``lambda_DM_over_a`` : float
        ``2 * pi * t / alpha`` (with ``a = 1``). ``float("inf")`` at
        ``alpha = 0``.
    ``regime`` : str
        ``"weak_DM"`` / ``"skx_regime"`` / ``"strong_DM"`` / ``"no_DM"``
        per the classifier above. ``"no_DM"`` is returned when
        ``alpha = 0``.
    ``J`` : float
        ``4 t^2 / U``.
    ``D_abs`` : float
        ``4 t |alpha| / U``.
    """
    t_v = float(t)
    U_v = float(U)
    alpha_v = float(alpha)
    if t_v <= 0.0:
        raise ValueError(f"t must be positive; got {t_v}.")
    if U_v <= 0.0:
        raise ValueError(f"U must be positive (strong-coupling); got {U_v}.")
    J = 4.0 * t_v * t_v / U_v
    D_abs = 4.0 * t_v * abs(alpha_v) / U_v
    if alpha_v == 0.0:
        return {
            "D_over_J": 0.0,
            "lambda_DM_over_a": float("inf"),
            "regime": "no_DM",
            "J": J,
            "D_abs": 0.0,
        }
    D_over_J = abs(alpha_v) / t_v
    lambda_over_a = (2.0 * math.pi * t_v) / abs(alpha_v) / _LATTICE_CONSTANT_A
    if lambda_over_a > 10.0:
        regime = "weak_DM"
    elif lambda_over_a >= 1.0:
        regime = "skx_regime"
    else:
        regime = "strong_DM"
    return {
        "D_over_J": float(D_over_J),
        "lambda_DM_over_a": float(lambda_over_a),
        "regime": str(regime),
        "J": float(J),
        "D_abs": float(D_abs),
    }


__all__ = ["strong_coupling_DM_ratio"]
