"""Pair-summed triple-Q detector and normalized skyrmion number.

Phase 1 / Phase 2 acceptance gates need to answer two questions on every
converged texture:

1. "Is this a triple-Q state at the expected commensurate ``p``-triad?"
   -- the FFT of the texture concentrates on the six ``+-Q_nu`` bins of
   ``commensurate_gamma_m_q_fracs(supercell, p)`` with comparable weights
   in each mode-pair.

2. "Is this texture topologically a skyrmion crystal?" -- the total
   Berg-Luescher charge ``Q_{BL,total}`` is a nonzero integer, and the
   per-elementary-cell ``n_sk = Q_{BL,total} / p^2`` is also a nonzero
   integer (the magnetic supercell contains ``p^2`` elementary skyrmion
   unit cells when the converged Q-triad is ``Q_nu = p * b_nu / L``).

This module exposes ``detect_q_triad`` (question 1) and
``normalized_skyrmion_number`` (question 2). It replaces a global
Gini-index test on ``extract_dominant_q``, which rejects legitimate
sharp triple-Q textures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hubbard_meanfield import MagneticSupercell, validate_texture


def _triad_bins(p: int, L1: int, L2: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return the three ``(+Q_nu_bin, -Q_nu_bin)`` integer-bin pairs.

    Mirrors the ``commensurate_gamma_m_q_fracs(supercell, p)`` convention:
    ``Q_1 = (p/L1, 0)``, ``Q_2 = (-p/L1, -p/L2)``, ``Q_3 = (0, p/L2)`` in
    fractional reciprocal-lattice coordinates; FFT bins are
    ``(k1 mod L1, k2 mod L2)``.
    """
    p_int = int(p)
    return [
        ((p_int % L1, 0), ((-p_int) % L1, 0)),
        (((-p_int) % L1, (-p_int) % L2), (p_int % L1, p_int % L2)),
        ((0, p_int % L2), (0, (-p_int) % L2)),
    ]


def detect_q_triad(
    s_field: ArrayLike,
    supercell: MagneticSupercell,
    *,
    p: int,
    triad_concentration_threshold: float = 0.7,
    triad_balance_threshold: float = 1.3,
) -> dict[str, Any]:
    """FFT the spin texture and report pair-summed triple-Q diagnostics.

    Computes the spatial FFT of ``s_field`` over the magnetic supercell
    and pair-sums the six ``+-Q_nu`` bins on the expected commensurate
    ``p``-triad into three real mode powers
    ``P_nu = |S(Q_nu)|^2 + |S(-Q_nu)|^2`` for ``nu = 1, 2, 3``.

    Parameters
    ----------
    s_field
        Real spin texture of shape ``(N_c, 3)`` or ``(L1, L2, 3)``.
    supercell
        Magnetic supercell defining ``L1, L2`` and the site-index
        ordering.
    p
        Commensurate index for the expected Q-triad (the L=9 reference
        cell uses ``p = 2``; other cells must pass ``p`` explicitly).
    triad_concentration_threshold
        Minimum ``concentration_ratio`` for ``is_triad_q = True``.
    triad_balance_threshold
        Maximum ``balance_ratio`` for ``is_triad_q = True``.

    Returns
    -------
    dict with keys:

    ``power_spectrum`` : ndarray, shape ``(L1, L2)``
        Total ``|S(q)|^2 = sum_alpha |S^alpha(q)|^2`` from the spatial
        FFT (no shift). Same orientation as
        ``extract_dominant_q.power_spectrum``.
    ``power_q0`` : float
        ``|S(q = 0)|^2``, the uniform-component power. Reported separately
        and *not* included in the triad denominator (Zeeman / SOC sweeps
        drive ``power_q0`` up via canting -- the triad gate is about the
        Fourier-mode structure of the modulation, not the background).
    ``power_finite_q`` : float
        Total power at ``q != 0``.
    ``power_triad`` : float
        Sum of the three pair-summed mode powers
        ``sum_nu (|S(Q_nu)|^2 + |S(-Q_nu)|^2)``.
    ``mode_powers`` : list of three floats
        Per-mode pair-summed powers ``[P_1, P_2, P_3]``.
    ``concentration_ratio`` : float
        ``power_triad / power_finite_q`` (set to 0 when the
        ``q != 0`` power is zero).
    ``balance_ratio`` : float
        ``max(P_nu) / min(P_nu)`` over the three modes (set to ``inf``
        when any ``P_nu == 0`` and the triad still has some power, or
        to 1 when all three ``P_nu`` are zero).
    ``is_triad_q`` : bool
        ``True`` iff
        ``concentration_ratio >= triad_concentration_threshold`` and
        ``balance_ratio <= triad_balance_threshold``.
    ``p`` : int
        Echo of the commensurate index used.
    ``triad_bins`` : list of 3 pairs of ``(k1, k2)`` tuples
        The six ``+-Q_nu`` integer FFT bins, ordered as
        ``[((+Q1_bin), (-Q1_bin)), ...]`` for traceability.
    """
    arr = validate_texture(s_field, supercell)
    L1, L2 = int(supercell.L1), int(supercell.L2)
    arr_2d = arr.reshape(L1, L2, 3)
    fft = np.fft.fftn(arr_2d, axes=(0, 1))
    power = np.sum(np.abs(fft) ** 2, axis=2).astype(float)

    power_q0 = float(power[0, 0])
    power_total = float(np.sum(power))
    power_finite_q = power_total - power_q0
    if power_finite_q < 0.0:
        # Numerical noise: clamp to zero.
        power_finite_q = 0.0

    bins = _triad_bins(int(p), L1, L2)
    mode_powers: list[float] = []
    for plus, minus in bins:
        p_plus = float(power[plus[0], plus[1]])
        p_minus = float(power[minus[0], minus[1]])
        mode_powers.append(p_plus + p_minus)
    power_triad = float(sum(mode_powers))

    if power_finite_q > 0.0:
        concentration_ratio = power_triad / power_finite_q
    else:
        concentration_ratio = 0.0

    min_p = min(mode_powers)
    max_p = max(mode_powers)
    if max_p == 0.0:
        balance_ratio = 1.0
    elif min_p == 0.0:
        balance_ratio = float("inf")
    else:
        balance_ratio = max_p / min_p

    is_triad_q = (
        concentration_ratio >= float(triad_concentration_threshold)
        and balance_ratio <= float(triad_balance_threshold)
    )

    return {
        "power_spectrum": power,
        "power_q0": power_q0,
        "power_finite_q": float(power_finite_q),
        "power_triad": float(power_triad),
        "mode_powers": [float(v) for v in mode_powers],
        "concentration_ratio": float(concentration_ratio),
        "balance_ratio": float(balance_ratio),
        "is_triad_q": bool(is_triad_q),
        "p": int(p),
        "triad_bins": [
            ((int(plus[0]), int(plus[1])), (int(minus[0]), int(minus[1])))
            for plus, minus in bins
        ],
        "triad_concentration_threshold": float(triad_concentration_threshold),
        "triad_balance_threshold": float(triad_balance_threshold),
    }


def normalized_skyrmion_number(
    berg_luescher_result: dict[str, Any],
    p: int,
    *,
    integer_tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """Report ``(q_bl_total, n_sk, is_skx)`` from a Berg-Luescher dict.

    Given the magnetic-cell-total Berg-Luescher charge and the
    commensurate index ``p`` (so the supercell contains ``p^2``
    elementary skyrmion unit cells), returns:

    - ``q_bl_total``: total ``berg_luescher_skyrmion_number.number``
      (or ``None`` when the BL diagnostic is undefined, e.g. on a
      zero-moment texture).
    - ``n_sk``: ``q_bl_total / p**2``, expected integer for a SkX (or
      ``None`` when ``q_bl_total`` is ``None``).
    - ``is_skx``: ``True`` iff ``q_bl_total`` is a nonzero integer
      (within ``integer_tolerance`` absolute) and ``n_sk`` is a nonzero
      integer (within ``integer_tolerance`` absolute).

    Parameters
    ----------
    berg_luescher_result
        Output of
        :func:`hubbard_meanfield.berg_luescher_skyrmion_number`.
    p
        Commensurate index of the converged Q-triad.
    integer_tolerance
        Absolute tolerance for the "is integer" check on both
        ``q_bl_total`` and ``n_sk``.
    """
    status = berg_luescher_result.get("status")
    if status != "ok":
        return {
            "q_bl_total": None,
            "n_sk": None,
            "is_skx": False,
            "status": str(status) if status is not None else "missing",
            "p": int(p),
            "integer_tolerance": float(integer_tolerance),
        }

    q_bl_total = float(berg_luescher_result["number"])
    p_int = int(p)
    if p_int <= 0:
        raise ValueError(f"p must be a positive integer; got {p_int}.")
    n_sk = q_bl_total / float(p_int**2)
    tol = float(integer_tolerance)
    bl_is_integer = abs(q_bl_total - round(q_bl_total)) <= tol
    n_sk_is_integer = abs(n_sk - round(n_sk)) <= tol
    is_skx = (
        bl_is_integer
        and n_sk_is_integer
        and abs(q_bl_total) > tol
        and abs(n_sk) > tol
    )
    return {
        "q_bl_total": float(q_bl_total),
        "n_sk": float(n_sk),
        "is_skx": bool(is_skx),
        "status": "ok",
        "p": p_int,
        "integer_tolerance": tol,
        "q_bl_is_integer": bool(bl_is_integer),
        "n_sk_is_integer": bool(n_sk_is_integer),
    }


__all__ = [
    "detect_q_triad",
    "normalized_skyrmion_number",
]
