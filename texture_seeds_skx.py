"""Canted triple-Q skyrmion-crystal seed library.

Seed textures such as the ``triple_q_tetrahedral_orthogonal`` family have
*modulated* ``S^z(R)`` and no uniform out-of-plane background, so any
canted-SkX ansatz that forces a purely in-plane Fourier modulation cannot
promote them.

This module exposes a general canted triple-Q ansatz with arbitrary
complex 3-vector mode spinors plus an optional uniform ``S^z_0`` background:

    S(R) = S^z_0 zhat + sum_{nu=1..3} [ S_{Q_nu} e^{i Q_nu . R} + c.c. ]

and four preset variants (Bloch / Neel / tetrahedral / coplanar-120) that
build the most common ``mode_spinors`` for Phase 1 (Zeeman) and Phase 2
(Rashba) sweeps.

Conventions
-----------
- ``q_triad`` defaults to ``commensurate_gamma_m_q_fracs(supercell, p)``
  (fractional reciprocal-lattice coordinates: ``Q_1 = p b1 / L1``,
  ``Q_2 = -p b1 / L1 - p b2 / L2``, ``Q_3 = p b2 / L2``). The L=9
  reference cell uses ``p = 2``; other cells must pass ``p`` explicitly.
- For Bloch / Neel SkX the per-mode in-plane unit vectors are derived
  from each Q's Cartesian direction: ``ehat_par_nu = Qhat_nu`` (in-plane)
  and ``ehat_perp_nu = zhat x ehat_par_nu``.
- ``A`` is the *spinor* amplitude as it appears in each preset's formula
  (so ``A`` and the texture rms are not the same — see each docstring).
- ``rms_normalize=True`` renormalizes the output to the analytic natural
  rms ``sqrt(2 * sum_n |S_Qn|^2 + S_z0^2)``. This is a sanity-normalize
  no-op on the raw texture and gates against FFT-mode numerical drift
  in the canted case where Fourier and uniform parts mix at finite
  precision.
- At ``A = 0`` (zero Fourier modulation), every preset reduces to a
  uniform ``S^z_0 zhat`` texture, matching
  ``uniform_ferromagnetic_texture(amplitude=|S_z0|, axis=(0, 0, +-1))``
  exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hubbard_meanfield import (
    MagneticSupercell,
    commensurate_gamma_m_q_fracs,
    normalize_texture_rms,
    texture_from_mode_spinors,
    uniform_ferromagnetic_texture,
    validate_texture,
)
from hubbard_nesting import RECIPROCAL_B1, RECIPROCAL_B2


def _resolved_q_fracs(
    supercell: MagneticSupercell,
    p: int,
    q_triad: Optional[ArrayLike],
) -> NDArray[np.float64]:
    """Return the (3, 2) fractional-reciprocal-lattice Q triad."""
    if q_triad is None:
        return np.asarray(commensurate_gamma_m_q_fracs(supercell, int(p)), dtype=float)
    arr = np.asarray(q_triad, dtype=float)
    if arr.shape != (3, 2):
        raise ValueError(
            f"q_triad must have shape (3, 2); got {arr.shape}."
        )
    return arr


def _natural_rms(mode_spinors: NDArray[np.complex128], s_z0: float) -> float:
    """Return the analytic rms predicted by the canted triple-Q formula.

    For Q_n strictly nonzero (non-degenerate Fourier modes), the texture
    ``sum_n S_Qn e^{i Q_n.R} + c.c. + S_z0 zhat`` has variance
    ``<|S|^2>_R = 2 sum_n |S_Qn|^2 + S_z0^2`` (the cross terms average
    to zero whenever ``Q_n != 0``).
    """
    return float(
        np.sqrt(2.0 * np.sum(np.abs(mode_spinors) ** 2) + float(s_z0) ** 2)
    )


def triple_q_with_uniform_canting(
    supercell: MagneticSupercell,
    *,
    mode_spinors: ArrayLike,
    S_z0: float = 0.0,
    p: int = 2,
    q_triad: Optional[ArrayLike] = None,
    rms_normalize: bool = True,
) -> NDArray[np.float64]:
    """Return ``texture_from_mode_spinors(...) + S_z0 zhat`` on a magnetic cell.

    Parameters
    ----------
    supercell
        Target magnetic supercell.
    mode_spinors
        Complex array of shape ``(3, 3)``: one ``C^3`` spinor per mode.
        ``mode_spinors[n]`` is the complex amplitude ``S_{Q_n}``.
    S_z0
        Real uniform out-of-plane background. Added to every site as
        ``(0, 0, S_z0)``.
    p
        Commensurate index used when ``q_triad is None`` to default to
        ``commensurate_gamma_m_q_fracs(supercell, p)``. The L=9 reference
        cell uses ``p = 2``; other cells must pass ``p`` explicitly.
    q_triad
        Optional override: a ``(3, 2)`` array of fractional reciprocal-
        lattice coordinates. Use the same convention as
        ``texture_from_mode_spinors`` (no Cartesian conversion).
    rms_normalize
        When ``True`` (default), normalize the output to the analytic
        natural rms ``sqrt(2 * sum_n |S_Qn|^2 + S_z0^2)``. This is a
        sanity-rescale that pins the texture rms exactly at the formula
        value (otherwise tiny FFT-mode drift propagates).

    Returns
    -------
    texture
        Real array of shape ``(N_c, 3)``.
    """
    spinors = np.asarray(mode_spinors, dtype=complex)
    if spinors.shape != (3, 3):
        raise ValueError(
            f"mode_spinors must have shape (3, 3); got {spinors.shape}."
        )
    s_z0 = float(S_z0)
    q_fracs = _resolved_q_fracs(supercell, int(p), q_triad)

    fourier_part = texture_from_mode_spinors(
        supercell, q_fracs, spinors, amplitude=None
    )
    field = validate_texture(fourier_part, supercell).copy()
    if s_z0 != 0.0:
        field[:, 2] += s_z0

    if rms_normalize:
        target = _natural_rms(spinors, s_z0)
        if target > 0.0:
            field = normalize_texture_rms(field, supercell, amplitude=target)
        # At target == 0 the field is identically zero; nothing to renormalize.
    return field


def _qhat_xy(supercell: MagneticSupercell, q_fracs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``(3, 2)`` Cartesian unit vectors along each Q_n (xy-plane)."""
    q_cart = (
        q_fracs[:, 0:1] * RECIPROCAL_B1[None, :]
        + q_fracs[:, 1:2] * RECIPROCAL_B2[None, :]
    )
    norms = np.linalg.norm(q_cart, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(
            "every Q in q_triad must be nonzero (a zero Q breaks the +-Q "
            "mode-pair Fourier convention and the in-plane direction of "
            "the Bloch / Neel SkX is undefined)."
        )
    return q_cart / norms


def canted_bloch_skx(
    supercell: MagneticSupercell,
    *,
    A: float,
    S_z0: float,
    p: int = 2,
    phases: Sequence[float] = (0.0, 0.0, 0.0),
    q_triad: Optional[ArrayLike] = None,
    rms_normalize: bool = True,
) -> NDArray[np.float64]:
    """Bloch-chirality canted SkX (Wang 2020 / DM-Bloch ansatz).

    ``S_{Q_nu} = (A/2) e^{i phi_nu} (ehat_par_nu + i ehat_perp_nu)`` with
    ``ehat_par_nu = Qhat_nu`` (in-plane Cartesian unit vector along
    ``Q_nu``) and ``ehat_perp_nu = zhat x ehat_par_nu``. Net out-of-plane
    contribution is ``S_z0 zhat`` (Fourier mode spinors have zero
    z-component). The per-mode magnitude is ``|S_{Q_nu}| = A / sqrt(2)``
    so the natural rms is ``sqrt(3 * A^2 + S_z0^2)``.
    """
    phases_arr = np.asarray(phases, dtype=float)
    if phases_arr.shape != (3,):
        raise ValueError(f"phases must have shape (3,); got {phases_arr.shape}.")
    q_fracs = _resolved_q_fracs(supercell, int(p), q_triad)
    qhat = _qhat_xy(supercell, q_fracs)
    mode_spinors = np.zeros((3, 3), dtype=complex)
    a_half = 0.5 * float(A)
    for n in range(3):
        e_par = np.array([qhat[n, 0], qhat[n, 1], 0.0], dtype=complex)
        e_perp = np.array([-qhat[n, 1], qhat[n, 0], 0.0], dtype=complex)
        mode_spinors[n] = (
            a_half * np.exp(1.0j * phases_arr[n]) * (e_par + 1.0j * e_perp)
        )
    return triple_q_with_uniform_canting(
        supercell,
        mode_spinors=mode_spinors,
        S_z0=float(S_z0),
        p=int(p),
        q_triad=q_fracs,
        rms_normalize=bool(rms_normalize),
    )


def canted_neel_skx(
    supercell: MagneticSupercell,
    *,
    A: float,
    S_z0: float,
    p: int = 2,
    phases: Sequence[float] = (0.0, 0.0, 0.0),
    q_triad: Optional[ArrayLike] = None,
    rms_normalize: bool = True,
) -> NDArray[np.float64]:
    """Neel-chirality canted SkX (alternative DM ansatz).

    ``S_{Q_nu} = (A/2) e^{i phi_nu} (ehat_perp_nu + i ehat_par_nu)`` -- the
    in-plane and rotated-perpendicular components are swapped relative to
    Bloch. Net out-of-plane: ``S_z0 zhat``. Per-mode magnitude
    ``|S_{Q_nu}| = A / sqrt(2)`` and natural rms ``sqrt(3 * A^2 + S_z0^2)``,
    same as ``canted_bloch_skx``; the two are related by a 90-degree
    in-plane rotation of every spin and differ only in chirality sign
    (Phase 2's seed family covers both signs explicitly).
    """
    phases_arr = np.asarray(phases, dtype=float)
    if phases_arr.shape != (3,):
        raise ValueError(f"phases must have shape (3,); got {phases_arr.shape}.")
    q_fracs = _resolved_q_fracs(supercell, int(p), q_triad)
    qhat = _qhat_xy(supercell, q_fracs)
    mode_spinors = np.zeros((3, 3), dtype=complex)
    a_half = 0.5 * float(A)
    for n in range(3):
        e_par = np.array([qhat[n, 0], qhat[n, 1], 0.0], dtype=complex)
        e_perp = np.array([-qhat[n, 1], qhat[n, 0], 0.0], dtype=complex)
        mode_spinors[n] = (
            a_half * np.exp(1.0j * phases_arr[n]) * (e_perp + 1.0j * e_par)
        )
    return triple_q_with_uniform_canting(
        supercell,
        mode_spinors=mode_spinors,
        S_z0=float(S_z0),
        p=int(p),
        q_triad=q_fracs,
        rms_normalize=bool(rms_normalize),
    )


def canted_tetrahedral_skx(
    supercell: MagneticSupercell,
    *,
    A: float,
    S_z0: float,
    p: int = 2,
    q_triad: Optional[ArrayLike] = None,
    rms_normalize: bool = True,
) -> NDArray[np.float64]:
    """Tetrahedral canted SkX seed family.

    ``S_{Q_nu} = A ehat_nu`` with three mutually-orthogonal real axes
    ``(xhat, yhat, zhat)``. Per-mode magnitude ``|S_{Q_nu}| = A`` and
    natural rms ``sqrt(6 * A^2 + S_z0^2)``.

    At ``S_z0 = 0`` this reproduces the existing
    ``triple_q_tetrahedral_orthogonal`` seed exactly (including the
    spatially modulated ``S^z(R) = 2 A cos(Q_3 . R)`` — *not* a uniform
    canted background). The BL = 8 topology is a property of this seed
    family: any A > 0 with ``S_z0 = 0`` gives the same integer
    ``Q_{BL,total} = 8`` (BL is scale-invariant on the unit-vector field).
    """
    ex = np.array([1.0, 0.0, 0.0], dtype=complex)
    ey = np.array([0.0, 1.0, 0.0], dtype=complex)
    ez = np.array([0.0, 0.0, 1.0], dtype=complex)
    mode_spinors = float(A) * np.stack([ex, ey, ez])
    return triple_q_with_uniform_canting(
        supercell,
        mode_spinors=mode_spinors,
        S_z0=float(S_z0),
        p=int(p),
        q_triad=q_triad,
        rms_normalize=bool(rms_normalize),
    )


def canted_coplanar_120_skx(
    supercell: MagneticSupercell,
    *,
    A: float,
    S_z0: float,
    p: int = 2,
    q_triad: Optional[ArrayLike] = None,
    rms_normalize: bool = True,
) -> NDArray[np.float64]:
    """Coplanar-120 canted SkX with uniform out-of-plane background.

    Real in-plane mode spinors at 120 degrees:
        ``S_{Q_1} = A (1, 0, 0)``,
        ``S_{Q_2} = A (-1/2, sqrt(3)/2, 0)``,
        ``S_{Q_3} = A (-1/2, -sqrt(3)/2, 0)``.

    Plus uniform ``S_z0 zhat``. Per-mode magnitude ``|S_{Q_nu}| = A`` and
    natural rms ``sqrt(6 * A^2 + S_z0^2)``. At ``S_z0 = 0`` reproduces the
    candidate-library ``triple_q_coplanar_120`` seed (with ``A`` chosen
    so the rms matches the library's ``m_value`` -- i.e., ``A = m / sqrt(6)``).
    """
    sqrt3_2 = float(np.sqrt(3.0)) / 2.0
    axis_0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    axis_1 = np.array([-0.5, sqrt3_2, 0.0], dtype=complex)
    axis_2 = np.array([-0.5, -sqrt3_2, 0.0], dtype=complex)
    mode_spinors = float(A) * np.stack([axis_0, axis_1, axis_2])
    return triple_q_with_uniform_canting(
        supercell,
        mode_spinors=mode_spinors,
        S_z0=float(S_z0),
        p=int(p),
        q_triad=q_triad,
        rms_normalize=bool(rms_normalize),
    )


def dm_cycloid_helix(
    supercell: MagneticSupercell,
    *,
    q_vector: ArrayLike,
    A: float,
    rms_normalize: bool = True,
) -> NDArray[np.float64]:
    """Single-Q DM-driven chiral cycloid -- the SkX's natural competitor.

    Mode spinor ``S_{Q} = (A / sqrt(2)) (ehat_par + i zhat)`` with
    ``ehat_par`` the in-plane Cartesian unit vector along ``q_vector``.
    Real-space texture is

        S(R) = sqrt(2) * A * (ehat_par * cos(Q . R) - zhat * sin(Q . R)),

    a cycloid that rotates in the ``(ehat_par, zhat)`` plane as one moves
    along ``q_vector``. Every site has the same moment magnitude
    ``|S| = sqrt(2) A``; the natural rms is ``sqrt(2) A`` and is
    constant in space.

    Parameters
    ----------
    supercell
        Magnetic supercell.
    q_vector
        Two-component fractional reciprocal-lattice coordinates (same
        convention as ``commensurate_gamma_m_q_fracs``). Must be nonzero.
    A
        Helix amplitude. Per-site moment magnitude is ``sqrt(2) A``.
    rms_normalize
        When ``True`` (default), normalize to ``sqrt(2) A``.

    Notes
    -----
    This is the Bogdanov-Yablonsky cycloid that wins at intermediate
    ``|D|/J`` ratios in the standard DM phase diagram (no field, no
    anisotropy). It is the "null result" Phase 2 must distinguish from
    a SkX winner: a single-Q helix has ``chirality_summary.mean_signed
    = 0`` by symmetry (no net scalar-chirality density), whereas a
    triple-Q Bloch / Neel canted SkX has a nonzero chirality density
    with sign tied to the choice of in-plane vs rotated-in-plane basis.
    """
    q_frac = np.asarray(q_vector, dtype=float).reshape(-1)
    if q_frac.shape != (2,):
        raise ValueError(
            f"q_vector must have length 2 (fractional reciprocal-lattice "
            f"coordinates); got shape {q_frac.shape}."
        )
    q_cart = q_frac[0] * RECIPROCAL_B1 + q_frac[1] * RECIPROCAL_B2
    q_norm = float(np.linalg.norm(q_cart))
    if q_norm == 0.0:
        raise ValueError(
            "q_vector must be nonzero (a zero Q makes the helix direction "
            "undefined and breaks the +-Q Fourier convention)."
        )
    e_par = q_cart / q_norm
    a_root_half = float(A) / float(np.sqrt(2.0))
    spinor = a_root_half * np.array(
        [complex(e_par[0]), complex(e_par[1]), 1.0j], dtype=complex
    )
    fourier = texture_from_mode_spinors(
        supercell,
        q_frac.reshape(1, 2),
        spinor.reshape(1, 3),
        amplitude=None,
    )
    if rms_normalize:
        target = float(np.sqrt(2.0) * abs(float(A)))
        if target > 0.0:
            fourier = normalize_texture_rms(fourier, supercell, amplitude=target)
    return fourier


__all__ = [
    "canted_bloch_skx",
    "canted_coplanar_120_skx",
    "canted_neel_skx",
    "canted_tetrahedral_skx",
    "dm_cycloid_helix",
    "triple_q_with_uniform_canting",
]
