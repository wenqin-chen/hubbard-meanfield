"""Stationarity check for restricted-MF winners.

A code review pointed out that the restricted-MF scan ranks
fixed-texture saddles by their one-shot Helmholtz free energy, but
does not verify that those textures are SCF stationary points. A
texture can have low restricted-MF F yet flow under unrestricted SCF
to a *different* family (FM, a different multi-Q state, or an
amplitude collapse), in which case the restricted result is a
variational probe but not a real saddle.

This module exposes :func:`stationarity_check` which, given a
restricted-MF candidate (texture, free energy, parameters), runs a
short unrestricted SCF from that texture as a warm-start seed and
reports the stationarity gap:

- ``F_scf - F_restricted``: how much the SCF lowers the energy.
- ``M_scf / M_restricted``: amplitude relaxation factor.
- ``texture_overlap``: per-site cosine similarity between the
  warm-start texture and the SCF-converged texture (mean of
  ``<S_seed[i], S_scf[i]> / (|S_seed[i]| |S_scf[i]|)`` over sites
  where both norms exceed a threshold).
- ``family_label_after_scf``: Berg-Lüscher / triad / rms diagnostics
  on the SCF-converged texture, classified into a coarse family.
- ``converged``, ``n_iter``, ``status``: SCF metadata.

A "well-anchored" restricted winner has small ``F_scf - F_restricted``
(<~ 1e-4 per site in practice), high ``texture_overlap``
(> ~0.8 in the SkX regime), and the same family label after SCF.
A texture that flows to FM or hops to a different family is a
variational probe only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from hubbard_meanfield import (
    MagneticSupercell,
    TriangularParams,
    berg_luescher_skyrmion_number,
    validate_texture,
)
from hubbard_unrestricted_meanfield import self_consistent_solve

from triad_detector import detect_q_triad


def _coarse_family_label(
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
    *,
    commensurate_p: int,
) -> str:
    """Return a one-word family label for a (N_sites, 3) texture.

    Heuristic: rms moment + (BL well-defined vs undefined / 0 vs
    nonzero integer) + triad detector. Crude but deterministic;
    useful for spotting flow between families.

    BL labelling convention (post-review):

    - ``BL.status == "ok"`` and ``BL.number`` rounds to a *nonzero*
      integer within tolerance: return ``BL_<n>[_triad]``.
    - ``BL.status == "ok"`` and ``BL.number`` rounds to 0: return
      ``BL_0[_triad]`` — the BL is *defined* and equals 0.
    - ``BL.status != "ok"`` (typically
      ``"undefined_zero_moment"``): return
      ``BL_undefined[_triad]`` — the texture has zero-moment sites,
      so the Berg-Lüscher solid-angle integral is not defined.
      Common for in-plane multi-Q textures with alternating-magnitude
      sites.
    """
    arr = np.asarray(texture, dtype=float)
    site_norms = np.linalg.norm(arr, axis=1)
    rms_M = float(np.sqrt(np.mean(site_norms ** 2)))
    if rms_M < 1.0e-3:
        return "paramagnet"
    site_std_over_mean = float(np.std(site_norms) / max(np.mean(site_norms), 1e-12))
    if site_std_over_mean < 0.01:
        # Uniform-magnitude texture (FM-like, single-Q helix, or
        # any rank-2 in-plane texture with uniform |S|).
        return "uniform_M"
    # Multi-magnitude: SkX-like or canted_bloch-like.
    bl = berg_luescher_skyrmion_number(arr, supercell)
    bl_status = bl.get("status") if isinstance(bl, dict) else None
    bl_num = bl.get("number") if isinstance(bl, dict) else None
    if commensurate_p > 1:
        try:
            triad = detect_q_triad(arr, supercell, p=int(commensurate_p))
            is_triad = bool(triad.get("is_triad_q", False))
        except Exception:
            is_triad = False
    else:
        is_triad = False
    triad_suffix = "_triad" if is_triad else ""
    # BL is defined and well-rounded?
    if bl_status == "ok" and bl_num is not None:
        if abs(round(bl_num) - bl_num) < 0.05:
            n = int(round(bl_num))
            if n == 0:
                return f"BL_0{triad_suffix}"
            return f"BL_{n}{triad_suffix}"
        # Defined but non-integer (e.g. partial solid angle) -- rare.
        return f"BL_noninteger{triad_suffix}"
    # BL is undefined (typically: zero-moment sites trip the
    # threshold). Honest report: BL is undefined, not "BL=0".
    return f"BL_undefined{triad_suffix}"


def _texture_overlap(
    seed: NDArray[np.float64],
    converged: NDArray[np.float64],
    *,
    norm_threshold: float = 1.0e-4,
) -> float:
    """Mean per-site cosine similarity between two textures.

    Sites where either |S_seed| or |S_conv| < ``norm_threshold`` are
    skipped (a near-zero spin vector has undefined direction). If no
    sites pass the threshold, returns ``nan``.
    """
    a = np.asarray(seed, dtype=float)
    b = np.asarray(converged, dtype=float)
    n_a = np.linalg.norm(a, axis=1)
    n_b = np.linalg.norm(b, axis=1)
    mask = (n_a > norm_threshold) & (n_b > norm_threshold)
    if not np.any(mask):
        return float("nan")
    cos = (
        np.sum(a[mask] * b[mask], axis=1)
        / (n_a[mask] * n_b[mask])
    )
    return float(np.mean(cos))


@dataclass(frozen=True)
class StationarityResult:
    """One stationarity check outcome."""
    warm_start_label: str
    warm_start_amplitude: float
    warm_start_M: float
    F_restricted: float
    F_scf: Optional[float]
    F_gap: Optional[float]  # F_scf - F_restricted
    M_scf: Optional[float]
    M_relaxation: Optional[float]  # M_scf / M_seed
    texture_overlap: Optional[float]  # in [-1, 1]
    family_seed: str
    family_after_scf: str
    family_changed: bool
    # Raw Berg-Lüscher dicts for the seed and the SCF-converged
    # texture. Surfaces ``status``, ``number``, ``min_norm``, etc.,
    # so downstream callers can distinguish BL=defined-and-zero from
    # BL=undefined-due-to-zero-moment-sites without re-running the
    # helper.
    bl_seed: dict
    bl_after_scf: Optional[dict]
    scf_converged: bool
    scf_n_iter: int
    scf_status: str
    wall_seconds: float
    extra: dict


def stationarity_check(
    *,
    seed_texture: NDArray[np.float64],
    seed_label: str,
    seed_amplitude: float,
    F_restricted: float,
    params: TriangularParams,
    supercell: MagneticSupercell,
    filling: float,
    I: float,
    h_z: float = 0.0,
    alpha: float = 0.0,
    kappa_nk: int = 24,
    twist_grid: int = 1,
    mixing: str = "auto",
    mixing_alpha: float = 0.5,
    max_iter: int = 80,
    tol: float = 1.0e-5,
    workers: int = 1,
    commensurate_p: int = 1,
    norm_threshold: float = 1.0e-4,
) -> StationarityResult:
    """Run one unrestricted SCF from ``seed_texture`` and report whether
    the restricted-MF candidate is a stationary point.

    ``mixing="auto"`` selects ``"linear"`` for FM-like seeds (label
    matches ``"uniform_fm"`` or family classifier returns ``"uniform_M"``
    with low rms anisotropy) and ``"pulay"`` otherwise — the
    mixing rule that prevents the Pulay-FM limit cycle.
    """
    t0 = time.time()
    seed = validate_texture(seed_texture, supercell)
    family_seed = _coarse_family_label(
        seed, supercell, commensurate_p=int(commensurate_p),
    )
    bl_seed_raw = berg_luescher_skyrmion_number(seed, supercell)
    if not isinstance(bl_seed_raw, dict):
        bl_seed_raw = {"status": "ok", "number": float(bl_seed_raw)}
    if str(mixing).lower() == "auto":
        # FM-like → linear; multi-Q / SkX-like → Pulay.
        if seed_label == "uniform_fm" or family_seed in ("uniform_M",):
            mixing_resolved = "linear"
        else:
            mixing_resolved = "pulay"
    else:
        mixing_resolved = str(mixing).lower()

    params_with_field = TriangularParams(
        t=float(params.t), t2=float(params.t2), t3=float(params.t3),
        beta=float(params.beta),
        alpha_rashba=float(alpha),
        h_z=float(h_z),
        # CRITICAL: forward params.easy_axis_A. An earlier
        # implementation forgot this; consequence was that
        # any caller passing params=TriangularParams(easy_axis_A=A)
        # silently ran the SCF at A=0. Test:
        # test_stationarity_check.py::EasyAxisForwardingTests.
        easy_axis_A=float(params.easy_axis_A),
    )
    try:
        scf = self_consistent_solve(
            seed, params_with_field, supercell,
            coupling_I=float(I),
            kappa_nk=int(kappa_nk),
            twist_grid=int(twist_grid),
            fixed_filling=float(filling),
            mixing=mixing_resolved,
            mixing_alpha=float(mixing_alpha),
            tol=float(tol),
            max_iter=int(max_iter),
            workers=int(workers),
        )
    except Exception as exc:
        return StationarityResult(
            warm_start_label=str(seed_label),
            warm_start_amplitude=float(seed_amplitude),
            warm_start_M=float(np.sqrt(np.mean(np.sum(seed * seed, axis=1)))),
            F_restricted=float(F_restricted),
            F_scf=None, F_gap=None,
            M_scf=None, M_relaxation=None,
            texture_overlap=None,
            family_seed=family_seed,
            family_after_scf=f"error:{type(exc).__name__}",
            family_changed=True,
            bl_seed=bl_seed_raw,
            bl_after_scf=None,
            scf_converged=False,
            scf_n_iter=0,
            scf_status=f"exception:{exc!s}",
            wall_seconds=float(time.time() - t0),
            extra={"mixing_resolved": mixing_resolved},
        )
    converged = scf["S_converged"]
    family_after = _coarse_family_label(
        converged, supercell, commensurate_p=int(commensurate_p),
    )
    bl_after_raw = berg_luescher_skyrmion_number(converged, supercell)
    if not isinstance(bl_after_raw, dict):
        bl_after_raw = {"status": "ok", "number": float(bl_after_raw)}
    M_seed = float(np.sqrt(np.mean(np.sum(seed * seed, axis=1))))
    M_scf = float(np.sqrt(np.mean(np.sum(converged * converged, axis=1))))
    overlap = _texture_overlap(seed, converged, norm_threshold=float(norm_threshold))
    return StationarityResult(
        warm_start_label=str(seed_label),
        warm_start_amplitude=float(seed_amplitude),
        warm_start_M=M_seed,
        F_restricted=float(F_restricted),
        F_scf=float(scf["energy"]),
        F_gap=float(scf["energy"] - F_restricted),
        M_scf=M_scf,
        M_relaxation=(M_scf / M_seed) if M_seed > 1e-12 else None,
        texture_overlap=overlap,
        family_seed=family_seed,
        family_after_scf=family_after,
        family_changed=(family_seed != family_after),
        bl_seed=bl_seed_raw,
        bl_after_scf=bl_after_raw,
        scf_converged=bool(scf["converged"]),
        scf_n_iter=int(scf["n_iter"]),
        scf_status=str(scf["status"]),
        wall_seconds=float(time.time() - t0),
        extra={
            "mixing_resolved": mixing_resolved,
            "tol": float(tol),
            "max_iter": int(max_iter),
            "kappa_nk": int(kappa_nk),
            "twist_grid": int(twist_grid),
            "I": float(I),
            "h_z": float(h_z),
            "alpha": float(alpha),
        },
    )


__all__ = [
    "StationarityResult",
    "stationarity_check",
]
