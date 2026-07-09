"""Restricted fixed-texture mean-field scan.

The cheapest cost-to-insight tool for the *finite-amplitude*
saddle-family ranking question. For each
``(I, h_z, alpha, texture_family, amplitude)`` point this helper

1. Builds the supercell-resolved spin texture from the seed library
   (or accepts a callable that returns one).
2. Diagonalizes the spinful Hamiltonian once on the supercell
   ``KappaGrid`` (eigenvalues only — no eigenvectors, no SCF
   iteration). Twist-averaging is supported by reusing the parent
   project's ``band_eigenvalues_twist_averaged``.
3. Evaluates the fixed-density Helmholtz free energy
   ``F(M; texture)`` via ``fixed_density_free_energy``.
4. Computes the topological / triad diagnostics on the texture
   (``berg_luscher_skyrmion_number``, ``detect_q_triad``,
   ``normalized_skyrmion_number``).

The scan ranks textures *directly* on F at each ``(I, field)``
parameter point. No iterative mixing means no convergence risk and
no random-seed long tail; the cost is bounded by the number of
diagonalizations.

Cost arithmetic: at L=9 ``kappa = 72`` ``twist = 1``,
one ``eigvalsh`` on the spinful supercell Hamiltonian is ~30 s with
``workers = 8``. For 7 texture families x 6 amplitudes x ~14
``(I, field)`` points = 588 diagonalizations x 30 s ~= 5 h, roughly
**5x cheaper** than the unrestricted-SCF arbiter at the same
parameter coverage (and without the random-seed tail / Pulay-FM
limit-cycle pathology).

Acceptance gates:

- *Reference regression.* At ``I/Ic_comm = 1.045, h_z = 0`` (the
  L=9 reference cell in commensurate units), a one-shot eigvalsh
  on the ``canted_tetrahedral_skx`` seed ansatz at ``A = 0.0894``
  (``M = sqrt(6) * A = 0.219``) reproduces the corresponding
  converged SCF energy *without any SCF iteration* — a
  regression gate on the fixed-texture free-energy evaluation.

- *Per-family F(h_z) curves.* At fixed ``I/Ic_comm = 1.045`` and
  the canonical ``h_z`` grid the per-family ``F(h_z)`` is
  monotone or shows a clean leader-flip; the leading family at
  each ``h_z`` is a Phase 2 narrow-SCF candidate (or a Phase 3
  candidate when scanning ``alpha`` instead of ``h_z``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from hubbard_meanfield import (
    MagneticSupercell,
    TriangularParams,
    band_eigenvalues_twist_averaged,
    berg_luescher_skyrmion_number,
    fixed_density_free_energy,
    validate_texture,
)

from triad_detector import detect_q_triad, normalized_skyrmion_number


# ---------------------------------------------------------------------------
# Texture spec (callable + label + amplitude grid).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextureSpec:
    """A named texture family + its amplitude grid.

    ``builder(supercell, amplitude) -> texture (N_sites, 3)`` constructs
    the fixed texture at the requested amplitude. ``rms_label`` is a
    short tag describing how ``amplitude`` maps to rms moment (e.g.,
    ``"M = sqrt(6) * A"`` for canted_tetrahedral, ``"M = A"`` for
    uniform_fm). It does not enter the computation; it is recorded in
    the per-row output for posterity.

    ``commensurate_p`` is the commensurate-Q-triad index that the
    builder uses (``p = 2`` for the L=9 reference cell, ``p = 1`` for
    a uniform / single-Q seed). It enters
    :func:`triad_detector.normalized_skyrmion_number` as the
    ``p`` argument; ``n_sk = q_bl_total / p**2``. Defaults to ``1``,
    which gives ``n_sk = q_bl_total`` (the BL number itself).

    ``extra_metadata`` is an arbitrary dict attached to every output row
    that this spec produces (e.g., ``{"S_z0": 0.0}`` for a tetrahedral
    seed at ``S_z0 = 0``).
    """

    label: str
    builder: Callable[[MagneticSupercell, float], NDArray[np.float64]]
    amplitudes: tuple[float, ...]
    rms_label: str = ""
    commensurate_p: int = 1
    extra_metadata: dict = None  # type: ignore[assignment]

    def metadata(self) -> dict:
        return dict(self.extra_metadata) if self.extra_metadata else {}


# ---------------------------------------------------------------------------
# Per-row diagnostics.
# ---------------------------------------------------------------------------


def _texture_rms(texture: NDArray[np.float64]) -> float:
    """Return the rms moment ``sqrt(<|S_i|^2>_i)`` of a (N, 3) texture."""
    arr = np.asarray(texture, dtype=float)
    norms_sq = np.sum(arr * arr, axis=-1)
    return float(np.sqrt(np.mean(norms_sq)))


def _safe_berg_luescher(
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
) -> dict:
    """Wrap ``berg_luescher_skyrmion_number`` so callers always get a
    dict-shaped result (not a raw float / None) regardless of which
    branch the underlying helper hits.
    """
    result = berg_luescher_skyrmion_number(texture, supercell)
    if isinstance(result, dict):
        return result
    # Older / scalar return — wrap.
    return {"status": "ok", "number": float(result)}


def _row_diagnostics(
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
    *,
    commensurate_p: int,
    detect_triad: bool,
) -> dict:
    """Return the topological / triad diagnostics for one texture.

    ``commensurate_p`` is the texture spec's commensurate-Q-triad index,
    forwarded both to :func:`triad_detector.detect_q_triad` (which
    builds the expected p-triad from the supercell) and to
    :func:`triad_detector.normalized_skyrmion_number` so
    ``n_sk = q_bl_total / p**2``. The L=9 reference cell uses ``p = 2``;
    a uniform-FM seed uses ``p = 1`` (no triad).

    ``detect_triad`` is a per-row gate: when ``False`` the triad
    detector is skipped (useful for uniform / single-Q textures whose
    p=1 case the triad detector would either trivially false-match
    or reject as non-triad).
    """
    diag: dict = {"M": _texture_rms(texture)}
    bl = _safe_berg_luescher(texture, supercell)
    diag["BL"] = bl
    diag["BL_number"] = (
        float(bl["number"]) if bl.get("number") is not None else None
    )
    diag["n_sk"] = normalized_skyrmion_number(bl, int(commensurate_p))
    if detect_triad:
        triad = detect_q_triad(texture, supercell, p=int(commensurate_p))
        diag["triad"] = triad
        diag["is_triad_q"] = bool(triad.get("is_triad_q", False))
    else:
        diag["triad"] = None
        diag["is_triad_q"] = False
    return diag


# ---------------------------------------------------------------------------
# Per-row free-energy evaluation (single (I, h_z, alpha, texture, M) point).
# ---------------------------------------------------------------------------


def _evaluate_row(
    *,
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
    params_with_field: TriangularParams,
    filling: float,
    I: float,
    kappa_nk: int,
    twist_grid: int,
    workers: int,
) -> dict:
    """Diagonalize the supercell Hamiltonian for one fixed texture and
    return the per-row free energy + chemical potential (as
    ``fixed_density_free_energy`` consumes them)."""
    eigenvalues = band_eigenvalues_twist_averaged(
        texture, params_with_field, supercell,
        kappa_nk=int(kappa_nk),
        twist_grid=int(twist_grid),
        workers=int(workers),
    )
    # Forward params.easy_axis_A so the anisotropy free-energy term
    # ΔF_anis = -A * <(S_HS^z)²>_i is added to F (the band Hamiltonian
    # already includes the texture-proportional Zeeman from anisotropy
    # via build_supercell_hamiltonian; this term completes the
    # bookkeeping).
    F = fixed_density_free_energy(
        float(filling), eigenvalues, texture,
        float(I), float(params_with_field.beta),
        easy_axis_A=float(params_with_field.easy_axis_A),
    )
    return {
        "F": float(F),
        "n_eigenvalues": int(eigenvalues.size),
    }


# ---------------------------------------------------------------------------
# Top-level scan entry point.
# ---------------------------------------------------------------------------


def restricted_mf_scan(
    *,
    params: TriangularParams,
    supercell: MagneticSupercell,
    filling: float,
    textures: Sequence[TextureSpec],
    interactions: Sequence[float],
    h_z_list: Sequence[float] = (0.0,),
    alpha_list: Sequence[float] = (0.0,),
    easy_axis_A_list: Sequence[float] = (0.0,),
    kappa_nk: int = 72,
    twist_grid: int = 1,
    workers: int = 1,
    progress: bool = False,
) -> dict:
    """Scan the restricted fixed-texture free energy across all
    ``(I, h_z, alpha, texture, amplitude)`` combinations and return
    per-row results + per-cell winners.

    Parameters
    ----------
    params
        ``TriangularParams`` carrying ``t``, ``t2``, ``t3``, ``beta``.
        The fields ``alpha_rashba`` and ``h_z`` on this object are
        IGNORED — the scan iterates over its own ``alpha_list`` and
        ``h_z_list`` and constructs a fresh ``TriangularParams`` per
        ``(alpha, h_z)`` cell.
    supercell
        ``MagneticSupercell`` whose ``L1, L2`` set the cell extent
        (e.g. ``MagneticSupercell(9, 9)`` for the L=9 reference cell).
    filling
        Spinful filling per microscopic site (``0.1134`` for the Wang
        reference configuration).
    textures
        Iterable of ``TextureSpec``. Each ``TextureSpec.builder`` is
        called once per amplitude per ``(alpha, h_z)`` cell.
    interactions
        Iterable of Hubbard ``I`` values (absolute, *not* ratios).
        Callers convert from ``I/Ic_comm`` to absolute ``I`` outside
        this function.
    h_z_list
        Iterable of Zeeman field values; default ``(0.0,)``.
    alpha_list
        Iterable of Rashba couplings; default ``(0.0,)``.
    kappa_nk, twist_grid, workers
        Forwarded to ``band_eigenvalues_twist_averaged``.
    progress
        If ``True``, print one line per row with timing.

    The triad detector runs automatically per row whenever the
    texture spec's ``commensurate_p > 1`` (a meaningful triad lives
    on the L1 / p, L2 / p sub-lattice). For ``commensurate_p == 1``
    (uniform / single-Q seeds) the triad row field is ``None`` and
    ``is_triad_q = False``.

    Returns
    -------
    dict with:

    - ``schema_version``, ``params``, ``supercell``, ``filling``,
      ``kappa_nk``, ``twist_grid``, ``workers``, ``interactions``,
      ``h_z_list``, ``alpha_list``: parameter echo.
    - ``rows``: list of per-(I, h_z, alpha, texture, amplitude) dicts
      with ``F``, ``M``, ``BL``, ``BL_number``, ``n_sk``, ``triad``,
      ``is_triad_q``, ``texture_label``, ``amplitude``,
      ``wall_seconds``, ``texture_metadata``, ``I``, ``h_z``,
      ``alpha``.
    - ``winners_per_cell``: list of per-(I, h_z, alpha) dicts giving
      the lowest-F row's ``texture_label``, ``amplitude``,
      ``F``, ``M``, ``BL_number``, ``is_triad_q``.
    - ``total_wall_seconds``.
    """
    rows: list[dict] = []
    interactions_t = tuple(float(x) for x in interactions)
    h_z_t = tuple(float(x) for x in h_z_list)
    alpha_t = tuple(float(x) for x in alpha_list)
    A_t = tuple(float(x) for x in easy_axis_A_list)

    t0 = time.time()
    for alpha in alpha_t:
        for h_z in h_z_t:
            for A_anis in A_t:
                params_with_field = TriangularParams(
                    t=float(params.t), t2=float(params.t2), t3=float(params.t3),
                    beta=float(params.beta),
                    alpha_rashba=float(alpha),
                    h_z=float(h_z),
                    easy_axis_A=float(A_anis),
                )
                for spec in textures:
                    for amplitude in spec.amplitudes:
                        texture = spec.builder(supercell, float(amplitude))
                        validate_texture(texture, supercell)  # belt-and-braces
                        diag = _row_diagnostics(
                            texture, supercell,
                            commensurate_p=int(spec.commensurate_p),
                            detect_triad=bool(int(spec.commensurate_p) > 1),
                        )
                        for I in interactions_t:
                            ta = time.time()
                            e_row = _evaluate_row(
                                texture=texture, supercell=supercell,
                                params_with_field=params_with_field,
                                filling=float(filling), I=float(I),
                                kappa_nk=int(kappa_nk),
                                twist_grid=int(twist_grid),
                                workers=int(workers),
                            )
                            wall = time.time() - ta
                            row = {
                                "alpha": float(alpha),
                                "h_z": float(h_z),
                                "easy_axis_A": float(A_anis),
                                "I": float(I),
                                "texture_label": str(spec.label),
                                "amplitude": float(amplitude),
                                "rms_label": str(spec.rms_label),
                                "texture_metadata": spec.metadata(),
                                "F": float(e_row["F"]),
                                "M": float(diag["M"]),
                                "BL": diag["BL"],
                                "BL_number": diag["BL_number"],
                                "n_sk": diag["n_sk"],
                                "triad": diag.get("triad"),
                                "is_triad_q": bool(diag["is_triad_q"]),
                                "wall_seconds": float(wall),
                            }
                            rows.append(row)
                            if progress:
                                print(
                                    f"  alpha={alpha:5.3f} h_z={h_z:5.3f} "
                                    f"A={A_anis:5.3f} I={I:8.4f}"
                                    f"  tex={spec.label:35s} A_amp={amplitude:6.4f}"
                                    f"  M={diag['M']:.4f}  F={e_row['F']:+.6e}"
                                    f"  BL={diag['BL_number']}  triad={diag['is_triad_q']}"
                                    f"  dt={wall:5.1f}s",
                                    flush=True,
                                )
    total_wall = time.time() - t0

    # Per-cell winners (lowest F at each (I, h_z, alpha, easy_axis_A))
    winners: list[dict] = []
    cell_keys = sorted({
        (r["alpha"], r["h_z"], r["easy_axis_A"], r["I"]) for r in rows
    })
    for alpha, h_z, A_anis, I in cell_keys:
        cell_rows = [
            r for r in rows
            if r["alpha"] == alpha and r["h_z"] == h_z
            and r["easy_axis_A"] == A_anis and r["I"] == I
        ]
        winner = min(cell_rows, key=lambda r: r["F"])
        winners.append({
            "alpha": float(alpha),
            "h_z": float(h_z),
            "easy_axis_A": float(A_anis),
            "I": float(I),
            "winner_label": str(winner["texture_label"]),
            "winner_amplitude": float(winner["amplitude"]),
            "winner_F": float(winner["F"]),
            "winner_M": float(winner["M"]),
            "winner_BL_number": winner["BL_number"],
            "winner_is_triad_q": bool(winner["is_triad_q"]),
        })

    return {
        "schema_version": 2,  # v2 adds easy_axis_A axis to rows + winners_per_cell
        "params": {
            "t": float(params.t),
            "t2": float(params.t2),
            "t3": float(params.t3),
            "beta": float(params.beta),
        },
        "supercell": {
            "L1": int(supercell.L1),
            "L2": int(supercell.L2),
            "num_sites": int(supercell.num_sites),
        },
        "filling": float(filling),
        "kappa_nk": int(kappa_nk),
        "twist_grid": int(twist_grid),
        "workers": int(workers),
        "interactions": list(interactions_t),
        "h_z_list": list(h_z_t),
        "alpha_list": list(alpha_t),
        "easy_axis_A_list": list(A_t),
        "n_textures": int(len(textures)),
        "n_rows": int(len(rows)),
        "rows": rows,
        "winners_per_cell": winners,
        "total_wall_seconds": float(total_wall),
    }


__all__ = [
    "TextureSpec",
    "restricted_mf_scan",
]
