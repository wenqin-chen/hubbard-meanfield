"""U=0 SOC band-structure validator.

The acceptance gate is "eigenvalues at U = 0 match the Rashba
analytic form to <1e-8 relative error for ``alpha in {0.05, 0.1, 0.2}``
and ``|k| / |b_1| in {0.01, 0.05, 0.1}``". This module exposes
``validate_rashba_bands_at_U0`` which calls
``build_supercell_hamiltonian`` with ``texture = 0``, ``U = 0`` and
non-zero ``alpha_rashba``, then compares the spectrum to the exact
lattice Rashba dispersion

    E_pm(k) = eps(k) +- 2 alpha sqrt(F_x(k)^2 + F_y(k)^2),

where ``eps(k)`` is the triangular-lattice paramagnetic dispersion and
the auxiliary form factors are

    F_x(k) = -sum_{n=1..3} d_n^y * sin(k . d_n),
    F_y(k) =  sum_{n=1..3} d_n^x * sin(k . d_n),

with ``d_n`` the three primitive NN displacements
``(1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2)``.

The small-``|k|`` limit reduces to ``eps(k) +- 3 alpha |k|`` (the
Wang-2020 continuum form). The validator below compares the
build_supercell_hamiltonian spectrum to the **exact lattice form**, so
the residual is pure numerical (~1e-14) rather than truncation of the
continuum expansion at higher orders.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from hubbard_meanfield import (
    MagneticSupercell,
    build_supercell_hamiltonian,
    zero_texture,
)
from hubbard_nesting import (
    RECIPROCAL_B1,
    SQRT3,
    TriangularParams,
    triangular_dispersion,
)


# Three primitive NN displacements (Cartesian). The remaining three
# directions are their negations and contribute the Hermitian conjugate
# pair to the Rashba spinor block; see ``build_supercell_hamiltonian``.
_NN_PRIMITIVE_CART: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.5, 0.5 * SQRT3),
    (-0.5, 0.5 * SQRT3),
)


def exact_lattice_rashba_bands(
    kx: float,
    ky: float,
    *,
    t: float,
    alpha: float,
    t2: float = 0.0,
    t3: float = 0.0,
) -> tuple[float, float]:
    """Return ``(E_minus, E_plus) = eps(k) +- 2 alpha |F(k)|``.

    The split is purely from the NN Rashba spinor; ``t2`` / ``t3``
    contribute only to ``eps(k)`` (their bonds don't carry a Rashba
    coupling in this project's convention).
    """
    params = TriangularParams(
        t=float(t), t2=float(t2), t3=float(t3), beta=200.0,
    )
    eps = float(triangular_dispersion(float(kx), float(ky), params))
    f_x = 0.0
    f_y = 0.0
    for (dx, dy) in _NN_PRIMITIVE_CART:
        s = float(np.sin(kx * dx + ky * dy))
        f_x += -float(dy) * s
        f_y += float(dx) * s
    split = 2.0 * float(alpha) * float(np.sqrt(f_x * f_x + f_y * f_y))
    return eps - split, eps + split


def validate_rashba_bands_at_U0(
    alpha_grid: Sequence[float],
    k_over_b1_grid: Sequence[float],
    *,
    t: float = 1.0,
    supercell: MagneticSupercell | None = None,
    k_directions: Sequence[tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / SQRT3, 1.0),  # Gamma-K-ish direction
    ),
) -> dict[str, Any]:
    """Build the supercell Hamiltonian at ``U=0`` for each ``(alpha,
    |k|/|b_1|, direction)`` and compare the eigenvalues to the exact
    lattice Rashba bands above.

    Parameters
    ----------
    alpha_grid
        Iterable of Rashba couplings to test.
    k_over_b1_grid
        Iterable of ``|k|/|b_1|`` magnitudes (with ``|b_1| = 2 pi`` in
        Cartesian units of the triangular lattice).
    t
        NN hopping (default 1.0).
    supercell
        Magnetic supercell used to build the Hamiltonian. Default is
        ``MagneticSupercell(1, 1)`` (no zone folding -- the cleanest
        U=0 test case; the spectrum is exactly two spin-split bands per
        k-point). Larger supercells produce 2 N_c folded copies of the
        same dispersion and the validator still passes; this is useful
        as a regression cross-check.
    k_directions
        Iterable of 2D Cartesian unit-like vectors. Internally
        normalized before being scaled by ``|k|/|b_1| * |b_1|``.

    Returns
    -------
    dict with keys:

    - ``per_point``: list of dicts, one per ``(alpha, |k|/|b_1|, dir)``
      with ``alpha``, ``k_over_b1``, ``k_cart``, ``eigenvalues``
      (sorted ascending), ``expected``, ``abs_residual_max``,
      ``rel_residual_max``.
    - ``max_abs_residual``: scalar -- the worst over all points.
    - ``max_rel_residual``: scalar -- the worst over all points
      (denominator clamped at the larger of 1 and ``|eigenvalue|``
      to avoid blow-up near zero-crossings).
    - ``hermiticity_max_residual``: the worst ``||H - H^dag||`` over
      all sampled points.
    """
    cell = supercell if supercell is not None else MagneticSupercell(1, 1)
    bands_per_k = 2 * cell.num_sites
    b1_norm = float(np.linalg.norm(RECIPROCAL_B1))

    per_point: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    max_herm = 0.0

    for alpha in alpha_grid:
        params = TriangularParams(t=float(t), beta=200.0, alpha_rashba=float(alpha))
        zero_tex = zero_texture(cell)
        for k_frac in k_over_b1_grid:
            for direction in k_directions:
                d = np.asarray(direction, dtype=float)
                d_norm = float(np.linalg.norm(d))
                if d_norm == 0.0:
                    raise ValueError(
                        f"k_direction must be nonzero; got {direction!r}."
                    )
                k_mag = float(k_frac) * b1_norm
                k_cart = (k_mag / d_norm) * d
                h = build_supercell_hamiltonian(
                    k_cart, zero_tex, params, cell,
                )
                herm = float(np.max(np.abs(h - h.conj().T)))
                max_herm = max(max_herm, herm)
                eigs = np.sort(np.real(np.linalg.eigvalsh(h)))
                e_minus, e_plus = exact_lattice_rashba_bands(
                    float(k_cart[0]), float(k_cart[1]),
                    t=float(t), alpha=float(alpha),
                )
                # The L=1 cell has 2 bands; larger cells have 2*N_c
                # folded copies, which always include the L=1 doublet
                # at k_cart shifted by every supercell reciprocal-
                # lattice vector. The simplest gate that works for any
                # supercell choice is that {E_minus, E_plus} are among
                # the eigenvalues to <1e-10 absolute. Compute the
                # closest-match residuals on each.
                expected = np.array([e_minus, e_plus], dtype=float)
                # For each expected eigenvalue, find the nearest
                # numerical eigenvalue and compute the residual.
                abs_residuals: list[float] = []
                for e in expected:
                    nearest = float(np.min(np.abs(eigs - e)))
                    abs_residuals.append(nearest)
                abs_max = float(max(abs_residuals))
                rel_max = float(
                    max(
                        ar / max(1.0, abs(e))
                        for ar, e in zip(abs_residuals, expected)
                    )
                )
                max_abs = max(max_abs, abs_max)
                max_rel = max(max_rel, rel_max)
                per_point.append({
                    "alpha": float(alpha),
                    "k_over_b1": float(k_frac),
                    "k_cart": [float(k_cart[0]), float(k_cart[1])],
                    "direction_input": [float(direction[0]), float(direction[1])],
                    "expected_E_minus": float(e_minus),
                    "expected_E_plus": float(e_plus),
                    "abs_residual_E_minus": abs_residuals[0],
                    "abs_residual_E_plus": abs_residuals[1],
                    "rel_residual_max": rel_max,
                    "hermiticity_residual": herm,
                    "bands_per_k": int(bands_per_k),
                })

    return {
        "schema": "rashba_U0_validator_v1",
        "supercell": cell.to_dict(),
        "alpha_grid": [float(a) for a in alpha_grid],
        "k_over_b1_grid": [float(k) for k in k_over_b1_grid],
        "per_point": per_point,
        "max_abs_residual": float(max_abs),
        "max_rel_residual": float(max_rel),
        "hermiticity_max_residual": float(max_herm),
    }


__all__ = [
    "exact_lattice_rashba_bands",
    "validate_rashba_bands_at_U0",
]
