"""Tests for the U=0 Rashba band-structure validator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent
for path in (_PROJECT_ROOT,):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from hubbard_meanfield import MagneticSupercell  # noqa: E402
from rashba_band_validator import (  # noqa: E402
    exact_lattice_rashba_bands,
    validate_rashba_bands_at_U0,
)


class ExactRashbaBandsTests(unittest.TestCase):
    def test_zero_alpha_collapses_to_paramagnetic_dispersion(self) -> None:
        # At alpha = 0 the spin-split vanishes; both bands collapse onto
        # the paramagnetic dispersion eps(k).
        for k_cart in [(0.1, 0.0), (0.0, 0.2), (0.07, -0.05), (1.0, 1.2)]:
            e_minus, e_plus = exact_lattice_rashba_bands(
                *k_cart, t=1.0, alpha=0.0,
            )
            self.assertAlmostEqual(e_minus, e_plus, delta=1.0e-14)

    def test_small_k_expansion_matches_3_alpha_k(self) -> None:
        # At small |k|, the split is 2 alpha sqrt(F_x^2 + F_y^2)
        # = 3 alpha |k|. Verify at k|/|b_1| = 0.001 (very small).
        from hubbard_nesting import RECIPROCAL_B1
        b1_norm = float(np.linalg.norm(RECIPROCAL_B1))
        alpha = 0.1
        k_mag = 0.001 * b1_norm  # |k| << 1
        for direction in [(1.0, 0.0), (0.0, 1.0), (0.7, 0.7)]:
            d = np.asarray(direction, dtype=float)
            d /= np.linalg.norm(d)
            kx, ky = float(k_mag * d[0]), float(k_mag * d[1])
            e_minus, e_plus = exact_lattice_rashba_bands(
                kx, ky, t=1.0, alpha=alpha,
            )
            measured_split = (e_plus - e_minus) / 2.0
            expected_split = 3.0 * alpha * k_mag
            self.assertAlmostEqual(
                measured_split, expected_split, delta=1.0e-6,
            )


class ValidatorAtAcceptanceGridTests(unittest.TestCase):
    """The acceptance grid: ``alpha in {0.05, 0.1, 0.2}`` and
    ``|k|/|b_1| in {0.01, 0.05, 0.1}``. The validator must reproduce
    the exact-lattice eigenvalues to <1e-12 absolute (the project's
    Hamiltonian-build path achieves machine precision -- far tighter
    than the <1e-8 spec)."""

    def test_validator_matches_exact_form_at_acceptance_grid(self) -> None:
        report = validate_rashba_bands_at_U0(
            alpha_grid=[0.05, 0.1, 0.2],
            k_over_b1_grid=[0.01, 0.05, 0.1],
        )
        # The "<1e-8 relative" gate.
        self.assertLess(report["max_rel_residual"], 1.0e-8)
        # The tighter internal gate: build_supercell_hamiltonian's
        # output must be Hermitian and agree with the exact lattice
        # form to within numerical roundoff.
        self.assertLess(report["max_abs_residual"], 1.0e-12)
        self.assertLess(report["hermiticity_max_residual"], 1.0e-12)
        # Cross-check: every per-point row has both expected eigenvalues
        # close to a measured eigenvalue.
        self.assertEqual(
            len(report["per_point"]),
            3 * 3 * 3,  # |alpha_grid| * |k_grid| * |directions|
        )

    def test_validator_on_L9_supercell_still_passes(self) -> None:
        # Larger supercells produce folded copies of the same dispersion
        # at any given k_cart; the L=1 expected eigenvalues must still
        # appear in the folded spectrum to numerical precision.
        report = validate_rashba_bands_at_U0(
            alpha_grid=[0.1],
            k_over_b1_grid=[0.05],
            supercell=MagneticSupercell(9, 9),
        )
        self.assertLess(report["max_abs_residual"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()
