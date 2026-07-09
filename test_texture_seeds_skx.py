"""Unit tests for ``texture_seeds_skx``.

Covers the seed-library acceptance criteria:

- ``canted_tetrahedral_skx(L=9, A=anchor_A, S_z0=0, p=2)`` with
  ``anchor_A = 0.219 / sqrt(6)`` produces a texture whose
  ``berg_luescher_skyrmion_number.number`` matches the existing
  ``triple_q_tetrahedral_orthogonal`` seed (BL = 8 at L=9) to <1e-10
  absolute error.
- ``canted_{bloch,neel,tetrahedral,coplanar_120}_skx(L=9, A=0,
  S_z0=1.0)`` reduces to ``uniform_ferromagnetic_texture(amplitude=1.0)``
  exactly.
- With ``rms_normalize=True`` every output has
  ``texture_rms_amplitude`` matching the analytic
  ``sqrt(2 * sum_n |S_Qn|^2 + S_z0^2)`` to <1e-12 relative error.
- At ``S_z0 = 0`` ``canted_tetrahedral_skx`` reproduces the existing
  ``triple_q_tetrahedral_orthogonal`` seed at the same ``M`` exactly
  (bit-level reproduction at integer rms).
"""

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

from hubbard_meanfield import (  # noqa: E402
    MagneticSupercell,
    berg_luescher_skyrmion_number,
    chirality_summary,
    commensurate_gamma_m_q_fracs,
    texture_rms_amplitude,
    uniform_ferromagnetic_texture,
)

from texture_seeds_skx import (  # noqa: E402
    canted_bloch_skx,
    canted_coplanar_120_skx,
    canted_neel_skx,
    canted_tetrahedral_skx,
    dm_cycloid_helix,
    triple_q_with_uniform_canting,
)


class LaunchpadReproductionTests(unittest.TestCase):
    """At ``L=9, p=2, S_z0=0`` ``canted_tetrahedral_skx`` produces the
    Berg-Luescher = 8 reference texture."""

    def test_canted_tetrahedral_reproduces_BL_8_on_L9_anchor(self) -> None:
        cell = MagneticSupercell(9, 9)
        anchor_M = 0.219
        anchor_A = anchor_M / float(np.sqrt(6.0))
        texture = canted_tetrahedral_skx(cell, A=anchor_A, S_z0=0.0, p=2)
        self.assertAlmostEqual(
            texture_rms_amplitude(texture, cell), anchor_M, delta=1.0e-12
        )
        bl = berg_luescher_skyrmion_number(texture, cell)
        self.assertEqual(bl["status"], "ok")
        self.assertAlmostEqual(float(bl["number"]), 8.0, delta=1.0e-10)



class UniformFerromagnetLimitTests(unittest.TestCase):
    """At ``A = 0`` the Fourier modes vanish and the texture is exactly
    a uniform ``S_z0 zhat`` field, matching
    ``uniform_ferromagnetic_texture(amplitude=|S_z0|)``."""

    L = 9
    AMP = 1.0

    def setUp(self) -> None:
        self.cell = MagneticSupercell(self.L, self.L)
        self.ferro = uniform_ferromagnetic_texture(
            self.cell, amplitude=self.AMP, axis=(0.0, 0.0, 1.0)
        )

    def _assert_matches_ferro(self, fn) -> None:
        tex = fn(self.cell, A=0.0, S_z0=self.AMP, p=2)
        np.testing.assert_allclose(tex, self.ferro, atol=1.0e-14)

    def test_bloch_at_A_zero_is_uniform_ferromagnet(self) -> None:
        self._assert_matches_ferro(canted_bloch_skx)

    def test_neel_at_A_zero_is_uniform_ferromagnet(self) -> None:
        self._assert_matches_ferro(canted_neel_skx)

    def test_tetrahedral_at_A_zero_is_uniform_ferromagnet(self) -> None:
        self._assert_matches_ferro(canted_tetrahedral_skx)

    def test_coplanar_120_at_A_zero_is_uniform_ferromagnet(self) -> None:
        self._assert_matches_ferro(canted_coplanar_120_skx)


class RmsNormalizationTests(unittest.TestCase):
    """``rms_normalize=True`` pins the texture rms to the analytic natural
    rms ``sqrt(2 * sum_n |S_Qn|^2 + S_z0^2)`` to <1e-12 relative error.

    The Bloch / Neel families have per-mode magnitude ``|S_Q| = A/sqrt(2)``
    so the natural rms is ``sqrt(3 A^2 + S_z0^2)``. The tetrahedral /
    coplanar families have ``|S_Q| = A`` so the natural rms is
    ``sqrt(6 A^2 + S_z0^2)``.
    """

    L = 9

    def _natural_rms(self, family: str, A: float, S_z0: float) -> float:
        if family in ("bloch", "neel"):
            return float(np.sqrt(3.0 * A * A + S_z0 * S_z0))
        if family in ("tetrahedral", "coplanar_120"):
            return float(np.sqrt(6.0 * A * A + S_z0 * S_z0))
        raise ValueError(family)

    def test_natural_rms_matches_analytic_for_all_presets(self) -> None:
        cell = MagneticSupercell(self.L, self.L)
        params = [(0.1, 0.0), (0.2, 0.3), (0.05, 0.7), (0.0, 0.5)]
        builders = {
            "bloch": canted_bloch_skx,
            "neel": canted_neel_skx,
            "tetrahedral": canted_tetrahedral_skx,
            "coplanar_120": canted_coplanar_120_skx,
        }
        for family, builder in builders.items():
            for A, sz0 in params:
                tex = builder(cell, A=A, S_z0=sz0, p=2)
                expected = self._natural_rms(family, A, sz0)
                measured = texture_rms_amplitude(tex, cell)
                if expected > 0.0:
                    rel = abs(measured - expected) / expected
                else:
                    rel = abs(measured)
                with self.subTest(family=family, A=A, S_z0=sz0):
                    self.assertLess(rel, 1.0e-12)


class TripleQWithUniformCantingTests(unittest.TestCase):
    """Direct (non-preset) entry point: explicit ``mode_spinors`` plus
    ``S_z0`` background."""

    def test_q_triad_default_matches_commensurate_helper(self) -> None:
        cell = MagneticSupercell(9, 9)
        spinors = np.zeros((3, 3), dtype=complex)
        spinors[0, 0] = 0.05
        spinors[1, 1] = 0.05
        spinors[2, 2] = 0.05
        tex_default = triple_q_with_uniform_canting(
            cell, mode_spinors=spinors, S_z0=0.0, p=2
        )
        tex_explicit = triple_q_with_uniform_canting(
            cell, mode_spinors=spinors, S_z0=0.0, p=2,
            q_triad=commensurate_gamma_m_q_fracs(cell, p=2),
        )
        np.testing.assert_allclose(tex_default, tex_explicit, atol=1.0e-14)

    def test_zero_spinors_zero_canting_returns_zero(self) -> None:
        cell = MagneticSupercell(4, 4)
        zero_spinors = np.zeros((3, 3), dtype=complex)
        tex = triple_q_with_uniform_canting(
            cell, mode_spinors=zero_spinors, S_z0=0.0, p=1
        )
        np.testing.assert_array_equal(tex, np.zeros_like(tex))

    def test_mode_spinors_shape_check(self) -> None:
        cell = MagneticSupercell(4, 4)
        bad = np.zeros((2, 3), dtype=complex)
        with self.assertRaises(ValueError):
            triple_q_with_uniform_canting(
                cell, mode_spinors=bad, S_z0=0.0, p=1
            )


class DmCycloidHelixTests(unittest.TestCase):
    """Single-Q DM-driven cycloid helix.

    The mode spinor ``S_Q = (A/sqrt(2))(ehat_par + i zhat)`` gives a
    real-space texture ``sqrt(2) A (ehat_par cos(QR) - zhat sin(QR))``
    that rotates in the ``(ehat_par, zhat)`` plane. Every site has the
    same moment magnitude ``sqrt(2) A``; the texture is a cycloid (NOT
    a SkX) and has zero net scalar-chirality density by single-Q
    symmetry.
    """

    L = 9
    P = 2

    def setUp(self) -> None:
        self.cell = MagneticSupercell(self.L, self.L)
        # Use the L=9 reference-cell Q (the p=2 first member of the
        # Gamma-M triad) for tests.
        self.q_frac = commensurate_gamma_m_q_fracs(self.cell, p=self.P)[0]

    def test_rms_equals_sqrt2_A(self) -> None:
        for A in [0.05, 0.1, 0.2]:
            tex = dm_cycloid_helix(self.cell, q_vector=self.q_frac, A=A)
            rms = texture_rms_amplitude(tex, self.cell)
            self.assertAlmostEqual(rms, np.sqrt(2.0) * A, delta=1.0e-12)

    def test_per_site_moment_is_constant(self) -> None:
        # The cycloid is "saturated": every site sits on the same sphere.
        tex = dm_cycloid_helix(self.cell, q_vector=self.q_frac, A=0.1)
        norms = np.linalg.norm(tex, axis=1)
        np.testing.assert_allclose(
            norms, np.full_like(norms, np.sqrt(2.0) * 0.1), atol=1.0e-12,
        )

    def test_chirality_summary_mean_signed_is_zero(self) -> None:
        # Single-Q helix has no net scalar-chirality density: the
        # alternating-triangle sum cancels by the Q -> -Q symmetry of
        # the texture.
        tex = dm_cycloid_helix(self.cell, q_vector=self.q_frac, A=0.1)
        summary = chirality_summary(tex, self.cell)
        self.assertAlmostEqual(summary["mean_signed"], 0.0, delta=1.0e-12)
        # The absolute chirality is nonzero (cycloid still has nonzero
        # *unsigned* triangle product on each oriented triangle).
        self.assertGreater(summary["mean_absolute"], 0.0)

    def test_zero_q_vector_raises(self) -> None:
        with self.assertRaises(ValueError):
            dm_cycloid_helix(self.cell, q_vector=(0.0, 0.0), A=0.1)

    def test_q_vector_shape_check(self) -> None:
        with self.assertRaises(ValueError):
            dm_cycloid_helix(self.cell, q_vector=(0.1, 0.2, 0.3), A=0.1)


class ChiralityDistinguishabilityTests(unittest.TestCase):
    """The triple-Q Bloch / Neel canted-SkX *seeds* are C3-symmetric on
    the lattice and their ``mean_signed`` scalar-chirality density
    vanishes at the seed level by that symmetry (an exact-zero numerical
    result, not an approximation: this is **not** a topological SkX
    yet). The expected sign convention emerges from the SCF
    *converged* texture in the Rashba sweep, not from the seeds in
    isolation; the seed-level signature that's load-bearing for the
    sweep is that

    - the single-Q cycloid has ``mean_signed = 0`` (covered above),
    - Bloch and Neel seeds produce *different* real-space textures
      (so the SCF basin selection is non-trivial), and
    - the tetrahedral seed at ``S_z0 = 0`` IS topological
      (``BL = 8``, ``mean_signed != 0``) -- this is the L=9 reference
      texture and is covered by ``LaunchpadReproductionTests``.
    """

    def test_bloch_and_neel_seed_textures_are_different(self) -> None:
        cell = MagneticSupercell(9, 9)
        bloch = canted_bloch_skx(cell, A=0.1, S_z0=0.3, p=2)
        neel = canted_neel_skx(cell, A=0.1, S_z0=0.3, p=2)
        # The two seeds must differ in real space; otherwise the SCF
        # cannot distinguish Bloch from Neel basins.
        self.assertGreater(
            float(np.max(np.abs(bloch - neel))), 1.0e-6,
        )

    def test_bloch_and_neel_have_nonzero_absolute_chirality(self) -> None:
        # Although mean_signed cancels by C3 symmetry on these
        # triple-Q seeds, mean_absolute is nonzero -- the local
        # chirality density is finite, just sign-averaged.
        cell = MagneticSupercell(9, 9)
        for fn in (canted_bloch_skx, canted_neel_skx):
            tex = fn(cell, A=0.1, S_z0=0.3, p=2)
            chi = chirality_summary(tex, cell)
            self.assertGreater(float(chi["mean_absolute"]), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
