"""Unit tests for ``triad_detector``.

Covers the detector acceptance criteria:

- ``detect_q_triad`` on the L=9 BL=8 reference texture returns
  ``is_triad_q = True`` with ``concentration_ratio > 0.7`` and
  ``balance_ratio < 1.3`` at ``p = 2``.
- A uniform-FM texture returns ``is_triad_q = False`` and
  ``power_finite_q == 0`` (all weight on ``power_q0``).
- A single-Q collinear texture returns ``is_triad_q = False`` (only one
  mode pair has weight; ``balance_ratio = inf`` violates the gate).
- ``normalized_skyrmion_number(BL = 8, p = 2)`` returns
  ``q_bl_total = 8``, ``n_sk = 2``, ``is_skx = True``.
- ``normalized_skyrmion_number`` on a no-moment BL result returns
  ``is_skx = False`` and ``q_bl_total = None``.
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
    single_q_collinear_texture,
    uniform_ferromagnetic_texture,
)

from texture_seeds_skx import (  # noqa: E402
    canted_bloch_skx,
    canted_tetrahedral_skx,
)
from triad_detector import (  # noqa: E402
    detect_q_triad,
    normalized_skyrmion_number,
)


class DetectQTriadTests(unittest.TestCase):
    """Pair-summed triple-Q detector on reference + non-triad textures."""

    def test_canted_tetrahedral_l9_is_balanced_triad_at_p_2(self) -> None:
        cell = MagneticSupercell(9, 9)
        tex = canted_tetrahedral_skx(cell, A=0.219 / float(np.sqrt(6.0)), S_z0=0.0, p=2)
        res = detect_q_triad(tex, cell, p=2)
        self.assertTrue(res["is_triad_q"])
        self.assertGreater(float(res["concentration_ratio"]), 0.7)
        self.assertLess(float(res["balance_ratio"]), 1.3)
        # Three mode-pair powers should be equal (exact tetrahedral symmetry).
        powers = res["mode_powers"]
        self.assertAlmostEqual(powers[0], powers[1], delta=1.0e-10)
        self.assertAlmostEqual(powers[1], powers[2], delta=1.0e-10)

    def test_canted_bloch_with_canting_keeps_triad_signature(self) -> None:
        cell = MagneticSupercell(9, 9)
        tex = canted_bloch_skx(cell, A=0.2, S_z0=0.3, p=2)
        res = detect_q_triad(tex, cell, p=2)
        # Bloch SkX has equal weight on all 3 modes by 120-degree symmetry,
        # and S_z0 canting adds weight at q = 0 (not in the triad denominator).
        self.assertTrue(res["is_triad_q"])
        self.assertGreater(float(res["power_q0"]), 0.0)

    def test_uniform_ferromagnet_is_not_a_triad(self) -> None:
        cell = MagneticSupercell(9, 9)
        tex = uniform_ferromagnetic_texture(cell, amplitude=1.0)
        res = detect_q_triad(tex, cell, p=2)
        self.assertFalse(res["is_triad_q"])
        # All weight on q = 0
        self.assertAlmostEqual(float(res["power_finite_q"]), 0.0, delta=1.0e-12)
        self.assertGreater(float(res["power_q0"]), 0.0)

    def test_single_q_collinear_is_not_a_balanced_triad(self) -> None:
        # A single-Q collinear texture concentrates power on one mode pair;
        # the other two pairs have zero weight, so balance_ratio = inf.
        cell = MagneticSupercell(9, 9)
        tex = single_q_collinear_texture(
            cell, q_frac=(2.0 / 9.0, 0.0), amplitude=0.2, axis=(0.0, 0.0, 1.0)
        )
        res = detect_q_triad(tex, cell, p=2)
        self.assertFalse(res["is_triad_q"])
        # The first mode has all the weight; the other two are zero, so
        # balance_ratio = inf and concentration_ratio = 1.0.
        powers = res["mode_powers"]
        self.assertGreater(powers[0], 0.0)
        self.assertAlmostEqual(powers[1], 0.0, delta=1.0e-12)
        self.assertAlmostEqual(powers[2], 0.0, delta=1.0e-12)
        self.assertEqual(float(res["balance_ratio"]), float("inf"))

    def test_triad_bins_match_commensurate_p_2_on_L9(self) -> None:
        cell = MagneticSupercell(9, 9)
        tex = canted_tetrahedral_skx(cell, A=0.1, S_z0=0.0, p=2)
        res = detect_q_triad(tex, cell, p=2)
        # Q_1 = (2/9, 0)   -> +bin (2, 0), -bin (7, 0)
        # Q_2 = (-2/9, -2/9) -> +bin (7, 7), -bin (2, 2)
        # Q_3 = (0, 2/9)   -> +bin (0, 2), -bin (0, 7)
        expected = [
            ((2, 0), (7, 0)),
            ((7, 7), (2, 2)),
            ((0, 2), (0, 7)),
        ]
        self.assertEqual(res["triad_bins"], expected)


class NormalizedSkyrmionNumberTests(unittest.TestCase):
    """``q_bl_total -> n_sk = q_bl_total / p^2`` with integer-tolerance gate."""

    def test_launchpad_BL_8_p_2_gives_n_sk_2_and_is_skx(self) -> None:
        cell = MagneticSupercell(9, 9)
        tex = canted_tetrahedral_skx(cell, A=0.1, S_z0=0.0, p=2)
        bl = berg_luescher_skyrmion_number(tex, cell)
        res = normalized_skyrmion_number(bl, p=2)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(float(res["q_bl_total"]), 8.0, delta=1.0e-3)
        self.assertAlmostEqual(float(res["n_sk"]), 2.0, delta=1.0e-3)
        self.assertTrue(res["is_skx"])

    def test_undefined_BL_result_is_not_skx(self) -> None:
        result = {"status": "undefined_zero_moment", "min_norm": 0.0, "threshold": 1.0e-10}
        res = normalized_skyrmion_number(result, p=2)
        self.assertEqual(res["status"], "undefined_zero_moment")
        self.assertIsNone(res["q_bl_total"])
        self.assertIsNone(res["n_sk"])
        self.assertFalse(res["is_skx"])

    def test_zero_BL_with_ok_status_is_not_skx(self) -> None:
        # A coplanar-120 in-plane texture is a triple-Q state but is *not*
        # a skyrmion crystal: net Berg-Luescher integer charge is zero.
        result = {"status": "ok", "number": 0.0, "min_norm": 0.05, "threshold": 1.0e-10}
        res = normalized_skyrmion_number(result, p=2)
        self.assertFalse(res["is_skx"])
        self.assertAlmostEqual(float(res["q_bl_total"]), 0.0)
        self.assertAlmostEqual(float(res["n_sk"]), 0.0)

    def test_non_integer_BL_is_not_skx(self) -> None:
        result = {"status": "ok", "number": 7.42, "min_norm": 0.05, "threshold": 1.0e-10}
        res = normalized_skyrmion_number(result, p=2)
        self.assertFalse(res["is_skx"])
        self.assertFalse(res["q_bl_is_integer"])

    def test_invalid_p_raises(self) -> None:
        result = {"status": "ok", "number": 8.0, "min_norm": 0.05, "threshold": 1.0e-10}
        with self.assertRaises(ValueError):
            normalized_skyrmion_number(result, p=0)


if __name__ == "__main__":
    unittest.main()
