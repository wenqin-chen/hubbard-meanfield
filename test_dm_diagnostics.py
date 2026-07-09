"""Unit tests for ``dm_diagnostics``."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dm_diagnostics import strong_coupling_DM_ratio  # noqa: E402


class StrongCouplingDMTests(unittest.TestCase):
    def test_lambda_times_alpha_equals_2_pi_t(self) -> None:
        # Identity: lambda_DM * alpha = 2 pi t (a = 1).
        for t, U, alpha in [
            (1.0, 28.0, 0.05),
            (1.0, 28.0, 0.1),
            (1.0, 28.0, 0.2),
            (2.0, 50.0, 0.3),
        ]:
            res = strong_coupling_DM_ratio(t, U, alpha)
            self.assertAlmostEqual(
                res["lambda_DM_over_a"] * abs(alpha), 2.0 * math.pi * t,
                delta=1.0e-12,
            )

    def test_D_over_J_equals_alpha_over_t(self) -> None:
        # At second-order strong-coupling, the Hubbard U drops out of
        # the dimensionless DM ratio.
        for t, U, alpha in [(1.0, 28.0, 0.1), (1.0, 50.0, 0.1)]:
            res = strong_coupling_DM_ratio(t, U, alpha)
            self.assertAlmostEqual(res["D_over_J"], alpha / t, delta=1.0e-14)

    def test_zero_alpha_gives_no_DM(self) -> None:
        res = strong_coupling_DM_ratio(1.0, 28.0, 0.0)
        self.assertEqual(res["regime"], "no_DM")
        self.assertEqual(res["D_over_J"], 0.0)
        self.assertEqual(res["D_abs"], 0.0)
        self.assertEqual(res["lambda_DM_over_a"], float("inf"))

    def test_regime_classifier_alpha_in_skx_window(self) -> None:
        # The SkX regime is 1 <= lambda_DM/a <= 10 (Wang-2020 compatible
        # window where the DM length-scale fits a few-cell SkX). Pick
        # alpha = 0.7 (lambda/a = 2 pi / 0.7 approx 8.98) — well inside.
        # Note: the L=9 reference cell uses Q = p * b_1 / L with p = 2
        # (not p = 1), so the SkX wavelength on that cell is L/p =
        # 9/2 = 4.5 a, not 9 a. The classifier is independent of cell
        # geometry; whether a given alpha "fits" a given cell is
        # a separate per-cell question handled by a triage step that
        # compares lambda_DM/a against L = 9 directly, not against L/p.
        res = strong_coupling_DM_ratio(1.0, 28.0, 0.7)
        self.assertEqual(res["regime"], "skx_regime")
        self.assertAlmostEqual(
            res["lambda_DM_over_a"], 2.0 * math.pi / 0.7, delta=1.0e-12,
        )

    def test_weak_DM_regime_for_small_alpha(self) -> None:
        # alpha = 0.1 gives lambda/a = 2 pi / 0.1 approx 62.8 > 10.
        res = strong_coupling_DM_ratio(1.0, 28.0, 0.1)
        self.assertEqual(res["regime"], "weak_DM")
        self.assertGreater(res["lambda_DM_over_a"], 10.0)

    def test_strong_DM_regime_for_large_alpha(self) -> None:
        # alpha = 10 gives lambda/a = 2 pi / 10 approx 0.628 < 1.
        res = strong_coupling_DM_ratio(1.0, 28.0, 10.0)
        self.assertEqual(res["regime"], "strong_DM")
        self.assertLess(res["lambda_DM_over_a"], 1.0)

    def test_invalid_t_or_U_raises(self) -> None:
        with self.assertRaises(ValueError):
            strong_coupling_DM_ratio(0.0, 28.0, 0.1)
        with self.assertRaises(ValueError):
            strong_coupling_DM_ratio(1.0, 0.0, 0.1)
        with self.assertRaises(ValueError):
            strong_coupling_DM_ratio(-1.0, 28.0, 0.1)


if __name__ == "__main__":
    unittest.main()
