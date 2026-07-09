"""Tests for the ``params_for_cell`` helper that prevents
silently-defaulted alpha_rashba / easy_axis_A forwarding bugs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
for path in (_PROJECT_ROOT,):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from hubbard_nesting import TriangularParams  # noqa: E402

from cell_params import params_for_cell  # noqa: E402


class ParamsForCellTests(unittest.TestCase):
    """The helper builds a fully-populated TriangularParams with all
    sweep-relevant fields visible at the call site."""

    def test_default_returns_full_dataclass(self) -> None:
        """Calling with only required alpha sets defaults for everything else."""
        params = params_for_cell(alpha=0.0)
        self.assertIsInstance(params, TriangularParams)
        self.assertEqual(params.alpha_rashba, 0.0)
        self.assertEqual(params.easy_axis_A, 0.0)
        self.assertEqual(params.h_z, 0.0)
        self.assertEqual(params.t, 1.0)
        self.assertEqual(params.beta, 200.0)

    def test_canted_neel_baseline_cell(self) -> None:
        """An example baseline cell: alpha=1.4, A=0.3."""
        params = params_for_cell(alpha=1.4, A=0.3)
        self.assertEqual(params.alpha_rashba, 1.4)
        self.assertEqual(params.easy_axis_A, 0.3)
        self.assertEqual(params.h_z, 0.0)

    def test_alpha_is_keyword_only(self) -> None:
        """alpha must be a keyword arg (forces caller to be explicit)."""
        with self.assertRaises(TypeError):
            # Try to pass alpha positionally — should fail.
            params_for_cell(1.4)  # type: ignore[misc]

    def test_h_z_forwarded(self) -> None:
        """Non-default Zeeman is forwarded correctly."""
        params = params_for_cell(alpha=1.4, A=0.3, h_z=0.1)
        self.assertEqual(params.h_z, 0.1)

    def test_dataclass_is_immutable_like_canonical(self) -> None:
        """params_for_cell returns a regular TriangularParams (the
        result must round-trip through to_dict)."""
        params = params_for_cell(alpha=1.4, A=0.3, h_z=0.05, beta=100.0)
        d = params.to_dict()
        self.assertEqual(d["alpha_rashba"], 1.4)
        self.assertEqual(d["easy_axis_A"], 0.3)
        self.assertEqual(d["h_z"], 0.05)
        self.assertEqual(d["beta"], 100.0)

    def test_matches_explicit_dataclass_construction(self) -> None:
        """params_for_cell must be bit-identical to the explicit
        TriangularParams construction for the same inputs."""
        params_a = params_for_cell(alpha=1.4, A=0.3, h_z=0.0)
        params_b = TriangularParams(
            t=1.0, t2=0.0, t3=0.0, beta=200.0,
            alpha_rashba=1.4, h_z=0.0, easy_axis_A=0.3,
        )
        self.assertEqual(params_a, params_b)


if __name__ == "__main__":
    unittest.main()
