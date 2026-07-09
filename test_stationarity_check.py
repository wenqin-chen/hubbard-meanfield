"""Smoke tests for the stationarity-check helper.

Cheap tests on a small cell (L=4, kappa=12) so the unrestricted SCF
finishes in ~1-2 s per row.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
for path in (_PROJECT_ROOT,):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from hubbard_meanfield import (  # noqa: E402
    MagneticSupercell,
    TriangularParams,
    uniform_ferromagnetic_texture,
)

from stationarity_check import stationarity_check  # noqa: E402
from texture_seeds_skx import canted_tetrahedral_skx  # noqa: E402


class FMSeedStationarityTests(unittest.TestCase):
    """A uniform FM seed at L=4 kappa=12 must:
    - run SCF without error
    - converge under linear mixing
    - stay in the FM family (family_changed=False)
    """

    def test_fm_seed_stays_fm(self) -> None:
        params = TriangularParams(t=1.0, beta=200.0)
        supercell = MagneticSupercell(4, 4)
        # Approximate Ic at L=4 ~ 9.4; use I = 9.0 just below FM crossover
        # so the SCF doesn't immediately blow up; FM should stay FM.
        I = 9.0
        M_seed = 0.5
        seed = uniform_ferromagnetic_texture(supercell, amplitude=M_seed)
        # The restricted-MF F at this seed (one diag, no SCF) — we
        # don't need the actual value, just use a placeholder for the
        # F_gap diagnostic.
        result = stationarity_check(
            seed_texture=seed,
            seed_label="uniform_fm",
            seed_amplitude=M_seed,
            F_restricted=0.0,  # placeholder; F_gap won't be meaningful
            params=params, supercell=supercell,
            filling=0.1134, I=I, h_z=0.0, alpha=0.0,
            kappa_nk=12, twist_grid=1,
            mixing="auto", mixing_alpha=0.3, max_iter=80, tol=1e-4,
            workers=1, commensurate_p=1,
        )
        # FM seed should resolve to "linear" mixing auto-route.
        self.assertEqual(result.extra["mixing_resolved"], "linear")
        # SCF should run without error.
        self.assertNotEqual(result.scf_status[:9], "exception")
        # FM seed should NOT change family (linear mixing keeps it FM).
        self.assertFalse(result.family_changed,
                         f"family_seed={result.family_seed}, "
                         f"family_after_scf={result.family_after_scf}")

    def test_tetrahedral_seed_runs(self) -> None:
        params = TriangularParams(t=1.0, beta=200.0)
        supercell = MagneticSupercell(4, 4)
        I = 9.0
        # canted_tetrahedral_skx at L=4 p=1.
        seed = canted_tetrahedral_skx(
            supercell, A=0.0894, S_z0=0.0, p=1,
        )
        result = stationarity_check(
            seed_texture=seed,
            seed_label="canted_tetrahedral_skx (Sz=0, p=1)",
            seed_amplitude=0.0894,
            F_restricted=0.0,
            params=params, supercell=supercell,
            filling=0.1134, I=I, h_z=0.0, alpha=0.0,
            kappa_nk=12, twist_grid=1,
            mixing="auto", mixing_alpha=0.3, max_iter=40, tol=1e-3,
            workers=1, commensurate_p=1,
        )
        # Should auto-route to pulay (non-uniform_fm label).
        self.assertEqual(result.extra["mixing_resolved"], "pulay")
        # And run without crashing.
        self.assertNotEqual(result.scf_status[:9], "exception")
        # F_scf must be a float (SCF returned a result).
        self.assertIsInstance(result.F_scf, float)


class BLLabelTests(unittest.TestCase):
    """The classifier must distinguish BL=defined-and-zero from
    BL=undefined-due-to-zero-moment-sites."""

    def test_in_plane_bloch_at_sz0_zero_returns_bl_undefined(self) -> None:
        # canted_bloch_skx with S_z0 = 0 has alternating-magnitude
        # sites whose minimum norm trips the BL threshold; BL must
        # be reported as undefined, NOT as "BL=0".
        from hubbard_meanfield import berg_luescher_skyrmion_number  # noqa: E402
        from stationarity_check import _coarse_family_label  # noqa: E402
        from texture_seeds_skx import canted_bloch_skx  # noqa: E402

        supercell = MagneticSupercell(9, 9)
        texture = canted_bloch_skx(
            supercell, A=0.3, S_z0=0.0, p=2,
        )
        # Verify the raw BL helper does flag this as undefined
        bl = berg_luescher_skyrmion_number(texture, supercell)
        self.assertEqual(bl.get("status"), "undefined_zero_moment")
        # And the classifier must NOT lie about it.
        label = _coarse_family_label(texture, supercell, commensurate_p=2)
        self.assertEqual(label, "BL_undefined_triad",
                         f"classifier label was {label!r}; expected "
                         "BL_undefined_triad (BL is undefined because "
                         "the bloch S_z0=0 texture has zero-moment "
                         "sites; saying 'BL=0' would be incorrect).")

    def test_uniform_fm_returns_uniform_M(self) -> None:
        from stationarity_check import _coarse_family_label  # noqa: E402

        supercell = MagneticSupercell(9, 9)
        texture = uniform_ferromagnetic_texture(supercell, amplitude=1.115)
        label = _coarse_family_label(texture, supercell, commensurate_p=1)
        self.assertEqual(label, "uniform_M")

    def test_tetrahedral_sz0_zero_returns_bl_8_triad(self) -> None:
        # The BL=8 reference texture has well-defined BL = 8.
        from stationarity_check import _coarse_family_label  # noqa: E402
        from texture_seeds_skx import canted_tetrahedral_skx  # noqa: E402

        supercell = MagneticSupercell(9, 9)
        texture = canted_tetrahedral_skx(
            supercell, A=0.0894, S_z0=0.0, p=2,
        )
        label = _coarse_family_label(texture, supercell, commensurate_p=2)
        self.assertEqual(label, "BL_8_triad")


class EasyAxisForwardingTests(unittest.TestCase):
    """Regression for a silent-default bug: stationarity_check
    must forward params.easy_axis_A into its internally-rebuilt
    params_with_field. Otherwise any caller passing
    TriangularParams(easy_axis_A=A) silently runs the SCF at A=0.

    Test: at FM-like uniform texture along z (which makes the
    easy-axis term ACTIVE), F_scf must differ between
    easy_axis_A=0 and easy_axis_A=0.5. If the forwarding bug is
    re-introduced, F_scf would be identical.
    """

    def test_easy_axis_A_changes_F_scf_for_uniform_FM_seed(self) -> None:
        from hubbard_meanfield import (  # noqa: E402
            MagneticSupercell, uniform_ferromagnetic_texture,
        )
        from hubbard_nesting import TriangularParams  # noqa: E402

        from stationarity_check import stationarity_check  # noqa: E402

        supercell = MagneticSupercell(4, 4)
        seed = uniform_ferromagnetic_texture(supercell, amplitude=0.5)
        common_kwargs = dict(
            seed_texture=seed,
            seed_label="uniform_fm",
            seed_amplitude=0.5,
            F_restricted=0.0,
            supercell=supercell, filling=0.1134,
            I=9.0, h_z=0.0, alpha=0.0,
            kappa_nk=12, twist_grid=1,
            mixing="linear", mixing_alpha=0.3, max_iter=80, tol=1e-4,
            workers=1, commensurate_p=1, norm_threshold=1e-3,
        )
        r_A0 = stationarity_check(
            params=TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.0),
            **common_kwargs,
        )
        r_A1 = stationarity_check(
            params=TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.5),
            **common_kwargs,
        )
        # F_scf must differ between A=0 and A=0.5 (easy-axis lowers
        # F via -A*M^2 for a uniform FM-along-z seed). If the
        # forwarding bug is back, both runs return identical F.
        self.assertIsNotNone(r_A0.F_scf)
        self.assertIsNotNone(r_A1.F_scf)
        delta = float(r_A1.F_scf) - float(r_A0.F_scf)
        self.assertLess(delta, -1.0e-3,
                        f"F_scf(A=0.5) - F_scf(A=0) = {delta!r}; "
                        "expected < -1e-3 (easy-axis must lower F). "
                        "If this test fails, stationarity_check is "
                        "again failing to forward params.easy_axis_A.")


if __name__ == "__main__":
    unittest.main()
