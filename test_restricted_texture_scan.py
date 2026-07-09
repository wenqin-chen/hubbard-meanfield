"""Unit tests for the restricted fixed-texture MF scan.

The underlying regression gate: at I/Ic_comm = 1.045, h_z = 0 on the
L=9 cell, a one-shot eigvalsh on the canted_tetrahedral_skx seed
ansatz at A = 0.0894 (M = sqrt(6) * A = 0.219) reproduces the
corresponding converged SCF energy *without any SCF iteration*.

That full L=9 regression is too heavy for a unit test (one
diagonalization at L=9, kappa=72, twist=1 takes ~30 s, and the full
scan is hours). The fast smoke tests here use a small cell + small
kappa for ~5 s wall time and exercise the same code paths.
"""

from __future__ import annotations

import os
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
    TriangularParams,
    commensurate_gamma_m_q_fracs,
    uniform_ferromagnetic_texture,
)

from restricted_texture_scan import (  # noqa: E402
    TextureSpec,
    restricted_mf_scan,
)
from texture_seeds_skx import (  # noqa: E402
    canted_bloch_skx,
    canted_tetrahedral_skx,
    dm_cycloid_helix,
)


def _fm_builder(supercell, A):
    return uniform_ferromagnetic_texture(supercell, amplitude=float(A))


def _tetrahedral_builder_factory(S_z0: float, p: int = 2):
    def _build(supercell, A):
        return canted_tetrahedral_skx(
            supercell, A=float(A), S_z0=float(S_z0), p=int(p),
        )
    return _build


def _bloch_builder_factory(S_z0: float, p: int = 2):
    def _build(supercell, A):
        return canted_bloch_skx(
            supercell, A=float(A), S_z0=float(S_z0), p=int(p),
        )
    return _build


def _helix_builder_factory(qx: float, qy: float = 0.0):
    def _build(supercell, A):
        return dm_cycloid_helix(
            supercell, q_vector=(float(qx), float(qy)), A=float(A),
        )
    return _build


class SmallCellAPITests(unittest.TestCase):
    """Minimal smoke tests on a small cell so the scan API is
    exercised in <10 s wall time. Uses L=4 (Phase 3b workhorse cell)
    + kappa=12 + twist=1 with three textures and one (I, h_z, alpha)
    cell."""

    def setUp(self) -> None:
        self.params = TriangularParams(t=1.0, beta=200.0)
        self.supercell = MagneticSupercell(4, 4)
        self.filling = 0.1134
        self.kappa = 12
        self.twist = 1
        self.workers = 1
        # I_c at L=4 is ~9.4 (Phase 3b commensurate); use I = 9.0 just
        # below the FM crossover so several textures compete.
        self.interactions = (9.0,)

    def test_scan_returns_rows_for_each_amplitude(self) -> None:
        textures = (
            TextureSpec(
                label="uniform_fm",
                builder=_fm_builder,
                amplitudes=(0.1, 0.5),
                rms_label="M = A",
            ),
            TextureSpec(
                label="canted_tetrahedral_skx (Sz=0)",
                builder=_tetrahedral_builder_factory(S_z0=0.0, p=1),
                amplitudes=(0.05, 0.10),
                rms_label="M = sqrt(6) * A",
                commensurate_p=1,
                extra_metadata={"S_z0": 0.0, "p": 1},
            ),
        )
        result = restricted_mf_scan(
            params=self.params, supercell=self.supercell,
            filling=self.filling, textures=textures,
            interactions=self.interactions,
            kappa_nk=self.kappa, twist_grid=self.twist,
            workers=self.workers,
        )
        self.assertEqual(result["n_textures"], 2)
        # 2 textures * 2 amplitudes * 1 (I, h_z, alpha) cell = 4 rows
        self.assertEqual(result["n_rows"], 4)
        self.assertEqual(len(result["winners_per_cell"]), 1)
        # Each row carries the required fields
        for row in result["rows"]:
            for k in ("F", "M", "BL", "BL_number", "n_sk", "I", "h_z", "alpha",
                      "texture_label", "amplitude", "wall_seconds"):
                self.assertIn(k, row)
            self.assertIsInstance(row["F"], float)

    def test_per_cell_winner_is_lowest_F(self) -> None:
        textures = (
            TextureSpec(
                label="uniform_fm",
                builder=_fm_builder,
                amplitudes=(0.1,),
                rms_label="M = A",
            ),
            TextureSpec(
                label="canted_tetrahedral_skx (Sz=0)",
                builder=_tetrahedral_builder_factory(S_z0=0.0, p=1),
                amplitudes=(0.05,),
                rms_label="M = sqrt(6) * A",
                commensurate_p=1,
            ),
        )
        result = restricted_mf_scan(
            params=self.params, supercell=self.supercell,
            filling=self.filling, textures=textures,
            interactions=self.interactions,
            kappa_nk=self.kappa, twist_grid=self.twist,
            workers=self.workers,
        )
        self.assertEqual(len(result["winners_per_cell"]), 1)
        winner = result["winners_per_cell"][0]
        # The winner's F must equal the min F over the rows.
        F_min = min(r["F"] for r in result["rows"])
        self.assertAlmostEqual(winner["winner_F"], F_min, delta=1e-12)

    def test_h_z_grid_iterates(self) -> None:
        # 1 texture * 1 amplitude * 1 I * 2 h_z = 2 rows
        textures = (
            TextureSpec(
                label="uniform_fm",
                builder=_fm_builder,
                amplitudes=(0.3,),
                rms_label="M = A",
            ),
        )
        result = restricted_mf_scan(
            params=self.params, supercell=self.supercell,
            filling=self.filling, textures=textures,
            interactions=self.interactions, h_z_list=(0.0, 0.05),
            kappa_nk=self.kappa, twist_grid=self.twist,
            workers=self.workers,
        )
        self.assertEqual(result["n_rows"], 2)
        # F at h_z != 0 differs from F at h_z = 0 (Zeeman shifts mu).
        F_h0 = next(r["F"] for r in result["rows"] if r["h_z"] == 0.0)
        F_h_pos = next(r["F"] for r in result["rows"] if r["h_z"] == 0.05)
        self.assertNotAlmostEqual(F_h0, F_h_pos, delta=1e-8)


class TextureFamilyDiagnosticsTests(unittest.TestCase):
    """Per-family textures must trigger the right diagnostics:
    canted_tetrahedral_skx at S_z0=0 has BL=8 on L=9; uniform_fm
    is a trivial texture (BL undefined / 0); helix is rank-2.
    """

    def test_tetrahedral_at_L9_carries_BL8(self) -> None:
        # L=9 cell, p=2: canted_tetrahedral_skx at A=0.0894 must give
        # BL = 8 (this seed family's topology).
        params = TriangularParams(t=1.0, beta=200.0)
        supercell = MagneticSupercell(9, 9)
        textures = (
            TextureSpec(
                label="canted_tetrahedral_skx (Sz=0,p=2)",
                builder=_tetrahedral_builder_factory(S_z0=0.0, p=2),
                amplitudes=(0.0894,),
                rms_label="M = sqrt(6) * A",
                commensurate_p=2,
            ),
        )
        # Use a tiny kappa just to populate the row; the diagnostic
        # is the BL of the texture, NOT the F value.
        result = restricted_mf_scan(
            params=params, supercell=supercell,
            filling=0.1134, textures=textures,
            interactions=(9.847,),  # I = 1.045 * Ic_comm at L=9
            kappa_nk=4, twist_grid=1, workers=1,
        )
        self.assertEqual(result["n_rows"], 1)
        row = result["rows"][0]
        self.assertEqual(round(row["BL_number"]), 8)
        # M = sqrt(6) * 0.0894 ~= 0.219
        self.assertAlmostEqual(row["M"], np.sqrt(6.0) * 0.0894, delta=0.001)


if __name__ == "__main__":
    unittest.main()
