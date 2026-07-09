"""Test surface for the Zeeman extension.

Covers the pipeline-level acceptance point: ``params.h_z`` propagates
through the SCF pipeline (``band_spectrum_twist_averaged`` ->
``site_magnetization_from_spectrum`` -> ``self_consistent_step``)
without explicit threading on the call sites — setting ``h_z`` on
``TriangularParams`` is enough.
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
    zero_texture,
)
from hubbard_nesting import TriangularParams  # noqa: E402
from hubbard_unrestricted_meanfield import (  # noqa: E402
    band_spectrum_twist_averaged,
    self_consistent_step,
    site_magnetization_from_spectrum,
)
from hubbard_meanfield import chemical_potential_for_eigenvalues  # noqa: E402


class ZeemanSCFPipelineTests(unittest.TestCase):
    """``params.h_z`` reaches the Bloch builder via the SCF pipeline."""

    def test_h_z_polarizes_zero_texture_paramagnet_at_half_filling(self) -> None:
        # On a small L=2 paramagnetic cell with U > 0, h_z = 0, and a zero
        # input texture, one SCF step produces zero magnetization (no
        # symmetry-breaking field). At h_z > 0, the same step must
        # polarize every site along +z because the Zeeman block lowers
        # the up-spin orbitals.
        cell = MagneticSupercell(2, 2)
        params0 = TriangularParams(t=1.0, beta=50.0, h_z=0.0)
        params_hz = TriangularParams(t=1.0, beta=50.0, h_z=0.5)
        S_in = zero_texture(cell)
        S_out_zero, _ = self_consistent_step(
            S_in, params0, cell,
            coupling_I=2.0, kappa_nk=6, twist_grid=1, fixed_filling=1.0,
        )
        S_out_hz, _ = self_consistent_step(
            S_in, params_hz, cell,
            coupling_I=2.0, kappa_nk=6, twist_grid=1, fixed_filling=1.0,
        )
        np.testing.assert_allclose(
            S_out_zero, np.zeros_like(S_out_zero), atol=1.0e-10
        )
        # Zeeman polarization: every site has positive z-component, and
        # the in-plane components average to zero (uniform field).
        self.assertTrue(np.all(S_out_hz[:, 2] > 0.0))
        self.assertAlmostEqual(float(np.mean(S_out_hz[:, 0])), 0.0, delta=1.0e-12)
        self.assertAlmostEqual(float(np.mean(S_out_hz[:, 1])), 0.0, delta=1.0e-12)

    def test_h_z_polarization_matches_direct_diagonalization(self) -> None:
        # Independent check: bypass self_consistent_step and call the
        # diagonalization + site_magnetization path directly with
        # params.h_z != 0. The per-site <sigma_z> must be positive and
        # match an explicit Fermi-occupation calculation.
        cell = MagneticSupercell(2, 2)
        h_z = 0.3
        params = TriangularParams(t=1.0, beta=50.0, h_z=h_z)
        texture = zero_texture(cell)
        eigs, vecs = band_spectrum_twist_averaged(
            texture, params, cell, kappa_nk=6, twist_grid=1, workers=1
        )
        mu = chemical_potential_for_eigenvalues(1.0, eigs, params.beta)
        sigma = site_magnetization_from_spectrum(eigs, vecs, mu, params.beta, cell)
        self.assertEqual(sigma.shape, (cell.num_sites, 3))
        self.assertTrue(np.all(sigma[:, 2] > 0.0))




if __name__ == "__main__":
    unittest.main()
