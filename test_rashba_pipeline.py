"""Test surface for the Rashba extension.

``build_supercell_hamiltonian`` appends ``-i alpha [(zhat x dhat)
. sigma]`` per directed NN bond when ``params.alpha_rashba != 0``.
This file covers the pipeline-level gate: ``params.alpha_rashba``
propagates through the SCF pipeline (``self_consistent_step`` /
``band_spectrum_twist_averaged`` /
``site_magnetization_from_spectrum``) without explicit threading
on the call sites — setting ``alpha_rashba`` on
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
)


class RashbaSCFPipelineTests(unittest.TestCase):
    """``params.alpha_rashba`` reaches the Bloch builder via the SCF
    pipeline."""

    def test_alpha_rashba_perturbs_paramagnetic_step(self) -> None:
        # On a small L=2 paramagnetic cell with U > 0, alpha = 0, and a
        # zero input texture, one SCF step produces zero magnetization
        # (no symmetry-breaking). At alpha > 0 the in-plane Rashba
        # texture is still zero on a zero-input texture (Rashba is a
        # kinetic term, not a Zeeman field) -- but the band spectrum
        # changes, so the SCF step's reported energy must differ.
        cell = MagneticSupercell(2, 2)
        S_in = zero_texture(cell)
        p_zero = TriangularParams(t=1.0, beta=50.0, alpha_rashba=0.0)
        p_alpha = TriangularParams(t=1.0, beta=50.0, alpha_rashba=0.2)
        _, info_zero = self_consistent_step(
            S_in, p_zero, cell,
            coupling_I=2.0, kappa_nk=6, twist_grid=1, fixed_filling=1.0,
        )
        _, info_alpha = self_consistent_step(
            S_in, p_alpha, cell,
            coupling_I=2.0, kappa_nk=6, twist_grid=1, fixed_filling=1.0,
        )
        self.assertNotAlmostEqual(
            info_zero["energy"], info_alpha["energy"], delta=1.0e-9,
        )

    def test_alpha_rashba_spinor_eigenvectors_have_in_plane_components(self) -> None:
        # Direct diag: at alpha > 0 on a zero texture, the spinor
        # eigenvectors at a generic k have nonzero off-diagonal weight
        # (the bands are spin-split with momentum-locked spinors).
        cell = MagneticSupercell(2, 2)
        params = TriangularParams(t=1.0, beta=50.0, alpha_rashba=0.15)
        eigs, vecs = band_spectrum_twist_averaged(
            zero_texture(cell), params, cell,
            kappa_nk=6, twist_grid=1, workers=1,
        )
        # vecs shape: (twist=1, nk, nk, 2N_c, 2N_c). Average squared
        # off-diagonal weight in each eigenvector should be nonzero.
        # Off-diagonal in our basis: |<even-index| vec[:, lam]>|^2 +
        # ... no — easier: pick a sample (twist, kx, ky, lam) and
        # check the eigenvector has both spin components nonzero on at
        # least one site.
        sample = vecs[0, 0, 0, :, 0]  # eigenvector of band lam=0 at one k
        # Sites are ordered (site 0 up, site 0 down, site 1 up, ...).
        # Check that for at least one site, both up and down weights
        # are nonzero.
        for site in range(cell.num_sites):
            up = sample[2 * site]
            dn = sample[2 * site + 1]
            if abs(up) > 1.0e-6 and abs(dn) > 1.0e-6:
                return
        self.fail(
            "No site has both spin components nonzero in the lowest-band "
            "Rashba eigenvector; the alpha_rashba field is not "
            "propagating to the Bloch builder."
        )




if __name__ == "__main__":
    unittest.main()
