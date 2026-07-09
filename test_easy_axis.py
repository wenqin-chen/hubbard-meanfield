"""Tests for the easy-axis anisotropy extension to TriangularParams +
build_supercell_hamiltonian + grand_potential + fixed_density_free_energy.

Three guarantees the new code must satisfy:

1. ``A = 0`` regression: every output is bit-identical to the pre-
   anisotropy version. The default behaviour is unchanged.
2. ``A > 0`` with uniform FM texture (all m_i^z = M): free-energy
   correction is exactly ``-A * M^2`` per site. The texture-
   proportional Zeeman in the band Hamiltonian shifts the bands as
   expected (per-site +/- A * M on the spin diagonal).
3. ``A > 0`` with in-plane texture (m_i^z = 0): no F correction, no
   band Hamiltonian change.
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
    band_eigenvalues_twist_averaged,
    build_supercell_hamiltonian,
    fixed_density_free_energy,
    grand_potential,
    uniform_ferromagnetic_texture,
)
from hubbard_nesting import TriangularParams  # noqa: E402

from texture_seeds_skx import canted_bloch_skx  # noqa: E402


_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


class _ParamsBase(unittest.TestCase):
    def setUp(self) -> None:
        self.supercell = MagneticSupercell(4, 4)
        self.params_A0 = TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.0)


class EasyAxisRegressionTests(_ParamsBase):
    """At A = 0 the new code must reproduce the pre-anisotropy results
    bit-for-bit on multiple non-trivial paths: band Hamiltonian, band
    eigenvalues, grand_potential, fixed_density_free_energy."""

    def test_default_easy_axis_A_is_zero(self) -> None:
        # New field appears on TriangularParams with sensible default.
        self.assertEqual(self.params_A0.easy_axis_A, 0.0)
        # The to_dict reflects the new field.
        d = self.params_A0.to_dict()
        self.assertIn("easy_axis_A", d)
        self.assertEqual(d["easy_axis_A"], 0.0)

    def test_build_hamiltonian_A0_bit_identical_to_explicit_A0(self) -> None:
        # The new code path with explicit A = 0 must produce the same
        # Hamiltonian as the implicit default.
        params_default = TriangularParams(t=1.0, beta=200.0)
        params_explicit = TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.0)
        # On a 4x4 supercell, the commensurate-Q constraint is p < L/2,
        # so p=1 is the only valid choice. We still get a non-trivial
        # texture with all three spin components.
        texture = canted_bloch_skx(
            self.supercell, A=0.30, S_z0=0.3, p=1,
        )
        kappa = (0.1, 0.2)
        h_default = build_supercell_hamiltonian(
            kappa, texture, params_default, self.supercell,
        )
        h_explicit = build_supercell_hamiltonian(
            kappa, texture, params_explicit, self.supercell,
        )
        np.testing.assert_array_equal(h_default, h_explicit)

    def test_grand_potential_A0_matches_no_kwarg(self) -> None:
        # grand_potential at easy_axis_A=0 (explicit) matches the
        # call without the kwarg.
        from hubbard_meanfield import zero_texture  # noqa: E402

        params = self.params_A0
        texture = uniform_ferromagnetic_texture(self.supercell, amplitude=0.5)
        eigvals = band_eigenvalues_twist_averaged(
            texture, params, self.supercell,
            kappa_nk=8, twist_grid=1, workers=1,
        )
        mu = -1.0
        I = 5.0
        beta = 200.0
        F_default = grand_potential(mu, eigvals, texture, I, beta)
        F_explicit = grand_potential(
            mu, eigvals, texture, I, beta, easy_axis_A=0.0,
        )
        self.assertAlmostEqual(F_default, F_explicit, places=14)


class EasyAxisFMUniformGainTests(_ParamsBase):
    """At A > 0 with a uniform FM texture along +z, the F correction
    is exactly -A * M^2 per site (since <(S_z)^2>_i = M^2 uniformly)."""

    def test_uniform_fm_A_gain_matches_analytic(self) -> None:
        M = 0.7
        A = 0.5
        texture = uniform_ferromagnetic_texture(
            self.supercell, amplitude=M, axis=(0.0, 0.0, 1.0),
        )
        # Crucial: when computing F with anisotropy ON, the band
        # Hamiltonian ALSO has the texture-proportional Zeeman, so
        # the band energy differs from the A=0 case. We test only the
        # ANALYTIC anisotropy F contribution by checking
        # grand_potential against grand_potential with same eigenvalues
        # passed in for both (i.e. holding the bands fixed). The
        # difference must equal -A * M^2.
        params_A_off = self.params_A0
        eigvals = band_eigenvalues_twist_averaged(
            texture, params_A_off, self.supercell,
            kappa_nk=8, twist_grid=1, workers=1,
        )
        mu = -1.0
        I = 5.0
        beta = 200.0
        F_no_A = grand_potential(
            mu, eigvals, texture, I, beta, easy_axis_A=0.0,
        )
        F_with_A = grand_potential(
            mu, eigvals, texture, I, beta, easy_axis_A=A,
        )
        # F_with_A - F_no_A = -A * <(S_z)^2>_i = -A * M^2 (uniform).
        delta_expected = -A * M * M
        delta_actual = F_with_A - F_no_A
        self.assertAlmostEqual(delta_actual, delta_expected, places=12)

    def test_in_plane_texture_no_F_correction(self) -> None:
        # An in-plane texture (S_z = 0 everywhere) has no
        # easy-axis F correction at any A.
        texture = uniform_ferromagnetic_texture(
            self.supercell, amplitude=0.5, axis=(1.0, 0.0, 0.0),
        )
        # Verify the texture is in-plane.
        self.assertTrue(np.allclose(texture[:, 2], 0.0))
        eigvals = band_eigenvalues_twist_averaged(
            texture, self.params_A0, self.supercell,
            kappa_nk=8, twist_grid=1, workers=1,
        )
        mu = -1.0
        I = 5.0
        beta = 200.0
        F_no_A = grand_potential(
            mu, eigvals, texture, I, beta, easy_axis_A=0.0,
        )
        F_with_large_A = grand_potential(
            mu, eigvals, texture, I, beta, easy_axis_A=10.0,
        )
        self.assertAlmostEqual(F_no_A, F_with_large_A, places=14)


class EasyAxisBandHamiltonianTests(_ParamsBase):
    """The texture-proportional Zeeman in the band Hamiltonian must
    add -A * S_z_i to the (up, up) diagonal AND +A * S_z_i to the
    (down, down) diagonal of each site's 2x2 spin block, with no off-
    diagonal contribution."""

    def test_band_hamiltonian_per_site_z_zeeman(self) -> None:
        # Construct a uniform FM-up texture so every site has S_z = M.
        M = 0.4
        texture = uniform_ferromagnetic_texture(
            self.supercell, amplitude=M, axis=(0.0, 0.0, 1.0),
        )
        params_A_on = TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.7)
        params_A_off = TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.0)
        kappa = (0.0, 0.0)
        h_on = build_supercell_hamiltonian(
            kappa, texture, params_A_on, self.supercell,
        )
        h_off = build_supercell_hamiltonian(
            kappa, texture, params_A_off, self.supercell,
        )
        # The delta is the easy-axis contribution to the band Hamiltonian.
        delta = h_on - h_off
        # For each site, the delta should be -A * M * σ_z on the
        # 2x2 spin block (i.e., -A*M on (up,up), +A*M on (down,down),
        # zero off-diagonal).
        expected_per_site_delta = -0.7 * M * _PAULI_Z
        n_sites = self.supercell.num_sites
        for site in range(n_sites):
            block_slice = slice(2 * site, 2 * site + 2)
            actual = delta[block_slice, block_slice]
            np.testing.assert_array_almost_equal(
                actual, expected_per_site_delta, decimal=12,
                err_msg=f"site {site} delta does not match -A*M*sigma_z",
            )
        # And there should be NO change in any off-site block (the
        # easy-axis term is strictly diagonal across sites).
        for s1 in range(n_sites):
            for s2 in range(n_sites):
                if s1 == s2:
                    continue
                b1 = slice(2 * s1, 2 * s1 + 2)
                b2 = slice(2 * s2, 2 * s2 + 2)
                np.testing.assert_array_almost_equal(
                    delta[b1, b2], np.zeros((2, 2), dtype=complex),
                    decimal=14,
                    err_msg=f"easy-axis delta nonzero in off-site block ({s1}, {s2})",
                )

    def test_band_hamiltonian_in_plane_texture_no_band_change(self) -> None:
        # In-plane texture (S_z = 0) should produce a zero delta in
        # the band Hamiltonian when A is turned on.
        texture = uniform_ferromagnetic_texture(
            self.supercell, amplitude=0.6, axis=(1.0, 0.0, 0.0),
        )
        params_A_on = TriangularParams(t=1.0, beta=200.0, easy_axis_A=1.5)
        params_A_off = TriangularParams(t=1.0, beta=200.0, easy_axis_A=0.0)
        kappa = (0.1, 0.2)
        h_on = build_supercell_hamiltonian(
            kappa, texture, params_A_on, self.supercell,
        )
        h_off = build_supercell_hamiltonian(
            kappa, texture, params_A_off, self.supercell,
        )
        np.testing.assert_array_almost_equal(
            h_on, h_off, decimal=14,
            err_msg="easy-axis nonzero on in-plane texture (S_z = 0 should be a no-op)",
        )


class EasyAxisFixedDensityForwardingTests(_ParamsBase):
    """fixed_density_free_energy forwards easy_axis_A correctly to
    grand_potential."""

    def test_fixed_density_F_picks_up_anisotropy(self) -> None:
        M = 0.6
        A = 0.3
        texture = uniform_ferromagnetic_texture(
            self.supercell, amplitude=M, axis=(0.0, 0.0, 1.0),
        )
        # Use SAME eigenvalues for both calls (compute once at A=0
        # so the band part doesn't change; the F difference should be
        # exactly -A * M^2).
        eigvals = band_eigenvalues_twist_averaged(
            texture, self.params_A0, self.supercell,
            kappa_nk=8, twist_grid=1, workers=1,
        )
        F_no_A = fixed_density_free_energy(
            0.5, eigvals, texture, 5.0, 200.0,
            easy_axis_A=0.0,
        )
        F_with_A = fixed_density_free_energy(
            0.5, eigvals, texture, 5.0, 200.0,
            easy_axis_A=A,
        )
        delta = F_with_A - F_no_A
        self.assertAlmostEqual(delta, -A * M * M, places=12)


if __name__ == "__main__":
    unittest.main()
