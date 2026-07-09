"""Unrestricted self-consistent mean-field utilities for the Hubbard model.

Phase 4 module.  Lets the Hubbard-Stratonovich field ``S_i`` in
``R^3`` vary independently at every site of a magnetic supercell, with the
saddle-point condition ``S_i = I <sigma_i>`` enforced pointwise via
self-consistency.

The single-particle physics (Hamiltonian builder, twist grid, mu solver,
free-energy evaluator) is reused verbatim from :mod:`hubbard_meanfield`.
This module adds:

- :func:`band_spectrum_on_grid` -- ``eigh`` on one reduced-zone grid
  (eigenvalues *and* eigenvectors).
- :func:`band_spectrum_twist_averaged` -- ``eigh`` over all twist grids.
- :func:`site_magnetization_from_spectrum` -- ``<sigma_i>`` from occupied
  eigenvectors, twist-and-kappa averaged.

Phase 4.0 establishes this foundation.  Phase 4.1 will add the SCF loop with
mixing schemes and the seed library.

Conventions
-----------
The 2 N_c basis is ordered ``(site=0, spin=up), (site=0, spin=down),
(site=1, spin=up), ...``, matching :func:`hubbard_meanfield.build_supercell_hamiltonian`
where the on-site exchange block is inserted at rows/columns
``slice(2*site, 2*site + 2)``.

Eigenvector arrays are returned as ``V`` with ``V[..., :, lam]`` the
``lam``-th eigenvector at the given batch index.  This matches NumPy's
:func:`numpy.linalg.eigh` convention.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hubbard_meanfield import (
    KappaGrid,
    MagneticSupercell,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    band_eigenvalues_twist_averaged,
    build_supercell_hamiltonian,
    chemical_potential_for_eigenvalues,
    fixed_density_free_energy,
    kappa_twist_grids,
    texture_from_mode_spinors,
    validate_texture,
    zero_texture,
)
from hubbard_nesting import TriangularParams, fermi_function


__all__ = [
    "band_spectrum_on_grid",
    "band_spectrum_twist_averaged",
    "site_magnetization_from_spectrum",
    "site_magnetization_streaming",
    "self_consistent_step",
    "self_consistent_solve",
    "random_seed_perturbation",
    "extract_dominant_q",
]


def _spectrum_kappa_row(
    task: tuple[
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        TriangularParams,
        MagneticSupercell,
    ],
) -> tuple[int, NDArray[np.float64], NDArray[np.complex128]]:
    """Diagonalize one row of a kappa grid; return (row_index, E, V)."""

    row_index, kx_row, ky_row, spin_field, params, supercell = task
    bands = 2 * supercell.num_sites
    nk = int(kx_row.size)
    row_eigvals = np.empty((nk, bands), dtype=float)
    row_eigvecs = np.empty((nk, bands, bands), dtype=np.complex128)
    for j in range(nk):
        kappa = (float(kx_row[j]), float(ky_row[j]))
        hamiltonian = build_supercell_hamiltonian(
            kappa, spin_field, params, supercell
        )
        eigvals, eigvecs = np.linalg.eigh(hamiltonian)
        row_eigvals[j, :] = eigvals
        row_eigvecs[j, :, :] = eigvecs
    return int(row_index), row_eigvals, row_eigvecs


def _spectrum_twist_kappa_row(
    task: tuple[
        int,
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        TriangularParams,
        MagneticSupercell,
    ],
) -> tuple[int, int, NDArray[np.float64], NDArray[np.complex128]]:
    """Diagonalize one (grid, row) tile; return (grid, row, E, V)."""

    grid_index, row_index, kx_row, ky_row, spin_field, params, supercell = task
    _, row_eigvals, row_eigvecs = _spectrum_kappa_row(
        (row_index, kx_row, ky_row, spin_field, params, supercell)
    )
    return int(grid_index), int(row_index), row_eigvals, row_eigvecs


def band_spectrum_on_grid(
    texture: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    grid: KappaGrid,
    *,
    workers: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Diagonalize the supercell Hamiltonian on one reduced-zone grid.

    Returns ``(eigenvalues, eigenvectors)`` with shapes::

        eigenvalues:  (nk, nk, 2 N_c)
        eigenvectors: (nk, nk, 2 N_c, 2 N_c)

    where ``eigenvectors[i, j, :, lam]`` is the ``lam``-th eigenvector of
    the Hamiltonian at the ``(i, j)`` reduced-zone point.  Uses
    :func:`numpy.linalg.eigh` (eigenvalues *and* eigenvectors), unlike
    :func:`hubbard_meanfield.band_eigenvalues_on_grid` which uses
    ``eigvalsh`` (eigenvalues only).

    ``workers > 1`` dispatches one task per row of the kappa grid to a
    :class:`concurrent.futures.ProcessPoolExecutor`, mirroring the
    parallelization pattern of
    :func:`hubbard_meanfield.band_eigenvalues_on_grid`.
    """

    if grid.supercell != supercell:
        raise ValueError("grid.supercell must match the Hamiltonian supercell.")
    spin_field = validate_texture(texture, supercell)
    bands = 2 * supercell.num_sites
    nk = int(grid.nk)
    energies = np.empty((nk, nk, bands), dtype=float)
    vectors = np.empty((nk, nk, bands, bands), dtype=np.complex128)
    kx_mesh, ky_mesh = grid.cartesian_mesh()
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be >= 1.")
    if worker_count > 1 and nk > 1:
        tasks = [
            (i, kx_mesh[i, :], ky_mesh[i, :], spin_field, params, supercell)
            for i in range(nk)
        ]
        with ProcessPoolExecutor(max_workers=min(worker_count, nk)) as executor:
            for i, row_eigvals, row_eigvecs in executor.map(
                _spectrum_kappa_row, tasks
            ):
                energies[i, :, :] = row_eigvals
                vectors[i, :, :, :] = row_eigvecs
        return energies, vectors
    for i in range(nk):
        for j in range(nk):
            kappa = (float(kx_mesh[i, j]), float(ky_mesh[i, j]))
            hamiltonian = build_supercell_hamiltonian(
                kappa, spin_field, params, supercell
            )
            eigvals, eigvecs = np.linalg.eigh(hamiltonian)
            energies[i, j, :] = eigvals
            vectors[i, j, :, :] = eigvecs
    return energies, vectors


def band_spectrum_twist_averaged(
    texture: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    *,
    kappa_nk: int,
    twist_grid: int = 1,
    workers: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return ``(eigenvalues, eigenvectors)`` on all reduced-zone twist grids.

    Output shapes::

        eigenvalues:  (twist_count, kappa_nk, kappa_nk, 2 N_c)
        eigenvectors: (twist_count, kappa_nk, kappa_nk, 2 N_c, 2 N_c)

    The twist count is ``twist_grid ** 2`` and matches the convention used
    by :func:`hubbard_meanfield.band_eigenvalues_twist_averaged`.

    With ``workers > 1`` the ``twist_count * kappa_nk`` rows are
    distributed across a :class:`concurrent.futures.ProcessPoolExecutor`.
    """

    grids = kappa_twist_grids(int(kappa_nk), supercell, int(twist_grid))
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be >= 1.")
    bands = 2 * supercell.num_sites
    twist_count = len(grids)

    if worker_count > 1 and twist_count * int(kappa_nk) > 1:
        spin_field = validate_texture(texture, supercell)
        energies = np.empty(
            (twist_count, int(kappa_nk), int(kappa_nk), bands), dtype=float
        )
        vectors = np.empty(
            (
                twist_count,
                int(kappa_nk),
                int(kappa_nk),
                bands,
                bands,
            ),
            dtype=np.complex128,
        )
        tasks = []
        for grid_index, grid in enumerate(grids):
            kx_mesh, ky_mesh = grid.cartesian_mesh()
            for row_index in range(int(grid.nk)):
                tasks.append(
                    (
                        grid_index,
                        row_index,
                        kx_mesh[row_index, :],
                        ky_mesh[row_index, :],
                        spin_field,
                        params,
                        supercell,
                    )
                )
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(tasks))
        ) as executor:
            for grid_index, row_index, row_eigvals, row_eigvecs in executor.map(
                _spectrum_twist_kappa_row, tasks
            ):
                energies[grid_index, row_index, :, :] = row_eigvals
                vectors[grid_index, row_index, :, :, :] = row_eigvecs
        return energies, vectors

    spectra = [
        band_spectrum_on_grid(texture, params, supercell, grid, workers=1)
        for grid in grids
    ]
    energies = np.stack([energy for energy, _ in spectra], axis=0)
    vectors = np.stack([vec for _, vec in spectra], axis=0)
    return energies, vectors


def site_magnetization_from_spectrum(
    eigenvalues: ArrayLike,
    eigenvectors: ArrayLike,
    mu: float,
    beta: float,
    supercell: MagneticSupercell,
) -> NDArray[np.float64]:
    """Compute ``<sigma_i>`` per site from a pre-computed spectrum.

    Parameters
    ----------
    eigenvalues
        Real array with shape ``(..., 2 N_c)``.  The leading axes are
        averaged (e.g. ``(twist, nk, nk)`` from
        :func:`band_spectrum_twist_averaged`); the last axis indexes the
        ``2 N_c`` bands.
    eigenvectors
        Complex array with shape ``(..., 2 N_c, 2 N_c)``.  ``V[..., :, lam]``
        is the ``lam``-th eigenvector of the supercell Hamiltonian at the
        given batch index.  The leading axes must match those of
        ``eigenvalues``.
    mu
        Chemical potential.
    beta
        Inverse temperature ``1 / k_B T``.
    supercell
        Magnetic supercell.  Used only to extract ``N_c``.

    Returns
    -------
    sigma : ndarray, shape ``(N_c, 3)``
        Per-site spin-density expectation value ``<sigma_a>`` averaged over
        the leading batch axes.

    Notes
    -----
    The formula evaluated is

    .. math::

        \\langle\\sigma^\\mu_a\\rangle
        = \\overline{\\sum_\\lambda f(E_\\lambda - \\mu)\\,
        \\sum_{\\alpha\\beta} V^*_{2a + \\alpha,\\,\\lambda}\\,
        \\sigma^\\mu_{\\alpha\\beta}\\,
        V_{2a + \\beta,\\,\\lambda}},

    where the overline denotes averaging over the leading batch axes
    (twist offset, reduced-zone momentum) and
    :math:`f(E) = 1/(e^{\\beta E} + 1)` is the Fermi function.  The
    eigenvector basis is ordered ``(site=0, spin=up), (site=0, spin=down),
    (site=1, spin=up), ...``, consistent with
    :func:`hubbard_meanfield.build_supercell_hamiltonian`.

    The Hermiticity of the per-site one-body density matrix and the
    Hermiticity of the Pauli matrices guarantee that the result is real;
    a small imaginary residual from finite-precision arithmetic is
    discarded by taking the real part.
    """

    energies = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=complex)

    n_sites = supercell.num_sites
    bands = 2 * n_sites

    if energies.shape[-1] != bands:
        raise ValueError(
            f"last axis of eigenvalues must be 2*N_c={bands}, "
            f"got {energies.shape[-1]}."
        )
    if vectors.shape[-2:] != (bands, bands):
        raise ValueError(
            f"last two axes of eigenvectors must be ({bands}, {bands}), "
            f"got {vectors.shape[-2:]}."
        )
    if vectors.shape[:-2] != energies.shape[:-1]:
        raise ValueError(
            "eigenvalues and eigenvectors must share the same batch shape "
            f"(got {energies.shape[:-1]} vs {vectors.shape[:-2]})."
        )

    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive.")

    weights = fermi_function(energies - float(mu), beta_value)  # (..., 2 N_c)

    # Reshape eigenvectors so the (site, spin) basis index splits into
    # (site, spin) axes:
    # vectors:   (..., 2 N_c, 2 N_c)
    # split[..., a, alpha, lam] = vectors[..., 2*a + alpha, lam]
    batch_shape = vectors.shape[:-2]
    split = vectors.reshape(*batch_shape, n_sites, 2, bands)

    sigma = np.empty((*batch_shape, n_sites, 3), dtype=float)
    for axis_idx, pauli in enumerate((PAULI_X, PAULI_Y, PAULI_Z)):
        # <sigma^mu_a>_batch
        # = sum_lambda f_lambda
        #   sum_{alpha, beta} V*_{a, alpha, lambda}
        #                     pauli_{alpha, beta}
        #                     V_{a, beta, lambda}
        expectation = np.einsum(
            "...ial,ab,...ibl,...l->...i",
            np.conj(split),
            pauli,
            split,
            weights,
        )
        sigma[..., axis_idx] = expectation.real

    if len(batch_shape) > 0:
        sigma_avg = np.mean(sigma, axis=tuple(range(len(batch_shape))))
    else:
        sigma_avg = sigma

    return np.asarray(sigma_avg, dtype=float)


def _streaming_kappa_row_sigma(
    task: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        TriangularParams,
        MagneticSupercell,
        float,
        float,
    ],
) -> tuple[NDArray[np.float64], float, float, int]:
    """Worker: diagonalize one ``(kappa_row)`` tile, accumulate sigma, discard V.

    Returns ``(sigma_row_sum, eigval_min, eigval_max, n_points)`` where
    ``sigma_row_sum`` is the un-normalized per-site spin-density sum over the
    points in the row.  The caller is responsible for dividing by the total
    point count after aggregating all rows / twists.

    Memory cost is ``O((2 N_c)^2)`` complex128 per worker (one ``eigh``
    result at a time), independent of ``kappa_nk``.
    """

    kx_row, ky_row, texture, params, supercell, mu, beta = task
    n_sites = supercell.num_sites
    bands = 2 * n_sites
    nk = int(kx_row.size)
    sigma_row_sum = np.zeros((n_sites, 3), dtype=float)
    eigval_min = float("inf")
    eigval_max = float("-inf")
    for j in range(nk):
        kappa = (float(kx_row[j]), float(ky_row[j]))
        hamiltonian = build_supercell_hamiltonian(
            kappa, texture, params, supercell
        )
        eigvals, eigvecs = np.linalg.eigh(hamiltonian)
        weights = fermi_function(eigvals - float(mu), float(beta))
        # Same bilinear formula as site_magnetization_from_spectrum, but
        # at a single (twist, kappa) point so no batch axes.
        v_split = eigvecs.reshape(n_sites, 2, bands)
        for axis_idx, pauli in enumerate((PAULI_X, PAULI_Y, PAULI_Z)):
            contribution = np.einsum(
                "ial,ab,ibl,l->i",
                np.conj(v_split),
                pauli,
                v_split,
                weights,
            )
            sigma_row_sum[:, axis_idx] += contribution.real
        eigval_min = min(eigval_min, float(np.min(eigvals)))
        eigval_max = max(eigval_max, float(np.max(eigvals)))
    return sigma_row_sum, eigval_min, eigval_max, nk


def site_magnetization_streaming(
    texture: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    *,
    mu: float,
    beta: float,
    kappa_nk: int,
    twist_grid: int = 1,
    workers: int = 1,
) -> tuple[NDArray[np.float64], dict[str, float | int]]:
    """Compute ``<sigma_i>`` per site without materializing eigenvectors.

    Mathematically equivalent to::

        eigvals, eigvecs = band_spectrum_twist_averaged(...)
        sigma = site_magnetization_from_spectrum(eigvals, eigvecs, mu, beta, ...)

    but iterates over ``(twist, kappa)`` tiles, diagonalizes one row of one
    twist at a time, accumulates each tile's per-site contribution, and
    discards eigenvectors after each tile.  This bounds memory to
    ``O((2 N_c)^2)`` complex128 per worker independent of ``kappa_nk``,
    versus the dense path's ``O(twist^2 * kappa_nk^2 * (2 N_c)^2)``.

    The dense path is approximately 22 GB at ``L = 16, kappa = 72``, 622 GB
    at ``L = 37, kappa = 72``.  The streaming path is roughly 60 MB per
    worker at ``L = 37``, regardless of ``kappa_nk``.

    Parameters
    ----------
    texture
        HS exchange field, shape ``(N_c, 3)`` or ``(L1, L2, 3)``.
    params, supercell, kappa_nk, twist_grid, workers
        Forwarded to the same row-task scheme used by
        :func:`band_spectrum_twist_averaged`.
    mu, beta
        Chemical potential and inverse temperature.  ``mu`` must be
        precomputed from a separate eigvalsh-only pass (see
        :func:`self_consistent_step` for the wiring).

    Returns
    -------
    sigma : ndarray, shape ``(N_c, 3)``
        Per-site spin-density expectation, twist-and-kappa averaged.
    info : dict
        ``{"eigenvalue_min", "eigenvalue_max", "n_points"}``.  The eigenvalue
        extrema are the global min/max over all ``(twist, kappa)`` points.
        ``n_points`` is the total number of momentum points averaged over
        (`= twist_grid**2 * kappa_nk**2`).

    Notes
    -----
    The streaming pass diagonalizes the texture-dependent Hamiltonian
    a *second* time relative to a parallel eigvalsh-only pass; total compute
    is roughly 1.5x the dense path (eigh is slower than eigvalsh, but only
    one of the two passes is full-eigh).  The justification is that this
    is the only path that scales beyond ``L = 4`` cells; storing the dense
    eigenvector array for a ``L = 9`` cell already requires ~2.2 GB.
    """

    spin_field = validate_texture(texture, supercell)
    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive.")

    grids = kappa_twist_grids(int(kappa_nk), supercell, int(twist_grid))
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be >= 1.")

    n_sites = supercell.num_sites
    sigma_total = np.zeros((n_sites, 3), dtype=float)
    eigval_min_global = float("inf")
    eigval_max_global = float("-inf")
    n_points_total = 0

    if worker_count > 1 and len(grids) * int(kappa_nk) > 1:
        tasks = []
        for grid in grids:
            kx_mesh, ky_mesh = grid.cartesian_mesh()
            for row_index in range(int(grid.nk)):
                tasks.append(
                    (
                        kx_mesh[row_index, :],
                        ky_mesh[row_index, :],
                        spin_field,
                        params,
                        supercell,
                        float(mu),
                        beta_value,
                    )
                )
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(tasks))
        ) as executor:
            for sigma_row, e_min, e_max, n_pts in executor.map(
                _streaming_kappa_row_sigma, tasks
            ):
                sigma_total += sigma_row
                if e_min < eigval_min_global:
                    eigval_min_global = e_min
                if e_max > eigval_max_global:
                    eigval_max_global = e_max
                n_points_total += int(n_pts)
    else:
        for grid in grids:
            kx_mesh, ky_mesh = grid.cartesian_mesh()
            for row_index in range(int(grid.nk)):
                sigma_row, e_min, e_max, n_pts = _streaming_kappa_row_sigma(
                    (
                        kx_mesh[row_index, :],
                        ky_mesh[row_index, :],
                        spin_field,
                        params,
                        supercell,
                        float(mu),
                        beta_value,
                    )
                )
                sigma_total += sigma_row
                if e_min < eigval_min_global:
                    eigval_min_global = e_min
                if e_max > eigval_max_global:
                    eigval_max_global = e_max
                n_points_total += int(n_pts)

    if n_points_total == 0:
        raise RuntimeError(
            "site_magnetization_streaming aggregated zero points; "
            "check kappa_nk and twist_grid."
        )

    sigma_avg = sigma_total / float(n_points_total)
    info: dict[str, float | int] = {
        "eigenvalue_min": float(eigval_min_global),
        "eigenvalue_max": float(eigval_max_global),
        "n_points": int(n_points_total),
    }
    return np.asarray(sigma_avg, dtype=float), info


def _band_only_fixed_density_free_energy(
    eigenvalues: NDArray[np.float64],
    beta: float,
    filling: float,
    supercell: MagneticSupercell,
) -> tuple[float, float]:
    """Helmholtz free-energy density of the noninteracting (no HS cost) band.

    Returned as ``(F, mu)`` where ``F = mu * n + Omega_fermion`` and
    ``Omega_fermion = -(k_B T / N_c) * mean_{kappa, twist}
    sum_lambda log(1 + exp(-beta (E_lambda - mu)))``.  This branch is used
    only when ``coupling_I = 0`` so that the HS cost ``|S|^2 / (2 I)``
    is not divided by zero.
    """

    mu_value = float(
        chemical_potential_for_eigenvalues(filling, eigenvalues, beta)
    )
    n_sites = supercell.num_sites
    omega_fermion = -float(
        np.mean(
            np.sum(
                np.logaddexp(0.0, -float(beta) * (eigenvalues - mu_value)),
                axis=-1,
            )
        )
        / (float(beta) * float(n_sites))
    )
    return omega_fermion + mu_value * float(filling), mu_value


def self_consistent_step(
    S_in: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    *,
    coupling_I: float,
    kappa_nk: int,
    twist_grid: int = 1,
    fixed_filling: float,
    beta: float | None = None,
    workers: int = 1,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Apply one iteration of the saddle-point map ``T[S] = I <sigma_i>(S)``.

    The pipeline is::

        S_in
          -> H(S_in)                      build_supercell_hamiltonian
          -> (E, V) on twist x kappa      band_spectrum_twist_averaged
          -> mu such that <n>(mu) = n     chemical_potential_for_eigenvalues
          -> <sigma_i>                    site_magnetization_from_spectrum
          -> S_out = I * <sigma_i>

    Parameters
    ----------
    S_in : array-like, shape ``(N_c, 3)`` or ``(L1, L2, 3)``
        Input HS exchange field.
    params : TriangularParams
        Lattice parameters; supplies the default ``beta``.
    supercell : MagneticSupercell
        Magnetic supercell describing the cluster.
    coupling_I : float
        Effective HS coupling ``I = U / 3``.  Must be non-negative; the
        ``I = 0`` branch is supported for the ``U = 0`` benchmark.
    kappa_nk : int
        Reduced-zone grid resolution along each direction.  Total grid
        size is ``kappa_nk * kappa_nk``.
    twist_grid : int
        Twist-grid resolution.  Total twist count is ``twist_grid ** 2``.
    fixed_filling : float
        Target spinful filling per microscopic site, in ``[0, 2]``.
    beta : float, optional
        Inverse temperature.  Defaults to ``params.beta`` when ``None``.

    Returns
    -------
    S_out : ndarray, shape ``(N_c, 3)``
        New HS texture from the saddle-point map.
    info : dict
        ``{"mu", "energy", "residual", "eigenvalue_min", "eigenvalue_max"}``,
        all floats.  ``residual`` is the un-mixed
        ``||S_out - S_in||_2 / sqrt(N_c)``; ``energy`` is the fixed-density
        Helmholtz ``F[S_in]`` (or the band-only ``F`` when ``I = 0``).
    """

    coupling = float(coupling_I)
    if coupling < 0.0:
        raise ValueError("coupling_I must be non-negative.")

    beta_value = float(beta if beta is not None else params.beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive.")

    filling = float(fixed_filling)
    if filling < 0.0 or filling > 2.0:
        raise ValueError("fixed_filling must satisfy 0 <= n <= 2.")

    spin_field = validate_texture(S_in, supercell)

    if coupling == 0.0:
        # Noninteracting limit: U = 0 means the HS field has no physical
        # effect at the saddle, so T[S] = I * <sigma_i> = 0 for any seed.
        # The reported energy/mu/spectrum MUST describe the noninteracting
        # band Hamiltonian H_kin, NOT the seeded H[S_in] = H_kin - sum_i
        # S_in . sigma (which would treat S_in as a fictitious external
        # Zeeman field and shift F by O(|S_in|^2 chi0)).  Diagonalize at
        # zero_texture, skip eigenvectors entirely (they're not needed
        # since <sigma_i> at S = 0 vanishes by spin degeneracy), and
        # return S_out = 0 paired with band-only diagnostics.
        eigenvalues_zero = band_eigenvalues_twist_averaged(
            zero_texture(supercell),
            params,
            supercell,
            kappa_nk=int(kappa_nk),
            twist_grid=int(twist_grid),
            workers=int(workers),
        )
        energy, mu_value = _band_only_fixed_density_free_energy(
            eigenvalues_zero, beta_value, filling, supercell,
        )
        S_out_zero = np.zeros((supercell.num_sites, 3), dtype=float)
        residual_zero = float(
            np.linalg.norm(S_out_zero - spin_field)
            / np.sqrt(float(supercell.num_sites))
        )
        info_zero: dict[str, float] = {
            "mu": mu_value,
            "energy": energy,
            "residual": residual_zero,
            "eigenvalue_min": float(np.min(eigenvalues_zero)),
            "eigenvalue_max": float(np.max(eigenvalues_zero)),
        }
        return S_out_zero, info_zero

    # Pass 1: eigenvalues only, via the existing eigvalsh path.  Cheap on
    # memory (size kappa^2 * twist^2 * 2 N_c floats), provides the global
    # spectrum needed to solve for mu at fixed filling and to evaluate the
    # fixed-density Helmholtz F.
    eigenvalues = band_eigenvalues_twist_averaged(
        spin_field,
        params,
        supercell,
        kappa_nk=int(kappa_nk),
        twist_grid=int(twist_grid),
        workers=int(workers),
    )
    mu_value = float(
        chemical_potential_for_eigenvalues(filling, eigenvalues, beta_value)
    )
    energy = float(
        fixed_density_free_energy(
            filling, eigenvalues, spin_field, coupling, beta_value,
            easy_axis_A=float(params.easy_axis_A),
        )
    )

    # Pass 2: streaming sigma.  Diagonalizes the same Hamiltonian again
    # (this time with eigh + eigenvectors), but accumulates <sigma_i>
    # tile-by-tile and discards the eigenvector array.  Memory is
    # O((2 N_c)^2) per worker, independent of kappa_nk -- this is the
    # path that scales to L > 4 in Phase 4.1.
    sigma_per_site, _ = site_magnetization_streaming(
        spin_field,
        params,
        supercell,
        mu=mu_value,
        beta=beta_value,
        kappa_nk=int(kappa_nk),
        twist_grid=int(twist_grid),
        workers=int(workers),
    )

    S_out = coupling * sigma_per_site
    residual = float(
        np.linalg.norm(S_out - spin_field) / np.sqrt(float(supercell.num_sites))
    )

    info: dict[str, float] = {
        "mu": mu_value,
        "energy": energy,
        "residual": residual,
        "eigenvalue_min": float(np.min(eigenvalues)),
        "eigenvalue_max": float(np.max(eigenvalues)),
    }
    return np.asarray(S_out, dtype=float), info


def _pulay_next_iterate(
    history_S: list[NDArray[np.float64]],
    history_T: list[NDArray[np.float64]],
    history_r: list[NDArray[np.float64]],
    *,
    mixing_alpha: float,
) -> NDArray[np.float64]:
    """Compute the next SCF iterate via Pulay (DIIS) mixing.

    Solves the constrained least-squares problem

        min_alpha  || sum_k alpha_k r_k ||^2     subject to  sum_k alpha_k = 1

    by augmented linear system

        [ B  1 ] [alpha]   [0]
        [ 1' 0 ] [lambda] = [1]

    where ``B[i, j] = r_i . r_j`` is the residual Gram matrix.  The new
    iterate is the damped Pulay form

        S_new = sum_k alpha_k [ (1 - mixing_alpha) S_k + mixing_alpha T[S_k] ].

    With ``mixing_alpha = 1`` this is the standard Pulay / Anderson
    "Type II" update; with ``mixing_alpha < 1`` the linear damping
    reduces aggressiveness near a stiff fixed point.  The mathematics
    is the same as Anderson acceleration (Anderson 1965); the scientific
    SCF literature usually calls it Pulay (DIIS) (Pulay 1980).

    For history length 1, the Gram system degenerates to a 1x1 trivial
    case ``alpha_0 = 1``, which gives ``S_new = (1-mixing_alpha) S_0 +
    mixing_alpha T[S_0]`` -- exactly the linear-mixing update.  So the
    Pulay path can take over immediately from iteration 0, with no
    "warmup" branch.
    """

    m = len(history_S)
    if m == 0:
        raise ValueError("Pulay history must contain at least one iterate.")
    if m == 1:
        # No real history yet; equivalent to linear mixing on the
        # single iterate (alpha_0 = 1 by sum-to-one constraint).
        return (1.0 - float(mixing_alpha)) * history_S[0] + float(
            mixing_alpha
        ) * history_T[0]

    n_sites = history_S[0].shape[0]
    # Residual Gram matrix B[i, j] = r_i . r_j, treating r as a flat vector.
    flat_r = np.array(
        [r.reshape(-1) for r in history_r], dtype=float,
    )  # (m, 3 * N_c)
    gram = flat_r @ flat_r.T  # (m, m)

    # Augmented system [[B, 1], [1', 0]] [alpha; lambda] = [0; 1].
    aug = np.zeros((m + 1, m + 1), dtype=float)
    aug[:m, :m] = gram
    aug[:m, m] = 1.0
    aug[m, :m] = 1.0
    rhs = np.zeros(m + 1, dtype=float)
    rhs[m] = 1.0
    try:
        sol = np.linalg.solve(aug, rhs)
    except np.linalg.LinAlgError:
        # Singular Gram matrix: history has linearly dependent residuals
        # (e.g. on a flat plateau).  Fall back to linear mixing on the
        # most recent iterate.
        return (1.0 - float(mixing_alpha)) * history_S[-1] + float(
            mixing_alpha
        ) * history_T[-1]
    alpha_coeffs = sol[:m]

    new_S = np.zeros((n_sites, 3), dtype=float)
    for a, S_k, T_k in zip(alpha_coeffs, history_S, history_T):
        new_S += float(a) * (
            (1.0 - float(mixing_alpha)) * S_k + float(mixing_alpha) * T_k
        )
    return new_S


# ---------------------------------------------------------------------------
# Dynamic mixing schedule (Phase 6 / Hall-crystal hunt)
#
# A "mixing schedule" lets the SCF switch mixing strategy mid-run based on the
# current residual.  Motivation (from the project's retracted Route 1c
# memo): Pulay mixing rapidly drives residual down to ~1e-3 but then
# limit-cycles instead of crawling to tol=1e-5.  Linear mixing is monotone but
# slow.  Combining them -- Pulay for the bulk descent, then linear for the
# final crawl -- recovers convergence at strong-coupling SCF saddles that
# refuse to converge under either alone.
#
# A schedule is a list of (residual_threshold, mixing_name, alpha) tuples.
# An entry "applies" when the most recent residual <= threshold.  The
# most-restrictive (smallest-threshold) applicable entry is the active one at
# each iteration.  A threshold of float('inf') means "always applies"; at
# least one such entry is required so the initial residual gets matched.
# ---------------------------------------------------------------------------


_MIXING_SCHEDULE_PRESETS = {
    "pulay_to_linear": [
        (float("inf"), "pulay", 0.5),
        (1.0e-3, "linear", 0.1),
        (1.0e-4, "linear", 0.05),
    ],
}


def _parse_mixing_schedule(
    schedule,
) -> list[tuple[float, bool, float]] | None:
    """Normalize ``mixing_schedule`` to a sorted list of (threshold, use_pulay, alpha).

    Returns
    -------
    None
        If the input is ``None`` (caller falls back to static ``mixing``).
    list[tuple[float, bool, float]]
        Sorted by descending threshold so that callers iterate from
        loosest-to-tightest.

    Raises
    ------
    ValueError, TypeError
        If the schedule is malformed.
    """
    if schedule is None:
        return None

    if isinstance(schedule, str):
        key = schedule.lower()
        if key not in _MIXING_SCHEDULE_PRESETS:
            raise ValueError(
                f"Unknown mixing_schedule preset {schedule!r}; "
                f"valid presets: {sorted(_MIXING_SCHEDULE_PRESETS)}"
            )
        schedule_iterable = _MIXING_SCHEDULE_PRESETS[key]
    else:
        if not hasattr(schedule, "__iter__"):
            raise TypeError(
                "mixing_schedule must be None, a preset name (str), "
                "or an iterable of (threshold, mixing_name, alpha) tuples."
            )
        schedule_iterable = list(schedule)

    if len(schedule_iterable) == 0:
        raise ValueError("mixing_schedule must contain at least one entry.")

    parsed: list[tuple[float, bool, float]] = []
    has_infinite_threshold = False
    for entry in schedule_iterable:
        try:
            threshold_raw, mixing_name_raw, alpha_raw = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each mixing_schedule entry must be a 3-tuple "
                "(threshold, mixing_name, alpha)."
            ) from exc

        threshold = float(threshold_raw)
        if not (threshold > 0.0):
            raise ValueError(
                f"mixing_schedule threshold must be > 0; got {threshold_raw!r}."
            )
        if threshold == float("inf"):
            has_infinite_threshold = True

        mixing_name = str(mixing_name_raw).lower()
        if mixing_name not in {"linear", "pulay", "anderson"}:
            raise ValueError(
                f"mixing_schedule mixing must be 'linear', 'pulay', or "
                f"'anderson'; got {mixing_name_raw!r}."
            )
        use_pulay = mixing_name in {"pulay", "anderson"}

        alpha_val = float(alpha_raw)
        if not (0.0 < alpha_val <= 1.0):
            raise ValueError(
                "mixing_schedule alpha must satisfy 0 < alpha <= 1; "
                f"got {alpha_raw!r}."
            )

        parsed.append((threshold, use_pulay, alpha_val))

    if not has_infinite_threshold:
        raise ValueError(
            "mixing_schedule must contain at least one entry with "
            "threshold=float('inf') so the initial (large) residual is "
            "always matched."
        )

    # Sort by descending threshold so iterating from index 0 walks
    # loosest-to-tightest.  The active entry is the LAST one whose threshold
    # still applies (see _pick_active_mixing).
    parsed.sort(key=lambda item: item[0], reverse=True)
    return parsed


def _pick_active_mixing(
    parsed_schedule: list[tuple[float, bool, float]],
    current_residual: float,
    *,
    prev_index: int = 0,
) -> tuple[int, tuple[float, bool, float]]:
    """Pick the most-restrictive schedule entry applicable to ``current_residual``.

    Walks the schedule in descending-threshold order.  An entry applies if
    ``current_residual <= threshold``.  Returns the entry with the smallest
    applicable threshold (the most-restrictive entry that still applies).

    **Monotonic ratchet (``prev_index``):**
    Returned index never decreases below ``prev_index``.  This implements
    one-way transitions: once the schedule has crossed into a tighter
    regime (e.g., linear after Pulay), the search refuses to roll back to
    a looser one even if the residual transiently rises above the
    threshold.  Without this, residual oscillations around a threshold
    cause repeated transitions, each of which clears the Pulay history
    (see ``self_consistent_solve``) -- producing a pathological
    stall.  This was the root cause of NERSC job 53284590 running 1 h
    with 0 SCFs completed.

    Returns
    -------
    (chosen_index, entry)
        ``chosen_index`` is the position in ``parsed_schedule``.
        ``entry`` is the (threshold, use_pulay, alpha) tuple.
    """
    if prev_index < 0:
        raise ValueError("prev_index must be >= 0")
    if prev_index >= len(parsed_schedule):
        raise ValueError(
            f"prev_index {prev_index} out of range for schedule length "
            f"{len(parsed_schedule)}"
        )
    chosen_idx = prev_index
    # Walk only forward (toward tighter entries).  Never roll back.
    for i in range(prev_index + 1, len(parsed_schedule)):
        if current_residual <= parsed_schedule[i][0]:
            chosen_idx = i
        else:
            break
    return chosen_idx, parsed_schedule[chosen_idx]


def self_consistent_solve(
    S_seed: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    *,
    coupling_I: float,
    kappa_nk: int,
    twist_grid: int = 1,
    fixed_filling: float,
    beta: float | None = None,
    mixing: str = "linear",
    mixing_alpha: float = 0.3,
    mixing_history: int = 5,
    mixing_schedule=None,
    tol: float = 1.0e-5,
    max_iter: int = 500,
    verbose: bool = False,
    workers: int = 1,
) -> dict:
    """Iterate the SCF map until residual ``< tol``.

    Three static mixing schemes are supported:

    - ``mixing = "linear"`` (Picard):
      ``S^{(k+1)} = (1 - alpha) S^{(k)} + alpha T[S^{(k)}]``.
      Robust but slow; default ``alpha = 0.3``.
    - ``mixing = "pulay"`` (DIIS, Pulay 1980):
      Maintain history of last ``mixing_history`` iterates and residuals;
      solve constrained least-squares for the optimal linear combination
      and damp by ``mixing_alpha``.  Default ``mixing_alpha = 0.3`` is
      the same value used for linear mixing; for Pulay the ``alpha = 1``
      "pure" form is also common and tends to converge faster but is
      less stable near critical points.
    - ``mixing = "anderson"``:
      Alias for ``"pulay"`` -- algebraically the same algorithm,
      different name conventions in different literatures (Anderson 1965).

    Additionally, a *dynamic* mixing schedule may be supplied via
    ``mixing_schedule``, which adapts the mixing strategy mid-run based on
    the current residual.  See :func:`_parse_mixing_schedule` and the
    ``_MIXING_SCHEDULE_PRESETS`` dictionary for the format.  When
    ``mixing_schedule`` is provided, the ``mixing`` and ``mixing_alpha``
    arguments are IGNORED.

    Convergence is declared when ``||T[S^{(k)}] - S^{(k)}||_2 / sqrt(N_c)
    < tol``; the converged texture returned is the un-mixed ``T[S^{(k)}]``.

    Parameters
    ----------
    S_seed : array-like
        Starting HS texture.
    params, supercell, coupling_I, kappa_nk, twist_grid, fixed_filling, beta
        Forwarded to :func:`self_consistent_step`.
    mixing : {"linear", "pulay", "anderson"}
        Static mixing scheme.  ``"anderson"`` is an alias for ``"pulay"``.
        Ignored if ``mixing_schedule`` is supplied.
    mixing_alpha : float
        Damping weight on the proposed update.  Must satisfy
        ``0 < mixing_alpha <= 1``.  Ignored if ``mixing_schedule`` is
        supplied (the schedule provides its own per-entry alpha).
    mixing_history : int
        Number of past iterates retained in the Pulay history.  Used by
        both static and dynamic Pulay branches; ignored entirely if the
        run never uses Pulay.
    mixing_schedule : None, str, or list of (threshold, mixing_name, alpha)
        If provided, the SCF picks an active (mixing_name, alpha) per
        iteration based on the most-recent residual.  String presets
        (e.g. ``"pulay_to_linear"``) expand to known schedules.  The
        Pulay history is reset on every mixing switch so a fresh
        accelerator doesn't inherit stale residual directions.
    tol, max_iter, verbose, workers
        Same as previously.

    Returns
    -------
    result : dict
        Keys: ``S_converged`` (``(N_c, 3)``), ``mu``, ``energy``,
        ``n_iter``, ``residual_history``, ``energy_history``,
        ``mu_history``, ``converged`` (bool), ``status``
        (``"ok"`` | ``"max_iter"``), ``mixing_switches`` (list of
        ``(iteration, threshold, mixing_name, alpha)`` tuples recording
        every active-mixing change; empty when no schedule is supplied).
    """

    parsed_schedule = _parse_mixing_schedule(mixing_schedule)

    if parsed_schedule is None:
        mixing_normalized = str(mixing).lower()
        if mixing_normalized not in {"linear", "pulay", "anderson"}:
            raise NotImplementedError(
                f"mixing={mixing!r} not supported; "
                f"expected one of 'linear', 'pulay', 'anderson'."
            )
        use_pulay_static = mixing_normalized in {"pulay", "anderson"}

        alpha_static = float(mixing_alpha)
        if not (0.0 < alpha_static <= 1.0):
            raise ValueError("mixing_alpha must satisfy 0 < alpha <= 1.")
    else:
        # Schedule supplied; per-entry mixing/alpha overrides static args.
        use_pulay_static = False  # unused; placeholder
        alpha_static = 0.0  # unused; placeholder

    history_size = int(mixing_history)
    if history_size < 1:
        raise ValueError("mixing_history must be >= 1.")

    tol_value = float(tol)
    if tol_value <= 0.0:
        raise ValueError("tol must be positive.")

    max_iter_value = int(max_iter)
    if max_iter_value < 1:
        raise ValueError("max_iter must be >= 1.")

    S_current = validate_texture(S_seed, supercell).copy()
    residual_history: list[float] = []
    energy_history: list[float] = []
    mu_history: list[float] = []
    mixing_switches: list[tuple[int, float, str, float]] = []
    converged = False
    last_info: dict[str, float] = {}

    # Pulay-only state (unused in the linear path).
    history_S: list[NDArray[np.float64]] = []
    history_T: list[NDArray[np.float64]] = []
    history_r: list[NDArray[np.float64]] = []

    # Track the previous iteration's active mixing so we can detect
    # schedule transitions and reset the Pulay history accordingly.
    # ``prev_active_index`` (None until first iteration) implements the
    # monotonic ratchet: once we transition to a tighter entry, the
    # picker never rolls back to a looser one (see _pick_active_mixing).
    prev_active: tuple[bool, float] | None = None
    prev_active_index: int = 0

    for iteration in range(max_iter_value):
        S_proposed, info = self_consistent_step(
            S_current,
            params,
            supercell,
            coupling_I=coupling_I,
            kappa_nk=kappa_nk,
            twist_grid=twist_grid,
            fixed_filling=fixed_filling,
            beta=beta,
            workers=workers,
        )
        last_info = info
        residual = float(info["residual"])
        residual_history.append(residual)
        energy_history.append(float(info["energy"]))
        mu_history.append(float(info["mu"]))

        if verbose:
            print(
                f"[scf] iter {iteration:4d}  "
                f"residual={residual:.3e}  "
                f"F={info['energy']:.6e}  "
                f"mu={info['mu']:.6e}"
            )

        if residual < tol_value:
            converged = True
            # Recompute mu/energy/eigenvalue diagnostics for S_proposed so
            # that the returned (S_converged, mu, energy) all describe the
            # same texture.  Without this, last_info would still hold
            # F[S_current_pre_update] and mu(S_current_pre_update), which
            # differ from F[S_proposed] and mu(S_proposed) by O(residual)
            # at fixed density and by O(residual^2) in the energy itself
            # (F is stationary at the saddle, but the API would lie about
            # which texture the diagnostics correspond to).
            _, last_info = self_consistent_step(
                S_proposed,
                params,
                supercell,
                coupling_I=coupling_I,
                kappa_nk=kappa_nk,
                twist_grid=twist_grid,
                fixed_filling=fixed_filling,
                beta=beta,
                workers=workers,
            )
            S_current = S_proposed
            break

        # Resolve active mixing for THIS iteration's update step.  If a
        # schedule is supplied, the most-recent residual selects an entry
        # subject to the monotonic ratchet (never go back to a looser
        # entry); otherwise the static mixing arguments apply.
        if parsed_schedule is not None:
            prev_active_index, (
                active_threshold, use_pulay_iter, alpha_iter,
            ) = _pick_active_mixing(
                parsed_schedule, residual,
                prev_index=prev_active_index,
            )
        else:
            active_threshold = float("inf")  # placeholder for logging
            use_pulay_iter = use_pulay_static
            alpha_iter = alpha_static

        # Detect schedule transition.  Reset Pulay history on any switch
        # so the new mixing strategy starts fresh rather than inheriting
        # stale residual directions from the prior strategy.
        active_key = (use_pulay_iter, alpha_iter)
        if prev_active is not None and active_key != prev_active:
            history_S.clear()
            history_T.clear()
            history_r.clear()
            mixing_name = "pulay" if use_pulay_iter else "linear"
            mixing_switches.append(
                (iteration, active_threshold, mixing_name, alpha_iter)
            )
            if verbose:
                print(
                    f"[scf] iter {iteration:4d}  "
                    f"mixing switch -> {mixing_name} alpha={alpha_iter:.3g} "
                    f"(threshold={active_threshold:g}, residual={residual:.3e})"
                )
        elif prev_active is None and parsed_schedule is not None:
            # Record the initial schedule entry too (iteration 0).
            mixing_name = "pulay" if use_pulay_iter else "linear"
            mixing_switches.append(
                (iteration, active_threshold, mixing_name, alpha_iter)
            )
        prev_active = active_key

        if use_pulay_iter:
            # Append to bounded-length histories (drop oldest if full).
            history_S.append(S_current.copy())
            history_T.append(S_proposed.copy())
            history_r.append(S_proposed - S_current)
            if len(history_S) > history_size:
                history_S.pop(0)
                history_T.pop(0)
                history_r.pop(0)
            S_current = _pulay_next_iterate(
                history_S, history_T, history_r,
                mixing_alpha=alpha_iter,
            )
        else:
            S_current = (1.0 - alpha_iter) * S_current + alpha_iter * S_proposed

    return {
        "S_converged": np.asarray(S_current, dtype=float),
        "mu": float(last_info.get("mu", float("nan"))),
        "energy": float(last_info.get("energy", float("nan"))),
        "n_iter": len(residual_history),
        "residual_history": residual_history,
        "energy_history": energy_history,
        "mu_history": mu_history,
        "mixing_switches": mixing_switches,
        "converged": converged,
        "status": "ok" if converged else "max_iter",
    }


# ---------------------------------------------------------------------------
# Seed library (Phase 4.1)
#
# ---------------------------------------------------------------------------


def random_seed_perturbation(
    supercell: MagneticSupercell,
    *,
    amplitude: float = 0.01,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Return a small isotropic random perturbation around ``S = 0``.

    Parameters
    ----------
    supercell
        Magnetic supercell defining ``N_c``.
    amplitude
        Standard deviation of each Cartesian component.  Default
        ``0.01`` matches the warm-start library convention of random
        seeds at amplitude 0.01.
    rng
        Optional ``numpy.random.Generator`` for reproducibility.  A new
        default-seeded generator is created when ``None`` is passed.

    Returns
    -------
    ndarray of shape ``(N_c, 3)``, real, drawn from
    ``N(0, amplitude^2)`` per component (so each site's vector has
    expected magnitude `~ amplitude * sqrt(3)`).

    Notes
    -----
    This routine deliberately does not project to the unit sphere
    (no `texture / |texture|` normalization).  Phase 4 SCF starts
    from a *small* perturbation of the paramagnetic fixed point and
    relaxes to a finite-amplitude saddle if one exists.  Imposing a
    fixed amplitude in the seed would prejudice the basin-of-attraction
    diagnostics.
    """

    amp = float(amplitude)
    if amp < 0.0:
        raise ValueError("amplitude must be non-negative.")
    generator = rng if rng is not None else np.random.default_rng()
    return amp * generator.standard_normal((supercell.num_sites, 3))






# ---------------------------------------------------------------------------
# Q-spectrum diagnostic (Phase 4.1)
#
# extract_dominant_q answers the open Phase 3b question on converged
# unrestricted textures: at which Q does the SCF actually lock in?
# ---------------------------------------------------------------------------


def _gini_index(values: NDArray[np.float64]) -> float:
    """Standard Gini index of a non-negative distribution.

    For values ``x_i >= 0`` (i = 1, ..., n) sorted ascending,

        G = (2 * sum_i i * x_i) / (n * sum_i x_i) - (n + 1) / n.

    G = 0 for a perfectly uniform distribution; G -> 1 - 1/n for a
    distribution concentrated at a single bin.  Used here as a measure
    of how localized the |S(q)|^2 spectrum is in q -- a single-Q SDW
    has G near 1, a paramagnet (uniform power) has G near 0.

    Returns 0.0 for an all-zero distribution.
    """

    flat = np.sort(values.ravel().astype(float))
    n = int(flat.size)
    if n == 0:
        return 0.0
    total = float(np.sum(flat))
    if total <= 0.0:
        return 0.0
    indices = np.arange(1, n + 1, dtype=float)
    return float(
        2.0 * np.sum(indices * flat) / (n * total) - (n + 1.0) / n
    )


def extract_dominant_q(
    S: ArrayLike,
    supercell: MagneticSupercell,
    *,
    min_rms_amplitude: float = 1.0e-4,
) -> dict[str, object]:
    """FFT ``|S(q)|^2`` over the supercell BZ; report dominant Q + diagnostics.

    Computes the discrete Fourier transform of the texture across the
    supercell, sums the per-component power, and returns the dominant
    wavevector together with localization diagnostics.  This is the
    function that *answers* Phase 3b's open Q-locking question on
    converged unrestricted textures: which Q does the SCF actually lock
    onto?

    A near-zero texture (rms ``|S_i|`` below ``min_rms_amplitude``) is
    classified as ``moment_status = "no_moment"``.  In that case the
    finite-Q diagnostics are unreliable -- numerical-noise FFT bins
    can produce a high-Gini "spike" that has no physical meaning -- so
    ``dominant_finite_q_frac`` is set to ``None`` and
    ``finite_peak_height`` to 0.  This guards production interpretation
    against false Q-locking claims for paramagnetic / decayed-seed SCF
    outputs.

    Parameters
    ----------
    S : array-like, shape ``(N_c, 3)`` or ``(L1, L2, 3)``
        Real exchange-field texture.
    supercell : MagneticSupercell
        Defines ``L1``, ``L2``, and the site-index ordering.
    min_rms_amplitude : float, optional
        Threshold below which the texture is treated as paramagnetic
        and finite-Q diagnostics are nulled.  Default ``1e-4`` is
        approximately ``100x`` the SCF residual tolerance (``1e-5``)
        and ``1000x`` smaller than physical-saddle moments
        (``M > 0.05`` typical), so it cleanly separates real saddles
        from noise.  Pass ``0.0`` to disable (legacy behaviour:
        always emit a finite-Q peak).

    Returns
    -------
    dict with keys:

    ``power_spectrum`` : ndarray, shape ``(L1, L2)``
        ``|S(q)|^2 = sum_alpha |S_alpha(q)|^2`` summed over Cartesian
        components.  No FFT shift is applied;
        ``power_spectrum[0, 0]`` is the q = 0 (uniform-FM) component.
    ``rms_amplitude`` : float
        ``sqrt(<|S_i|^2>)`` averaged over sites.  The threshold
        comparison used for ``moment_status``.
    ``moment_status`` : ``"ok"`` or ``"no_moment"``
        ``"ok"`` if ``rms_amplitude >= min_rms_amplitude``; else
        ``"no_moment"`` with finite-Q fields nulled.
    ``dominant_q_frac`` : tuple ``(q_1, q_2)``
        Fractional reciprocal-lattice coordinates of the global maximum
        of ``|S(q)|^2``.  Reported regardless of moment_status (in the
        true-zero limit it is ``(0, 0)``).
    ``dominant_q_index`` : tuple ``(k_1, k_2)``
        Integer grid index of the global maximum.
    ``peak_height`` : float
        Value of ``|S(q*)|^2`` at the dominant q.
    ``dominant_finite_q_frac`` : tuple or ``None``
        As ``dominant_q_frac`` but excluding ``q = 0``.  ``None`` when
        ``moment_status == "no_moment"`` or when all finite-Q
        components are exactly zero.  This is the relevant entry for
        the "where does the unrestricted Q lock?" question -- callers
        consuming this for production ranking MUST check
        ``moment_status`` first.
    ``dominant_finite_q_index`` : tuple or ``None``
        Integer grid index of the dominant finite Q (or ``None``).
    ``finite_peak_height`` : float
        Value of ``|S(q*)|^2`` at the dominant finite Q.  Set to 0 in
        the no-moment regime.
    ``q0_height`` : float
        ``|S(q = 0)|^2``, the uniform-FM weight.
    ``parseval_real``, ``parseval_q_over_Nc`` : float
        Real-space and Fourier-space ``sum |S|^2``; the ratio is the
        Parseval sanity check (should equal 1 up to numerical noise).
    ``gini`` : float in ``[0, 1)``
        Gini index of the power spectrum distribution.  Near 0 for a
        uniform spectrum (paramagnetic / random texture), near
        ``1 - 1/(L1 L2)`` for a single-Q-spike SDW.  Note that for
        a no-moment texture the Gini is computed but is dominated by
        numerical noise; rely on ``moment_status`` to gate
        interpretation.
    """

    arr = validate_texture(S, supercell)  # (N_c, 3) row-major site_index
    rms_amplitude = float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))))
    threshold = float(min_rms_amplitude)
    if threshold < 0.0:
        raise ValueError("min_rms_amplitude must be non-negative.")
    moment_status = "ok" if rms_amplitude >= threshold else "no_moment"

    arr_2d = arr.reshape(int(supercell.L1), int(supercell.L2), 3)
    fft = np.fft.fftn(arr_2d, axes=(0, 1))  # (L1, L2, 3) complex
    power = np.sum(np.abs(fft) ** 2, axis=2).astype(float)  # (L1, L2)

    # Global maximum (always reported -- in the strict-zero limit, this
    # is (0, 0) by argmax tie-breaking).
    flat_idx = int(np.argmax(power))
    k1, k2 = np.unravel_index(flat_idx, power.shape)
    dominant_q_frac = (
        float(k1) / float(supercell.L1),
        float(k2) / float(supercell.L2),
    )
    peak_height = float(power[k1, k2])

    # Dominant finite Q (excluding q = 0). Nulled in the no-moment
    # regime to prevent false Q-locking claims from numerical-noise
    # FFT bins.
    if moment_status == "ok":
        finite_power = power.copy()
        finite_power[0, 0] = 0.0
        if float(np.max(finite_power)) > 0.0:
            flat_idx_finite = int(np.argmax(finite_power))
            kf1, kf2 = np.unravel_index(flat_idx_finite, finite_power.shape)
            dominant_finite_q_frac: tuple[float, float] | None = (
                float(kf1) / float(supercell.L1),
                float(kf2) / float(supercell.L2),
            )
            dominant_finite_q_index: tuple[int, int] | None = (int(kf1), int(kf2))
            finite_peak_height = float(finite_power[kf1, kf2])
        else:
            dominant_finite_q_frac = None
            dominant_finite_q_index = None
            finite_peak_height = 0.0
    else:
        dominant_finite_q_frac = None
        dominant_finite_q_index = None
        finite_peak_height = 0.0

    q0_height = float(power[0, 0])

    # Parseval: sum_q |S(q)|^2 = N_c * sum_i |S_i|^2 (FFT is unnormalized,
    # so the factor is N_c rather than 1).
    sum_real = float(np.sum(arr * arr))
    sum_q = float(np.sum(power))
    parseval_check = sum_q / max(float(supercell.num_sites), 1.0)

    return {
        "power_spectrum": power,
        "rms_amplitude": rms_amplitude,
        "moment_status": moment_status,
        "min_rms_amplitude_threshold": threshold,
        "dominant_q_frac": dominant_q_frac,
        "dominant_q_index": (int(k1), int(k2)),
        "peak_height": peak_height,
        "dominant_finite_q_frac": dominant_finite_q_frac,
        "dominant_finite_q_index": dominant_finite_q_index,
        "finite_peak_height": finite_peak_height,
        "q0_height": q0_height,
        "parseval_real": sum_real,
        "parseval_q_over_Nc": parseval_check,
        "gini": _gini_index(power),
    }
