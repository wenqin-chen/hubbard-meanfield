"""Triangular-lattice Hubbard band and nesting diagnostics.

The routines in this module implement Phase 1 of the Hubbard-model project:
noninteracting triangular-lattice band structure, filling control, Fermi-surface
diagnostics, and the static Lindhard response.

Conventions
-----------
The triangular real-space primitive vectors are

    a1 = (1, 0),  a2 = (1/2, sqrt(3)/2),

so the reciprocal primitive vectors are

    b1 = (2*pi, -2*pi/sqrt(3)),  b2 = (0, 4*pi/sqrt(3)).

Filling is spinful electrons per site, ``0 <= n <= 2``. The susceptibility
returned by :func:`lindhard_chi0` is the positive static particle-hole response
per spin species:

    chi0(q) = - <[f(xi_k) - f(xi_{k+q})] / [xi_k - xi_{k+q}]>_k.

With this sign convention, larger positive values mean stronger finite-Q
magnetic tendency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


PI = float(np.pi)
SQRT3 = float(np.sqrt(3.0))

RECIPROCAL_B1 = np.array([2.0 * PI, -2.0 * PI / SQRT3], dtype=float)
RECIPROCAL_B2 = np.array([0.0, 4.0 * PI / SQRT3], dtype=float)


@dataclass(frozen=True)
class TriangularParams:
    """Hopping and temperature parameters for the triangular band.

    ``h_z`` and ``alpha_rashba`` extend the kinetic parameters for the
    skyrmion-SOC extensions.
    ``easy_axis_A``
    is a mean-field-level uniaxial single-ion anisotropy: at the
    classical-mean-field manifold it contributes an extra free-energy
    term ``ΔF_anis = -A * Σ_i (S_i^z_HS)²`` (with the HS-field
    convention used throughout this codebase, where the texture's
    z-component IS the HS exchange field's z-component, with the
    coupling absorbed). The band Hamiltonian gains a *texture-
    proportional* site-dependent z-Zeeman ``-A * S_i^z_HS * σ_z`` per
    site, completely distinct from the uniform-Zeeman ``h_z`` term.
    All three defaults to ``0.0``, which reproduces the spin-degenerate
    Hubbard Hamiltonian bit-for-bit; the consumers
    (:func:`hubbard_meanfield.build_supercell_hamiltonian`,
    :func:`hubbard_meanfield.grand_potential`) gate the
    new contributions on these fields being nonzero.
    """

    t: float = 1.0
    t2: float = 0.0
    t3: float = 0.0
    beta: float = 200.0
    h_z: float = 0.0
    alpha_rashba: float = 0.0
    easy_axis_A: float = 0.0

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "t": float(self.t),
            "t2": float(self.t2),
            "t3": float(self.t3),
            "beta": float(self.beta),
            "h_z": float(self.h_z),
            "alpha_rashba": float(self.alpha_rashba),
            "easy_axis_A": float(self.easy_axis_A),
        }


@dataclass(frozen=True)
class BZGrid:
    """Uniform reciprocal-coordinate grid on one primitive BZ parallelogram."""

    nk: int
    shift_u: float = 0.0
    shift_v: float = 0.0

    def __post_init__(self) -> None:
        if int(self.nk) < 2:
            raise ValueError("nk must be >= 2.")
        if not np.isfinite(float(self.shift_u)) or not np.isfinite(float(self.shift_v)):
            raise ValueError("grid shifts must be finite.")

    @property
    def frac_1d(self) -> NDArray[np.float64]:
        """Return centered fractional u-coordinates in [-1/2, 1/2)."""

        return self.frac_u_1d

    @property
    def frac_u_1d(self) -> NDArray[np.float64]:
        """Return centered fractional u-coordinates, optionally shifted."""

        return (np.arange(self.nk, dtype=float) - self.nk // 2 + float(self.shift_u)) / float(
            self.nk
        )

    @property
    def frac_v_1d(self) -> NDArray[np.float64]:
        """Return centered fractional v-coordinates, optionally shifted."""

        return (np.arange(self.nk, dtype=float) - self.nk // 2 + float(self.shift_v)) / float(
            self.nk
        )

    @property
    def reciprocal_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return RECIPROCAL_B1.copy(), RECIPROCAL_B2.copy()

    def fractional_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.frac_u_1d, self.frac_v_1d, indexing="ij")

    def cartesian_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        u, v = self.fractional_mesh()
        kx, ky = fractional_to_cartesian(u, v)
        return kx, ky

    def minimum_step_norm(self) -> float:
        step_vectors = (
            RECIPROCAL_B1 / float(self.nk),
            RECIPROCAL_B2 / float(self.nk),
            (RECIPROCAL_B1 - RECIPROCAL_B2) / float(self.nk),
            (RECIPROCAL_B1 + RECIPROCAL_B2) / float(self.nk),
        )
        return float(min(np.linalg.norm(vec) for vec in step_vectors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nk": int(self.nk),
            "frac_1d": self.frac_1d.tolist(),
            "shift_u": float(self.shift_u),
            "shift_v": float(self.shift_v),
            "b1": RECIPROCAL_B1.tolist(),
            "b2": RECIPROCAL_B2.tolist(),
        }


def twist_offsets(twist_grid: int = 1) -> list[tuple[float, float]]:
    """Return symmetric twist offsets inside one reciprocal grid cell.

    The offsets are measured in units of a single k-grid spacing. ``twist_grid=2``
    therefore averages over ``(0, 0)``, ``(0, 1/2)``, ``(1/2, 0)``, and
    ``(1/2, 1/2)``. The full ``m x m`` set is closed under triangular-lattice C6
    rotations modulo one grid cell.
    """

    side = int(twist_grid)
    if side < 1:
        raise ValueError("twist_grid must be >= 1.")
    return [(i / float(side), j / float(side)) for i in range(side) for j in range(side)]


def fractional_to_cartesian(
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert reciprocal fractional coordinates to Cartesian momenta."""

    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    kx = u_arr * RECIPROCAL_B1[0] + v_arr * RECIPROCAL_B2[0]
    ky = u_arr * RECIPROCAL_B1[1] + v_arr * RECIPROCAL_B2[1]
    return np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)


def cartesian_to_fractional(
    kx: ArrayLike,
    ky: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian momenta to reciprocal fractional coordinates."""

    matrix = np.column_stack((RECIPROCAL_B1, RECIPROCAL_B2))
    inv = np.linalg.inv(matrix)
    stacked = np.stack((np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)), axis=0)
    frac = np.tensordot(inv, stacked, axes=(1, 0))
    return np.asarray(frac[0], dtype=float), np.asarray(frac[1], dtype=float)


def wrap_fractional(
    u: ArrayLike,
    v: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrap reciprocal fractional coordinates into [-1/2, 1/2)."""

    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    return u_arr - np.floor(u_arr + 0.5), v_arr - np.floor(v_arr + 0.5)


def periodic_distance_fractional(
    frac_a: tuple[float, float],
    frac_b: tuple[float, float],
) -> float:
    """Return the shortest reciprocal-space distance between two fractional points."""

    du = float(frac_a[0]) - float(frac_b[0])
    dv = float(frac_a[1]) - float(frac_b[1])
    du -= round(du)
    dv -= round(dv)
    dx, dy = fractional_to_cartesian(du, dv)
    return float(np.hypot(dx, dy))


def triangular_dispersion(
    kx: ArrayLike,
    ky: ArrayLike,
    params: TriangularParams = TriangularParams(),
) -> NDArray[np.float64]:
    """Return the triangular-lattice dispersion from the project notes."""

    kx_arr = np.asarray(kx, dtype=float)
    ky_arr = np.asarray(ky, dtype=float)
    eps = (
        -2.0
        * float(params.t)
        * (
            np.cos(kx_arr)
            + 2.0 * np.cos(0.5 * kx_arr) * np.cos(0.5 * SQRT3 * ky_arr)
        )
    )
    eps += (
        -2.0
        * float(params.t2)
        * (
            np.cos(SQRT3 * ky_arr)
            + 2.0 * np.cos(1.5 * kx_arr) * np.cos(0.5 * SQRT3 * ky_arr)
        )
    )
    eps += (
        -2.0
        * float(params.t3)
        * (
            np.cos(2.0 * kx_arr)
            + 2.0 * np.cos(kx_arr) * np.cos(SQRT3 * ky_arr)
        )
    )
    return np.asarray(eps, dtype=float)


def high_symmetry_band_path(
    params: TriangularParams = TriangularParams(),
    *,
    points_per_segment: int = 120,
) -> dict[str, Any]:
    """Return band energies along the triangular ``Gamma-K-M-Gamma`` path."""

    if points_per_segment < 2:
        raise ValueError("points_per_segment must be >= 2.")

    nodes = [
        ("Gamma", (0.0, 0.0)),
        ("K", (2.0 / 3.0, 1.0 / 3.0)),
        ("M", (0.5, 0.5)),
        ("Gamma", (0.0, 0.0)),
    ]

    frac_u_parts: list[NDArray[np.float64]] = []
    frac_v_parts: list[NDArray[np.float64]] = []
    tick_positions = [0.0]
    cumulative = 0.0

    for segment_index, ((_, start), (_, stop)) in enumerate(zip(nodes[:-1], nodes[1:])):
        weights = np.linspace(0.0, 1.0, points_per_segment)
        if segment_index > 0:
            weights = weights[1:]
        start_arr = np.asarray(start, dtype=float)
        stop_arr = np.asarray(stop, dtype=float)
        segment = start_arr[None, :] + weights[:, None] * (stop_arr - start_arr)[None, :]
        frac_u_parts.append(segment[:, 0])
        frac_v_parts.append(segment[:, 1])

        start_kx, start_ky = fractional_to_cartesian(start[0], start[1])
        stop_kx, stop_ky = fractional_to_cartesian(stop[0], stop[1])
        segment_length = float(np.hypot(float(stop_kx - start_kx), float(stop_ky - start_ky)))
        cumulative += segment_length
        tick_positions.append(cumulative)

    frac_u = np.concatenate(frac_u_parts)
    frac_v = np.concatenate(frac_v_parts)
    kx, ky = fractional_to_cartesian(frac_u, frac_v)
    step_lengths = np.hypot(np.diff(kx), np.diff(ky))
    distance = np.concatenate(([0.0], np.cumsum(step_lengths)))
    energies = triangular_dispersion(kx, ky, params)

    return {
        "distance": distance,
        "energy": energies,
        "frac_u": frac_u,
        "frac_v": frac_v,
        "kx": kx,
        "ky": ky,
        "tick_positions": np.asarray(tick_positions, dtype=float),
        "tick_labels": [label for label, _ in nodes],
        "nodes_frac": [list(frac) for _, frac in nodes],
    }


def fermi_function(xi: ArrayLike, beta: float) -> NDArray[np.float64]:
    """Return a numerically stable Fermi function ``1 / (exp(beta xi) + 1)``."""

    x = np.clip(float(beta) * np.asarray(xi, dtype=float), -700.0, 700.0)
    return np.asarray(1.0 / (np.exp(x) + 1.0), dtype=float)


def fermi_derivative(xi: ArrayLike, beta: float) -> NDArray[np.float64]:
    """Return ``d n_F(xi) / d xi``."""

    f = fermi_function(xi, beta)
    return np.asarray(-float(beta) * f * (1.0 - f), dtype=float)


def band_energies(
    params: TriangularParams,
    grid: BZGrid,
) -> NDArray[np.float64]:
    kx, ky = grid.cartesian_mesh()
    return triangular_dispersion(kx, ky, params)


def filling_from_mu(
    mu: float,
    params: TriangularParams,
    grid: BZGrid,
    energies: NDArray[np.float64] | None = None,
) -> float:
    """Return spinful filling for a chemical potential."""

    eps = band_energies(params, grid) if energies is None else np.asarray(energies, dtype=float)
    return float(2.0 * np.mean(fermi_function(eps - float(mu), params.beta)))


def chemical_potential_for_filling(
    filling: float,
    params: TriangularParams,
    grid: BZGrid,
    *,
    tol: float = 1.0e-11,
    maxiter: int = 160,
) -> float:
    """Solve for the chemical potential that gives the requested spinful filling."""

    target = float(filling)
    if not 0.0 <= target <= 2.0:
        raise ValueError("filling must satisfy 0 <= filling <= 2.")

    eps = band_energies(params, grid)
    emin = float(np.min(eps))
    emax = float(np.max(eps))
    width = max(emax - emin, 1.0)
    margin = max(width + 1.0, 50.0 / float(params.beta))
    lo = emin - margin
    hi = emax + margin

    if target == 0.0:
        return float(lo)
    if target == 2.0:
        return float(hi)

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        value = filling_from_mu(mid, params, grid, eps)
        if abs(value - target) < tol:
            return float(mid)
        if value < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def lindhard_chi0(
    qx: float,
    qy: float,
    mu: float,
    params: TriangularParams,
    grid: BZGrid,
) -> float:
    """Return the positive static Lindhard response at one wavevector."""

    kx, ky = grid.cartesian_mesh()
    eps_k = band_energies(params, grid)
    f_k = fermi_function(eps_k - float(mu), params.beta)
    return lindhard_chi0_arrays(qx, qy, mu, params, kx, ky, eps_k, f_k)


def susceptibility_b1_cut(
    mu: float,
    params: TriangularParams,
    grid: BZGrid,
    *,
    num_points: int = 801,
    s_min: float = -0.5,
    s_max: float = 0.5,
    twist_grid: int = 1,
) -> dict[str, Any]:
    """Compute a dense direct Lindhard cut along ``q = s b1``.

    This evaluates the same kernel as :func:`lindhard_chi0` directly for each
    point on the line. It intentionally does not sample from a precomputed
    heatmap, because the heatmap grid is too coarse for resolving the weak
    ``2 k_F`` cusp. If ``twist_grid > 1``, raw ``chi0`` values are averaged
    over shifted k-grids before normalization.
    """

    if num_points < 2:
        raise ValueError("num_points must be >= 2.")
    if s_max <= s_min:
        raise ValueError("s_max must be greater than s_min.")

    s_values = np.linspace(float(s_min), float(s_max), int(num_points))
    chi_values = np.zeros_like(s_values, dtype=float)
    offsets = twist_offsets(twist_grid)
    for shift_u, shift_v in offsets:
        shifted_grid = BZGrid(grid.nk, shift_u=shift_u, shift_v=shift_v)
        kx, ky = shifted_grid.cartesian_mesh()
        eps_k = band_energies(params, shifted_grid)
        f_k = fermi_function(eps_k - float(mu), params.beta)

        for index, s_value in enumerate(s_values):
            qx = float(s_value * RECIPROCAL_B1[0])
            qy = float(s_value * RECIPROCAL_B1[1])
            chi_values[index] += lindhard_chi0_arrays(
                qx,
                qy,
                mu,
                params,
                kx,
                ky,
                eps_k,
                f_k,
            )
    chi_values /= float(len(offsets))

    gamma_index = int(np.argmin(np.abs(s_values)))
    chi_gamma = float(chi_values[gamma_index])
    if abs(chi_gamma) > 0.0:
        chi_over_gamma = chi_values / chi_gamma
        delta_over_gamma = (chi_values - chi_gamma) / chi_gamma
    else:
        chi_over_gamma = np.full_like(chi_values, np.nan)
        delta_over_gamma = np.full_like(chi_values, np.nan)

    b1_norm = float(np.linalg.norm(RECIPROCAL_B1))
    return {
        "direction": "b1",
        "grid_nk": int(grid.nk),
        "twist_grid": int(twist_grid),
        "twist_count": int(len(offsets)),
        "num_points": int(num_points),
        "s_min": float(s_min),
        "s_max": float(s_max),
        "s": s_values,
        "q_parallel": s_values * b1_norm,
        "chi0": chi_values,
        "chi0_gamma": chi_gamma,
        "chi0_over_gamma": chi_over_gamma,
        "delta_over_gamma": delta_over_gamma,
    }


def circular_kf_from_filling(filling: float) -> float:
    """Return the circular-pocket ``k_F`` implied by spinful filling."""

    filling = float(filling)
    if filling < 0.0:
        raise ValueError("filling must be non-negative.")
    bz_area = abs(
        float(
            RECIPROCAL_B1[0] * RECIPROCAL_B2[1]
            - RECIPROCAL_B1[1] * RECIPROCAL_B2[0]
        )
    )
    return float(np.sqrt(filling * bz_area / (2.0 * PI)))


def lindhard_chi0_arrays(
    qx: float,
    qy: float,
    mu: float,
    params: TriangularParams,
    kx: NDArray[np.float64],
    ky: NDArray[np.float64],
    eps_k: NDArray[np.float64],
    f_k: NDArray[np.float64],
) -> float:
    """Return ``chi0(q)`` using precomputed k-grid arrays.

    This is the same Lindhard kernel used by :func:`lindhard_chi0`, but avoids
    rebuilding ``kx``, ``ky``, ``eps_k``, and ``f_k`` for dense cuts or angular
    sweeps.
    """

    eps_kq = triangular_dispersion(kx + float(qx), ky + float(qy), params)
    xi_k = eps_k - float(mu)
    xi_kq = eps_kq - float(mu)
    f_kq = fermi_function(xi_kq, params.beta)

    denom = xi_k - xi_kq
    scale = max(1.0, float(np.max(np.abs(eps_k))), float(abs(mu)))
    degenerate = np.abs(denom) < 1.0e-10 * scale

    term = np.empty_like(denom, dtype=float)
    term[~degenerate] = -(f_k[~degenerate] - f_kq[~degenerate]) / denom[~degenerate]
    term[degenerate] = -fermi_derivative(0.5 * (xi_k[degenerate] + xi_kq[degenerate]), params.beta)
    return float(np.mean(term))


_lindhard_chi0_arrays = lindhard_chi0_arrays


def susceptibility_grid(
    mu: float,
    params: TriangularParams,
    k_grid: BZGrid,
    q_grid: BZGrid,
    *,
    twist_grid: int = 1,
) -> NDArray[np.float64]:
    """Compute ``chi0(q)`` on a reciprocal-coordinate grid."""

    qx, qy = q_grid.cartesian_mesh()

    chi = np.zeros((q_grid.nk, q_grid.nk), dtype=float)
    offsets = twist_offsets(twist_grid)
    for shift_u, shift_v in offsets:
        shifted_grid = BZGrid(k_grid.nk, shift_u=shift_u, shift_v=shift_v)
        kx, ky = shifted_grid.cartesian_mesh()
        eps_k = band_energies(params, shifted_grid)
        f_k = fermi_function(eps_k - float(mu), params.beta)

        for index in np.ndindex(chi.shape):
            chi[index] += lindhard_chi0_arrays(
                float(qx[index]),
                float(qy[index]),
                mu,
                params,
                kx,
                ky,
                eps_k,
                f_k,
            )
    chi /= float(len(offsets))
    return chi


def fermi_surface_points(
    mu: float,
    params: TriangularParams,
    grid: BZGrid,
    *,
    energy_window: float | None = None,
) -> dict[str, Any]:
    """Return grid points close to the Fermi surface."""

    kx, ky = grid.cartesian_mesh()
    u, v = grid.fractional_mesh()
    eps = triangular_dispersion(kx, ky, params)
    if energy_window is None:
        energy_window = max(4.0 / float(params.beta), 0.0125 * float(np.ptp(eps)))
    mask = np.abs(eps - float(mu)) <= float(energy_window)
    return {
        "energy_window": float(energy_window),
        "count": int(np.count_nonzero(mask)),
        "frac_u": u[mask],
        "frac_v": v[mask],
        "kx": kx[mask],
        "ky": ky[mask],
        "energy": eps[mask],
    }


def find_susceptibility_peaks(
    chi_grid: ArrayLike,
    q_grid: BZGrid,
    *,
    max_peaks: int = 12,
    exclude_gamma_radius: float = 1.0e-12,
    min_separation: float | None = None,
) -> list[dict[str, Any]]:
    """Find strong finite-Q susceptibility peaks on a periodic grid."""

    chi = np.asarray(chi_grid, dtype=float)
    if chi.shape != (q_grid.nk, q_grid.nk):
        raise ValueError("chi_grid shape must match q_grid.")
    if max_peaks < 1:
        raise ValueError("max_peaks must be >= 1.")

    u, v = q_grid.fractional_mesh()
    qx, qy = q_grid.cartesian_mesh()
    if min_separation is None:
        min_separation = 0.75 * q_grid.minimum_step_norm()

    local_max = np.ones_like(chi, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            local_max &= chi >= np.roll(np.roll(chi, di, axis=0), dj, axis=1)

    candidate_indices = np.argwhere(local_max)
    if candidate_indices.size == 0:
        candidate_indices = np.argwhere(np.ones_like(chi, dtype=bool))

    order = sorted(
        (tuple(map(int, idx)) for idx in candidate_indices),
        key=lambda idx: float(chi[idx]),
        reverse=True,
    )

    median = float(np.median(chi))
    peaks: list[dict[str, Any]] = []
    for i, j in order:
        frac = (float(u[i, j]), float(v[i, j]))
        q_norm = float(np.hypot(qx[i, j], qy[i, j]))
        if q_norm <= exclude_gamma_radius:
            continue
        too_close = any(
            periodic_distance_fractional(frac, tuple(peak["frac"])) < min_separation
            for peak in peaks
        )
        if too_close:
            continue
        value = float(chi[i, j])
        peaks.append(
            {
                "index": [int(i), int(j)],
                "frac": [frac[0], frac[1]],
                "q": [float(qx[i, j]), float(qy[i, j])],
                "norm": q_norm,
                "chi0": value,
                "relative_to_median": float(value / median) if median != 0.0 else float("inf"),
                "excess_over_median": float(value - median),
            }
        )
        if len(peaks) >= max_peaks:
            break

    return peaks


def _angle_delta(angle: NDArray[np.float64], reference: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Return angular difference in ``[-pi, pi)``."""

    return (angle - reference + PI) % (2.0 * PI) - PI


def symmetry_reduced_susceptibility_peaks(
    peaks: list[dict[str, Any]],
    *,
    sectors: int = 6,
    relative_tolerance: float = 1.0e-6,
    absolute_tolerance: float = 1.0e-10,
) -> list[dict[str, Any]]:
    """Collapse near-degenerate grid peaks into one representative per C6 sector.

    The q-grid can straddle an ideal symmetry direction and return two
    degenerate C6 orbits. This helper keeps the raw grid peaks available while
    adding a physical six-sector representation for plotting and quoted
    diagnostics.
    """

    if sectors < 1:
        raise ValueError("sectors must be >= 1.")
    if not peaks:
        return []
    if len(peaks) <= sectors:
        return [dict(peak) for peak in peaks]

    peak_q = np.asarray([peak["q"] for peak in peaks], dtype=float)
    peak_chi = np.asarray([peak["chi0"] for peak in peaks], dtype=float)
    max_chi = float(np.max(peak_chi))
    near_max = peak_chi >= max_chi - max(float(absolute_tolerance), float(relative_tolerance) * abs(max_chi))
    candidate_indices = np.nonzero(near_max)[0]
    if len(candidate_indices) <= sectors:
        return [dict(peaks[index]) for index in candidate_indices]

    candidate_q = peak_q[candidate_indices]
    candidate_chi = peak_chi[candidate_indices]
    radii = np.linalg.norm(candidate_q, axis=1)
    angles = np.arctan2(candidate_q[:, 1], candidate_q[:, 0])
    base_angle = float(np.arctan2(RECIPROCAL_B1[1], RECIPROCAL_B1[0]))
    sector_angles = base_angle + np.arange(sectors, dtype=float) * 2.0 * PI / float(sectors)
    sector_distances = np.abs(_angle_delta(angles[:, None], sector_angles[None, :]))
    sector_indices = np.argmin(sector_distances, axis=1)

    reduced: list[dict[str, Any]] = []
    for sector_index, sector_angle in enumerate(sector_angles):
        in_sector = sector_indices == sector_index
        if not np.any(in_sector):
            return [dict(peak) for peak in peaks]

        source_indices = candidate_indices[in_sector]
        radius = float(np.mean(radii[in_sector]))
        qx = float(radius * np.cos(sector_angle))
        qy = float(radius * np.sin(sector_angle))
        frac_u, frac_v = cartesian_to_fractional(qx, qy)
        chi0 = float(np.mean(candidate_chi[in_sector]))
        reduced.append(
            {
                "sector": int(sector_index),
                "frac": [float(frac_u), float(frac_v)],
                "q": [qx, qy],
                "norm": radius,
                "angle": float(sector_angle),
                "chi0": chi0,
                "source_peak_indices": [int(index) for index in source_indices],
                "source_peak_count": int(len(source_indices)),
                "reduction": "c6_sector_average",
            }
        )

    return reduced
