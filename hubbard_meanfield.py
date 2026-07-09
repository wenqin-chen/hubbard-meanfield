"""Restricted mean-field utilities for commensurate Hubbard spin textures.

This module starts Phase 3 of the project.  It builds magnetic-supercell
Bloch Hamiltonians for static Hubbard-Stratonovich exchange fields using the
same triangular-lattice hopping convention as :mod:`hubbard_nesting`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hubbard_nesting import (
    RECIPROCAL_B1,
    RECIPROCAL_B2,
    SQRT3,
    TriangularParams,
    fermi_function,
    fractional_to_cartesian,
    triangular_dispersion,
    twist_offsets,
)


REAL_A1 = np.array([1.0, 0.0], dtype=float)
REAL_A2 = np.array([0.5, 0.5 * SQRT3], dtype=float)

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

NN_DISPLACEMENTS = (
    (1, 0),
    (0, 1),
    (1, -1),
    (-1, 0),
    (0, -1),
    (-1, 1),
)
SECOND_NN_DISPLACEMENTS = (
    (1, 1),
    (2, -1),
    (1, -2),
    (-1, -1),
    (-2, 1),
    (-1, 2),
)
THIRD_NN_DISPLACEMENTS = (
    (2, 0),
    (0, 2),
    (2, -2),
    (-2, 0),
    (0, -2),
    (-2, 2),
)


def lattice_to_cartesian(n1: ArrayLike, n2: ArrayLike) -> NDArray[np.float64]:
    """Convert triangular primitive-lattice coordinates to Cartesian vectors."""

    n1_arr = np.asarray(n1, dtype=float)
    n2_arr = np.asarray(n2, dtype=float)
    x = n1_arr * REAL_A1[0] + n2_arr * REAL_A2[0]
    y = n1_arr * REAL_A1[1] + n2_arr * REAL_A2[1]
    return np.stack((x, y), axis=-1).astype(float)


def reduced_fractional_to_cartesian(
    u: ArrayLike,
    v: ArrayLike,
    supercell: "MagneticSupercell",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert reduced-BZ fractional coordinates to Cartesian momenta."""

    u_arr = np.asarray(u, dtype=float) / float(supercell.L1)
    v_arr = np.asarray(v, dtype=float) / float(supercell.L2)
    return fractional_to_cartesian(u_arr, v_arr)


@dataclass(frozen=True)
class MagneticSupercell:
    """Rectangular magnetic supercell in triangular primitive coordinates."""

    L1: int = 4
    L2: int = 4

    def __post_init__(self) -> None:
        if int(self.L1) < 1 or int(self.L2) < 1:
            raise ValueError("supercell dimensions must be positive.")

    @property
    def num_sites(self) -> int:
        return int(self.L1 * self.L2)

    @property
    def reciprocal_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return RECIPROCAL_B1 / float(self.L1), RECIPROCAL_B2 / float(self.L2)

    @property
    def cell_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return float(self.L1) * REAL_A1, float(self.L2) * REAL_A2

    def site_index(self, n1: int, n2: int) -> int:
        """Return the row-major site index for coordinates inside the cell."""

        if not (0 <= int(n1) < self.L1 and 0 <= int(n2) < self.L2):
            raise ValueError("site coordinates are outside the magnetic cell.")
        return int(n1) * int(self.L2) + int(n2)

    def site_coordinates(self, index: int) -> tuple[int, int]:
        """Return primitive-lattice coordinates for a site index."""

        idx = int(index)
        if idx < 0 or idx >= self.num_sites:
            raise ValueError("site index is outside the magnetic cell.")
        return idx // int(self.L2), idx % int(self.L2)

    def site_lattice_coordinates(self) -> NDArray[np.int64]:
        """Return all site coordinates as ``(n1, n2)`` integer rows."""

        coords = [
            self.site_coordinates(index)
            for index in range(self.num_sites)
        ]
        return np.asarray(coords, dtype=np.int64)

    def site_positions_cartesian(self) -> NDArray[np.float64]:
        """Return Cartesian positions of all microscopic sites in the cell."""

        coords = self.site_lattice_coordinates()
        return lattice_to_cartesian(coords[:, 0], coords[:, 1])

    def wrap_lattice_coordinate(self, n1: int, n2: int) -> tuple[int, int, int, int]:
        """Wrap coordinates into the cell and return magnetic-cell shifts."""

        wrapped_1 = int(n1) % int(self.L1)
        wrapped_2 = int(n2) % int(self.L2)
        shift_1 = (int(n1) - wrapped_1) // int(self.L1)
        shift_2 = (int(n2) - wrapped_2) // int(self.L2)
        return wrapped_1, wrapped_2, shift_1, shift_2

    def cell_shift_cartesian(self, shift_1: int, shift_2: int) -> NDArray[np.float64]:
        """Return the Cartesian magnetic-cell displacement."""

        cell_1, cell_2 = self.cell_vectors
        return int(shift_1) * cell_1 + int(shift_2) * cell_2

    def to_dict(self) -> dict[str, int]:
        return {"L1": int(self.L1), "L2": int(self.L2), "num_sites": self.num_sites}


@dataclass(frozen=True)
class KappaGrid:
    """Uniform reduced-zone grid for a magnetic supercell."""

    nk: int
    supercell: MagneticSupercell
    shift_u: float = 0.0
    shift_v: float = 0.0

    def __post_init__(self) -> None:
        if int(self.nk) < 1:
            raise ValueError("nk must be >= 1.")
        if not np.isfinite(float(self.shift_u)) or not np.isfinite(float(self.shift_v)):
            raise ValueError("grid shifts must be finite.")

    @property
    def frac_u_1d(self) -> NDArray[np.float64]:
        return (np.arange(self.nk, dtype=float) - self.nk // 2 + float(self.shift_u)) / float(
            self.nk
        )

    @property
    def frac_v_1d(self) -> NDArray[np.float64]:
        return (np.arange(self.nk, dtype=float) - self.nk // 2 + float(self.shift_v)) / float(
            self.nk
        )

    def fractional_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.frac_u_1d, self.frac_v_1d, indexing="ij")

    def cartesian_mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        u, v = self.fractional_mesh()
        return reduced_fractional_to_cartesian(u, v, self.supercell)

    def kappa_points(self) -> NDArray[np.float64]:
        kx, ky = self.cartesian_mesh()
        return np.stack((kx.ravel(), ky.ravel()), axis=-1).astype(float)

    def to_dict(self) -> dict[str, object]:
        return {
            "nk": int(self.nk),
            "shift_u": float(self.shift_u),
            "shift_v": float(self.shift_v),
            "supercell": self.supercell.to_dict(),
        }


def kappa_twist_grids(
    nk: int,
    supercell: MagneticSupercell,
    twist_grid: int = 1,
) -> list[KappaGrid]:
    """Return reduced-zone kappa grids over the standard twist offsets."""

    return [
        KappaGrid(int(nk), supercell, shift_u=shift_u, shift_v=shift_v)
        for shift_u, shift_v in twist_offsets(int(twist_grid))
    ]


def phase3_interaction_sweep(
    chi0_reference: float,
    ratios: Sequence[float] = (0.80, 0.95, 1.00, 1.05, 1.20, 1.50),
) -> list[dict[str, float]]:
    """Return interaction values tied to ``I_c = 1 / (2 chi0_reference)``."""

    chi = float(chi0_reference)
    if chi <= 0.0:
        raise ValueError("chi0_reference must be positive.")
    critical_i = 1.0 / (2.0 * chi)
    rows: list[dict[str, float]] = []
    for ratio in ratios:
        ratio_value = float(ratio)
        if ratio_value <= 0.0:
            raise ValueError("interaction ratios must be positive.")
        coupling = ratio_value * critical_i
        rows.append(
            {
                "I_over_Ic": ratio_value,
                "I": coupling,
                "U": 3.0 * coupling,
                "Ic": critical_i,
                "Uc": 3.0 * critical_i,
            }
        )
    return rows


def interaction_sweep_with_units(
    *,
    chi0_phase2: float,
    chi0_commensurate: float,
    ratios: Sequence[float],
    units: str,
) -> list[dict[str, float | str]]:
    """Return interaction rows in either Phase-2 or commensurate-Ic units.

    Each row carries both ``I_over_Ic_phase2`` and ``I_over_Ic_commensurate``
    so downstream consumers can present whichever scale the run was driven by
    without losing the cross-reference. ``units`` selects which Ic the input
    ``ratios`` are interpreted against.
    """

    unit = str(units)
    if unit not in {"phase2", "commensurate"}:
        raise ValueError("interaction ratio units must be 'phase2' or 'commensurate'.")
    chi_phase2 = float(chi0_phase2)
    chi_comm = float(chi0_commensurate)
    if chi_phase2 <= 0.0 or chi_comm <= 0.0:
        raise ValueError("chi0 references must be positive.")
    ic_phase2 = 1.0 / (2.0 * chi_phase2)
    ic_comm = 1.0 / (2.0 * chi_comm)
    selected_ic = ic_phase2 if unit == "phase2" else ic_comm
    rows: list[dict[str, float | str]] = []
    for ratio in ratios:
        ratio_value = float(ratio)
        if ratio_value <= 0.0 or not np.isfinite(ratio_value):
            raise ValueError("interaction ratios must be finite and positive.")
        coupling = ratio_value * selected_ic
        rows.append(
            {
                "I_over_Ic": ratio_value,
                "I_ratio_units": unit,
                "I_over_Ic_phase2": coupling / ic_phase2,
                "I_over_Ic_commensurate": coupling / ic_comm,
                "I": coupling,
                "U": 3.0 * coupling,
                "Ic": selected_ic,
                "Uc": 3.0 * selected_ic,
                "Ic_phase2": ic_phase2,
                "Uc_phase2": 3.0 * ic_phase2,
                "Ic_commensurate": ic_comm,
                "Uc_commensurate": 3.0 * ic_comm,
            }
        )
    return rows


def _directed_hoppings(params: TriangularParams) -> list[tuple[tuple[int, int], float]]:
    hoppings: list[tuple[tuple[int, int], float]] = []
    if float(params.t) != 0.0:
        hoppings.extend((disp, -float(params.t)) for disp in NN_DISPLACEMENTS)
    if float(params.t2) != 0.0:
        hoppings.extend((disp, -float(params.t2)) for disp in SECOND_NN_DISPLACEMENTS)
    if float(params.t3) != 0.0:
        hoppings.extend((disp, -float(params.t3)) for disp in THIRD_NN_DISPLACEMENTS)
    return hoppings


def validate_texture(texture: ArrayLike, supercell: MagneticSupercell) -> NDArray[np.float64]:
    """Return a texture as an ``(N_c, 3)`` real array after validation."""

    arr = np.asarray(texture, dtype=float)
    if arr.shape == (int(supercell.L1), int(supercell.L2), 3):
        arr = arr.reshape((supercell.num_sites, 3))
    if arr.shape != (supercell.num_sites, 3):
        raise ValueError(
            f"texture must have shape ({supercell.num_sites}, 3) or "
            f"({supercell.L1}, {supercell.L2}, 3)."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("texture contains non-finite values.")
    return arr.astype(float, copy=False)


def zero_texture(supercell: MagneticSupercell) -> NDArray[np.float64]:
    """Return the zero HS exchange field on a magnetic supercell."""

    return np.zeros((supercell.num_sites, 3), dtype=float)


def texture_rms_amplitude(texture: ArrayLike, supercell: MagneticSupercell) -> float:
    """Return ``sqrt((1/N_c) sum_a |S_a|^2)`` for a texture."""

    arr = validate_texture(texture, supercell)
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))))


def normalize_texture_rms(
    texture: ArrayLike,
    supercell: MagneticSupercell,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Scale a nonzero texture to the requested rms amplitude."""

    target = float(amplitude)
    if target < 0.0:
        raise ValueError("amplitude must be non-negative.")
    arr = validate_texture(texture, supercell).copy()
    current = texture_rms_amplitude(arr, supercell)
    if current == 0.0:
        if target == 0.0:
            return arr
        raise ValueError("cannot normalize a zero texture to nonzero amplitude.")
    return arr * (target / current)


def single_q_collinear_texture(
    supercell: MagneticSupercell,
    q_frac: Sequence[float],
    *,
    amplitude: float = 1.0,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """Return an rms-normalized collinear ``cos(Q.R + phase)`` texture."""

    q = np.asarray(q_frac, dtype=float)
    if q.shape != (2,):
        raise ValueError("q_frac must have length 2.")
    spin_axis = np.asarray(axis, dtype=float)
    if spin_axis.shape != (3,):
        raise ValueError("axis must have length 3.")
    axis_norm = float(np.linalg.norm(spin_axis))
    if axis_norm == 0.0:
        raise ValueError("axis must be nonzero.")
    spin_axis = spin_axis / axis_norm

    coords = supercell.site_lattice_coordinates()
    theta = 2.0 * np.pi * (q[0] * coords[:, 0] + q[1] * coords[:, 1]) + float(phase)
    raw = np.cos(theta)[:, np.newaxis] * spin_axis[np.newaxis, :]
    return normalize_texture_rms(raw, supercell, amplitude=amplitude)


def commensurate_gamma_m_q_fracs(
    supercell: MagneticSupercell,
    p: int = 1,
) -> list[tuple[float, float]]:
    """Return the closing Gamma-M triad for a rectangular magnetic cell.

    The ordering matches the Phase 2 ``gamma_m`` basis:
    ``Q1 = p b1 / L1``, ``Q2 = -p b1 / L1 - p b2 / L2``, ``Q3 = p b2 / L2``.
    Default ``p = 1`` is the smallest commensurate Gamma-M wavevector and, for
    the ``4 x 4`` cell, the rational approximation to the Phase 1 peak triad.
    Larger integer ``p`` selects a different rational approximant ``p / L`` of
    the Phase 1 peak ``2 k_F / |b1| ~= 0.246283`` and is used by the Phase 3b
    finite-cell rational-approximant check.

    Constraints: ``2 p < min(L1, L2)`` so each Q sits strictly inside the
    open half of the reduced zone. Equivalently ``p < L / 2``: at ``p = L/2``
    the wavevector lands on the zone-boundary M point where ``Q == -Q`` modulo
    a reciprocal lattice vector and the ``+-Q`` mode-pair bookkeeping
    breaks; ``p > L/2`` is redundant under
    ``p -> L - p`` with a sign flip.
    """

    p_int = int(p)
    upper = min(int(supercell.L1), int(supercell.L2))
    if p_int < 1 or 2 * p_int >= upper:
        raise ValueError(
            "commensurate_gamma_m_q_fracs requires 1 <= p < min(L1, L2) / 2 "
            "to keep Q strictly inside the open half-zone (avoid Q = -Q at "
            f"the M point); got p={p_int} for supercell "
            f"({supercell.L1}, {supercell.L2})."
        )
    return [
        (float(p_int) / float(supercell.L1), 0.0),
        (-float(p_int) / float(supercell.L1), -float(p_int) / float(supercell.L2)),
        (0.0, float(p_int) / float(supercell.L2)),
    ]


def texture_from_mode_spinors(
    supercell: MagneticSupercell,
    q_fracs: Sequence[Sequence[float]],
    spinors: ArrayLike,
    *,
    amplitude: float | None = None,
) -> NDArray[np.float64]:
    """Build a real texture from complex ``S_Q`` spin amplitudes.

    ``spinors[n]`` is the complex vector amplitude at ``+Q_n``.  The real-space
    field is ``sum_n S_Qn exp(i Q_n.R) + c.c.``.  If ``amplitude`` is supplied,
    the result is rescaled to that rms amplitude.
    """

    q_arr = np.asarray(q_fracs, dtype=float)
    z = np.asarray(spinors, dtype=complex)
    if q_arr.ndim != 2 or q_arr.shape[1] != 2:
        raise ValueError("q_fracs must have shape (num_modes, 2).")
    if z.shape != (q_arr.shape[0], 3):
        raise ValueError("spinors must have shape (num_modes, 3).")
    if not np.all(np.isfinite(q_arr)) or not np.all(np.isfinite(z)):
        raise ValueError("q_fracs and spinors must contain finite values.")

    coords = supercell.site_lattice_coordinates()
    field = np.zeros((supercell.num_sites, 3), dtype=complex)
    for q, spinor in zip(q_arr, z):
        theta = 2.0 * np.pi * (q[0] * coords[:, 0] + q[1] * coords[:, 1])
        phase = np.exp(1.0j * theta)
        field += phase[:, np.newaxis] * spinor[np.newaxis, :]
        field += np.conjugate(phase[:, np.newaxis] * spinor[np.newaxis, :])
    texture = np.real_if_close(field, tol=1000).real.astype(float)
    if amplitude is None:
        return validate_texture(texture, supercell)
    return normalize_texture_rms(texture, supercell, amplitude=float(amplitude))


def uniform_ferromagnetic_texture(
    supercell: MagneticSupercell,
    *,
    amplitude: float = 1.0,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> NDArray[np.float64]:
    """Return a uniform exchange field with the requested rms amplitude."""

    spin_axis = np.asarray(axis, dtype=float)
    if spin_axis.shape != (3,):
        raise ValueError("axis must have length 3.")
    norm = float(np.linalg.norm(spin_axis))
    if norm == 0.0:
        raise ValueError("axis must be nonzero.")
    texture = np.tile(spin_axis / norm, (supercell.num_sites, 1))
    return normalize_texture_rms(texture, supercell, amplitude=float(amplitude))


def oriented_triangle_indices(supercell: MagneticSupercell) -> NDArray[np.int64]:
    """Return the two counter-clockwise elementary triangles per site."""

    triangles: list[tuple[int, int, int]] = []
    for n1 in range(int(supercell.L1)):
        for n2 in range(int(supercell.L2)):
            i = supercell.site_index(n1, n2)

            up_1, up_2, _, _ = supercell.wrap_lattice_coordinate(n1 + 1, n2)
            up_3, up_4, _, _ = supercell.wrap_lattice_coordinate(n1, n2 + 1)
            triangles.append(
                (
                    i,
                    supercell.site_index(up_1, up_2),
                    supercell.site_index(up_3, up_4),
                )
            )

            down_1, down_2, _, _ = supercell.wrap_lattice_coordinate(n1 + 1, n2 - 1)
            down_3, down_4, _, _ = supercell.wrap_lattice_coordinate(n1 + 1, n2)
            triangles.append(
                (
                    i,
                    supercell.site_index(down_1, down_2),
                    supercell.site_index(down_3, down_4),
                )
            )
    return np.asarray(triangles, dtype=np.int64)


def scalar_chiralities(
    texture: ArrayLike,
    supercell: MagneticSupercell,
) -> NDArray[np.float64]:
    """Return ``S_i . (S_j x S_k)`` on oriented elementary triangles."""

    arr = validate_texture(texture, supercell)
    triangles = oriented_triangle_indices(supercell)
    values = [
        float(np.dot(arr[i], np.cross(arr[j], arr[k])))
        for i, j, k in triangles
    ]
    return np.asarray(values, dtype=float)


def chirality_summary(texture: ArrayLike, supercell: MagneticSupercell) -> dict[str, float]:
    """Return signed and absolute scalar-chirality diagnostics."""

    values = scalar_chiralities(texture, supercell)
    return {
        "mean_signed": float(np.mean(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "max_absolute": float(np.max(np.abs(values))),
    }


def berg_luescher_skyrmion_number(
    texture: ArrayLike,
    supercell: MagneticSupercell,
    *,
    min_norm_threshold: float = 1.0e-10,
) -> dict[str, float | str]:
    """Return the Berg-Luescher lattice skyrmion number, when defined."""

    arr = validate_texture(texture, supercell)
    norms = np.linalg.norm(arr, axis=1)
    min_norm = float(np.min(norms))
    if min_norm < float(min_norm_threshold):
        return {
            "status": "undefined_zero_moment",
            "min_norm": min_norm,
            "threshold": float(min_norm_threshold),
        }
    unit = arr / norms[:, np.newaxis]
    total = 0.0
    for i, j, k in oriented_triangle_indices(supercell):
        numerator = float(np.dot(unit[i], np.cross(unit[j], unit[k])))
        denominator = float(
            1.0
            + np.dot(unit[i], unit[j])
            + np.dot(unit[j], unit[k])
            + np.dot(unit[k], unit[i])
        )
        total += 2.0 * float(np.arctan2(numerator, denominator))
    return {
        "status": "ok",
        "number": float(total / (4.0 * np.pi)),
        "min_norm": min_norm,
        "threshold": float(min_norm_threshold),
    }


def hs_cost_density(texture: ArrayLike, supercell: MagneticSupercell, I: float) -> float:
    """Return ``(1/N_c) sum_a |S_a|^2 / (2 I)``."""

    coupling = float(I)
    if coupling <= 0.0:
        raise ValueError("I must be positive.")
    arr = validate_texture(texture, supercell)
    return float(np.mean(np.sum(arr * arr, axis=1)) / (2.0 * coupling))


def build_supercell_hamiltonian(
    kappa: Sequence[float],
    texture: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
) -> NDArray[np.complex128]:
    """Return the ``2 N_c x 2 N_c`` Bloch Hamiltonian for a static texture.

    When ``params.h_z`` or ``params.alpha_rashba`` is nonzero, additional
    contributions are appended (uniform Zeeman, NN Rashba spinor hopping).
    Both default to ``0.0`` and the additions are gated, so the spin-
    degenerate Hubbard Hamiltonian is bit-reproduced when the defaults
    are in effect.
    """

    kappa_vec = np.asarray(kappa, dtype=float)
    if kappa_vec.shape != (2,):
        raise ValueError("kappa must be a Cartesian momentum with length 2.")
    spin_field = validate_texture(texture, supercell)
    n_sites = supercell.num_sites
    h_site = np.zeros((n_sites, n_sites), dtype=complex)

    for source in range(n_sites):
        n1, n2 = supercell.site_coordinates(source)
        for (d1, d2), hopping in _directed_hoppings(params):
            target_n1, target_n2 = n1 + d1, n2 + d2
            wrapped_1, wrapped_2, shift_1, shift_2 = supercell.wrap_lattice_coordinate(
                target_n1,
                target_n2,
            )
            target = supercell.site_index(wrapped_1, wrapped_2)
            cell_shift = supercell.cell_shift_cartesian(shift_1, shift_2)
            phase = np.exp(1.0j * float(np.dot(kappa_vec, cell_shift)))
            h_site[source, target] += float(hopping) * phase

    h = np.kron(h_site, np.eye(2, dtype=complex))

    alpha = float(params.alpha_rashba)
    if alpha != 0.0:
        for source in range(n_sites):
            n1, n2 = supercell.site_coordinates(source)
            for d1, d2 in NN_DISPLACEMENTS:
                target_n1, target_n2 = n1 + d1, n2 + d2
                wrapped_1, wrapped_2, shift_1, shift_2 = supercell.wrap_lattice_coordinate(
                    target_n1,
                    target_n2,
                )
                target = supercell.site_index(wrapped_1, wrapped_2)
                cell_shift = supercell.cell_shift_cartesian(shift_1, shift_2)
                phase = np.exp(1.0j * float(np.dot(kappa_vec, cell_shift)))
                d_cart = lattice_to_cartesian(d1, d2)
                d_norm = float(np.linalg.norm(d_cart))
                d_hat_x = float(d_cart[0]) / d_norm
                d_hat_y = float(d_cart[1]) / d_norm
                spinor_block = (
                    -1.0j * alpha * (-d_hat_y * PAULI_X + d_hat_x * PAULI_Y) * phase
                )
                src_blk = slice(2 * source, 2 * source + 2)
                tgt_blk = slice(2 * target, 2 * target + 2)
                h[src_blk, tgt_blk] += spinor_block

    for site, (sx, sy, sz) in enumerate(spin_field):
        exchange = sx * PAULI_X + sy * PAULI_Y + sz * PAULI_Z
        block = slice(2 * site, 2 * site + 2)
        h[block, block] -= exchange

    h_zeeman = float(params.h_z)
    if h_zeeman != 0.0:
        zeeman_term = -h_zeeman * (0.5 * PAULI_Z)
        for site in range(n_sites):
            block = slice(2 * site, 2 * site + 2)
            h[block, block] += zeeman_term

    # Easy-axis anisotropy: site-dependent z-Zeeman proportional to
    # the texture's z-component. Convention is the HS-field-quadratic
    # form (see TriangularParams docstring):
    #
    #     ΔF_anis[S_HS] = -A * Σ_i (S_HS,i^z)²
    #
    # By the variational principle d(F_band + F_anis)/d(<σ_i^z>) at
    # saddle, the band Hamiltonian gains -A * S_HS,i^z * σ_z per site.
    # This is a TEXTURE-DEPENDENT term — completely distinct from the
    # uniform-Zeeman h_z above, which adds a texture-independent global
    # σ_z. At A = 0 this loop is a no-op and the band Hamiltonian is
    # bit-equal to the previous version.
    easy_axis_A = float(params.easy_axis_A)
    if easy_axis_A != 0.0:
        for site, (_, _, sz) in enumerate(spin_field):
            block = slice(2 * site, 2 * site + 2)
            h[block, block] -= easy_axis_A * float(sz) * PAULI_Z

    return np.asarray(h, dtype=np.complex128)


def folded_paramagnetic_bands(
    kappa: Sequence[float],
    params: TriangularParams,
    supercell: MagneticSupercell,
    *,
    include_spin: bool = True,
) -> NDArray[np.float64]:
    """Return folded triangular bands at ``S=0`` for a magnetic supercell."""

    kappa_vec = np.asarray(kappa, dtype=float)
    if kappa_vec.shape != (2,):
        raise ValueError("kappa must be a Cartesian momentum with length 2.")

    energies: list[float] = []
    for m1 in range(int(supercell.L1)):
        for m2 in range(int(supercell.L2)):
            shift = (m1 / float(supercell.L1)) * RECIPROCAL_B1
            shift += (m2 / float(supercell.L2)) * RECIPROCAL_B2
            momentum = kappa_vec + shift
            energy = triangular_dispersion(momentum[0], momentum[1], params)
            if include_spin:
                energies.extend([float(energy), float(energy)])
            else:
                energies.append(float(energy))
    return np.asarray(energies, dtype=float)


def band_eigenvalues_on_grid(
    texture: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    grid: KappaGrid,
    *,
    workers: int = 1,
) -> NDArray[np.float64]:
    """Diagonalize the supercell Hamiltonian on one reduced-zone grid."""

    if grid.supercell != supercell:
        raise ValueError("grid.supercell must match the Hamiltonian supercell.")
    spin_field = validate_texture(texture, supercell)
    bands = 2 * supercell.num_sites
    values = np.empty((int(grid.nk), int(grid.nk), bands), dtype=float)
    kx_mesh, ky_mesh = grid.cartesian_mesh()
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be >= 1.")
    if worker_count > 1 and int(grid.nk) > 1:
        tasks = [
            (i, kx_mesh[i, :], ky_mesh[i, :], spin_field, params, supercell)
            for i in range(int(grid.nk))
        ]
        with ProcessPoolExecutor(max_workers=min(worker_count, int(grid.nk))) as executor:
            for i, row_values in executor.map(_diagonalize_kappa_row, tasks):
                values[i, :, :] = row_values
        return values
    for i in range(int(grid.nk)):
        for j in range(int(grid.nk)):
            kappa = (float(kx_mesh[i, j]), float(ky_mesh[i, j]))
            hamiltonian = build_supercell_hamiltonian(kappa, spin_field, params, supercell)
            values[i, j, :] = np.linalg.eigvalsh(hamiltonian)
    return values


def _diagonalize_kappa_row(
    task: tuple[
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        TriangularParams,
        MagneticSupercell,
    ],
) -> tuple[int, NDArray[np.float64]]:
    i, kx_row, ky_row, spin_field, params, supercell = task
    row_values = np.empty((int(kx_row.size), 2 * supercell.num_sites), dtype=float)
    for j in range(int(kx_row.size)):
        kappa = (float(kx_row[j]), float(ky_row[j]))
        hamiltonian = build_supercell_hamiltonian(kappa, spin_field, params, supercell)
        row_values[j, :] = np.linalg.eigvalsh(hamiltonian)
    return int(i), row_values


def _diagonalize_twist_kappa_row(
    task: tuple[
        int,
        int,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        TriangularParams,
        MagneticSupercell,
    ],
) -> tuple[int, int, NDArray[np.float64]]:
    grid_index, row_index, kx_row, ky_row, spin_field, params, supercell = task
    _, row_values = _diagonalize_kappa_row(
        (row_index, kx_row, ky_row, spin_field, params, supercell)
    )
    return int(grid_index), int(row_index), row_values


def band_eigenvalues_twist_averaged(
    texture: ArrayLike,
    params: TriangularParams,
    supercell: MagneticSupercell,
    *,
    kappa_nk: int,
    twist_grid: int = 1,
    workers: int = 1,
) -> NDArray[np.float64]:
    """Return eigenvalues on all reduced-zone twist grids.

    The returned shape is ``(twist_count, kappa_nk, kappa_nk, 2 N_c)``.  The
    thermodynamic helpers average over all leading axes.
    """

    grids = kappa_twist_grids(int(kappa_nk), supercell, int(twist_grid))
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be >= 1.")
    if worker_count > 1 and len(grids) > 1:
        spin_field = validate_texture(texture, supercell)
        bands = 2 * supercell.num_sites
        values = np.empty(
            (len(grids), int(kappa_nk), int(kappa_nk), bands),
            dtype=float,
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
        with ProcessPoolExecutor(max_workers=min(worker_count, len(tasks))) as executor:
            for grid_index, row_index, row_values in executor.map(
                _diagonalize_twist_kappa_row,
                tasks,
            ):
                values[grid_index, row_index, :, :] = row_values
        return values
    return np.stack(
        [
            band_eigenvalues_on_grid(
                texture,
                params,
                supercell,
                grid,
                workers=int(workers),
            )
            for grid in grids
        ],
        axis=0,
    )


def _num_sites_from_eigenvalues(eigenvalues: ArrayLike) -> int:
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim < 1:
        raise ValueError("eigenvalues must have at least one dimension.")
    bands = int(values.shape[-1])
    if bands < 2 or bands % 2 != 0:
        raise ValueError("last eigenvalue axis must have length 2*N_c.")
    return bands // 2


def _texture_cost_density_from_bands(texture: ArrayLike, eigenvalues: ArrayLike, I: float) -> float:
    n_sites = _num_sites_from_eigenvalues(eigenvalues)
    coupling = float(I)
    if coupling <= 0.0:
        raise ValueError("I must be positive.")
    arr = np.asarray(texture, dtype=float)
    if arr.shape[-1:] != (3,):
        raise ValueError("texture must have a final axis of length 3.")
    arr = arr.reshape((-1, 3))
    if arr.shape != (n_sites, 3):
        raise ValueError(f"texture must contain {n_sites} sites.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("texture contains non-finite values.")
    return float(np.mean(np.sum(arr * arr, axis=1)) / (2.0 * coupling))


def filling_from_eigenvalues(mu: float, eigenvalues: ArrayLike, beta: float) -> float:
    """Return spinful filling per microscopic site from supercell eigenvalues."""

    values = np.asarray(eigenvalues, dtype=float)
    n_sites = _num_sites_from_eigenvalues(values)
    occupations = fermi_function(values - float(mu), float(beta))
    return float(np.mean(np.sum(occupations, axis=-1)) / float(n_sites))


def grand_potential(
    mu: float,
    eigenvalues: ArrayLike,
    texture: ArrayLike,
    I: float,
    beta: float,
    *,
    easy_axis_A: float = 0.0,
) -> float:
    """Return the fixed-chemical-potential grand-potential density.

    Optionally includes a uniaxial easy-axis correction
    ``ΔF_anis/N_sites = -A * <(S_i^z)²>_i`` (HS-field convention; see
    :class:`TriangularParams` docstring). At ``easy_axis_A = 0`` the
    return value is bit-identical to the pre-anisotropy version.
    """

    values = np.asarray(eigenvalues, dtype=float)
    n_sites = _num_sites_from_eigenvalues(values)
    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive.")
    hs_cost = _texture_cost_density_from_bands(texture, values, I)
    fermion = -float(
        np.mean(np.sum(np.logaddexp(0.0, -beta_value * (values - float(mu))), axis=-1))
        / (beta_value * float(n_sites))
    )
    anis = 0.0
    if float(easy_axis_A) != 0.0:
        arr = np.asarray(texture, dtype=float).reshape(-1, 3)
        if arr.shape[0] != int(n_sites):
            raise ValueError(
                "texture site count must match the eigenvalue site count for "
                "the easy-axis F correction; "
                f"got texture shape {arr.shape} vs n_sites={n_sites}."
            )
        anis = -float(easy_axis_A) * float(np.mean(arr[:, 2] ** 2))
    return hs_cost + fermion + anis


def chemical_potential_for_eigenvalues(
    filling: float,
    eigenvalues: ArrayLike,
    beta: float,
    *,
    tolerance: float = 1.0e-11,
    max_iterations: int = 200,
) -> float:
    """Solve for the chemical potential giving a target spinful filling."""

    target = float(filling)
    if target < 0.0 or target > 2.0:
        raise ValueError("filling must satisfy 0 <= filling <= 2.")
    values = np.asarray(eigenvalues, dtype=float)
    _num_sites_from_eigenvalues(values)
    beta_value = float(beta)
    if beta_value <= 0.0:
        raise ValueError("beta must be positive.")
    if target == 0.0:
        return float(np.min(values) - 50.0 / beta_value)
    if target == 2.0:
        return float(np.max(values) + 50.0 / beta_value)

    margin = max(10.0, 50.0 / beta_value)
    lower = float(np.min(values) - margin)
    upper = float(np.max(values) + margin)
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lower + upper)
        current = filling_from_eigenvalues(mid, values, beta_value)
        if abs(current - target) <= float(tolerance):
            return float(mid)
        if current < target:
            lower = mid
        else:
            upper = mid
    return float(0.5 * (lower + upper))


def fixed_density_free_energy(
    filling: float,
    eigenvalues: ArrayLike,
    texture: ArrayLike,
    I: float,
    beta: float,
    *,
    tolerance: float = 1.0e-11,
    easy_axis_A: float = 0.0,
) -> float:
    """Return the Helmholtz free-energy density at fixed spinful filling.

    ``easy_axis_A`` is forwarded to :func:`grand_potential`; at the
    default ``0.0`` the return value is bit-identical to the
    pre-anisotropy implementation.
    """

    mu = chemical_potential_for_eigenvalues(
        filling,
        eigenvalues,
        beta,
        tolerance=tolerance,
    )
    return grand_potential(
        mu, eigenvalues, texture, I, beta,
        easy_axis_A=float(easy_axis_A),
    ) + float(mu) * float(filling)


def fixed_density_free_energy_difference(
    filling: float,
    eigenvalues: ArrayLike,
    texture: ArrayLike,
    paramagnetic_eigenvalues: ArrayLike,
    paramagnetic_texture: ArrayLike,
    I: float,
    beta: float,
    *,
    tolerance: float = 1.0e-11,
) -> float:
    """Return ``F_n[S] - F_n[0]`` using the same supercell convention."""

    textured = fixed_density_free_energy(
        filling,
        eigenvalues,
        texture,
        I,
        beta,
        tolerance=tolerance,
    )
    reference = fixed_density_free_energy(
        filling,
        paramagnetic_eigenvalues,
        paramagnetic_texture,
        I,
        beta,
        tolerance=tolerance,
    )
    return float(textured - reference)
