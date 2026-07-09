"""Channel-resolved Stoner pre-flight under Rashba SOC.

The scalar (spin-degenerate) Stoner criterion ``1 - I chi_0(Q*) = 0``
splits under SOC into longitudinal (``chi_0^{zz}``) and transverse
(``chi_0^{+-}``) channels with potentially different ``I_c`` values.
This pre-flight identifies which channel goes unstable first as
``alpha`` grows -- this informs the seed-strategy choice for the
production sweep (out-of-plane ``S^z_0 > 0`` for longitudinal-leading,
in-plane / helix for transverse-leading).

The non-interacting susceptibility on the spin-orbit-coupled triangular
lattice is

.. math::

   \\chi_0^{ab}(Q) = -\\frac{1}{N_k} \\sum_{k, n, m}
   \\frac{f(E_n(k) - \\mu) - f(E_m(k + Q) - \\mu)}
        {E_n(k) - E_m(k + Q)}
   \\,\\bigl|\\langle u_n(k) | \\sigma^a | u_m(k+Q) \\rangle\\bigr|^2,

where ``|u_n(k)>`` is the 2-component spinor eigenstate of the
``2 x 2`` Bloch Hamiltonian at ``k`` with NN Rashba coupling. We use
the L=1 ``MagneticSupercell`` to build that Hamiltonian directly via
``build_supercell_hamiltonian(texture = 0, params.alpha_rashba = alpha)``;
the U=0 bands are independent of any supercell folding choice, so this
is the minimal-cost evaluation that respects the project's Hamiltonian
conventions (matrix elements, kappa shifts, etc.).

The helper returns the per-Q susceptibility in each channel, the
maximizing Q, and ``I_c = 1 / (2 chi_max)`` for each channel.

Heavy-pre-flight schedule
-------------------------

A coarse run at ``chi0_grid_nk = 24`` and a handful of Q points
completes in < 1 s. The full pre-flight at L=9,
``kappa = 72``, ``twist = 3`` translates here to ``chi0_grid_nk = 72``
and ``twist_grid = 3`` (the same total k-grid resolution); that takes
about 30 min per alpha when scanning a dense Q-ring. We ship the
helper + a coarse-grid validation; the heavy pre-flight is deferred
to the user. See ``README.md``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from hubbard_meanfield import (
    MagneticSupercell,
    build_supercell_hamiltonian,
    chemical_potential_for_eigenvalues,
    zero_texture,
)
from hubbard_nesting import (
    RECIPROCAL_B1,
    RECIPROCAL_B2,
    TriangularParams,
    fermi_derivative,
    fermi_function,
    twist_offsets,
)


def mu_for_target_filling(
    *,
    params: TriangularParams,
    alpha: float,
    filling: float,
    grid_nk: int,
    twist_grid: int = 1,
    h_z: float = 0.0,
) -> float:
    """Solve for the chemical potential ``mu`` such that the non-
    interacting (U=0) Rashba + Zeeman bands are filled to spinful
    density ``filling`` per microscopic site.

    Under Rashba SOC the bands spin-split (and shift in energy);
    under a Zeeman field the up / down bands rigid-shift by ``-h_z/2``
    and ``+h_z/2``. Either effect moves the Fermi level off the
    ``alpha = 0, h_z = 0`` Wang-filling value, so the Phase 1 / Phase 2
    / Phase 3 pre-flight must call this helper once per
    ``(alpha, h_z)`` before ``channel_resolved_stoner`` to keep the
    Lindhard bubble at fixed filling.

    Implementation: build the L=1 supercell U=0 Hamiltonian (with the
    requested ``alpha`` and ``h_z`` in ``TriangularParams``) on the
    same BZ grid the susceptibility uses, diagonalize, and bisect on
    mu via
    :func:`hubbard_meanfield.chemical_potential_for_eigenvalues`.
    """
    cell1 = MagneticSupercell(1, 1)
    params_with_field = TriangularParams(
        t=float(params.t), t2=float(params.t2), t3=float(params.t3),
        beta=float(params.beta),
        alpha_rashba=float(alpha),
        h_z=float(h_z),
    )
    tex = zero_texture(cell1)
    eig_grid: list[NDArray[np.float64]] = []
    for shift_u, shift_v in twist_offsets(int(twist_grid)):
        k_pts = _bz_k_points(int(grid_nk), float(shift_u), float(shift_v))
        n_k = k_pts.shape[0]
        row = np.empty((n_k, 2), dtype=float)
        for idx in range(n_k):
            H = build_supercell_hamiltonian(
                k_pts[idx], tex, params_with_field, cell1,
            )
            row[idx] = np.linalg.eigvalsh(H)
        eig_grid.append(row)
    eigenvalues = np.concatenate(eig_grid, axis=0)
    return float(
        chemical_potential_for_eigenvalues(
            float(filling), eigenvalues, float(params.beta),
        )
    )


def _bz_k_points(nk: int, shift_u: float, shift_v: float) -> NDArray[np.float64]:
    """Return ``(nk*nk, 2)`` Cartesian momenta on the centered BZ grid
    with the given fractional shift (the ``twist_offsets`` convention).
    """
    u = (np.arange(nk, dtype=float) - nk // 2 + shift_u) / float(nk)
    v = (np.arange(nk, dtype=float) - nk // 2 + shift_v) / float(nk)
    U, V = np.meshgrid(u, v, indexing="ij")
    kx = U * RECIPROCAL_B1[0] + V * RECIPROCAL_B2[0]
    ky = U * RECIPROCAL_B1[1] + V * RECIPROCAL_B2[1]
    return np.stack((kx.ravel(), ky.ravel()), axis=-1)


def _chi0_at_q(
    q_cart: NDArray[np.float64],
    *,
    params_alpha: TriangularParams,
    mu: float,
    chi0_grid_nk: int,
    twist_grid: int,
) -> tuple[float, float, float]:
    """Return ``(chi_zz, chi_pm, chi_mp)`` for one Q (Cartesian) by
    summing the noninteracting Lindhard expression over a BZ k-grid
    with twist averaging.

    ``params_alpha`` carries both ``alpha_rashba`` and ``h_z`` -- the
    Hamiltonian builder adds the Zeeman block when ``h_z != 0``, so
    this helper handles Rashba-only, Zeeman-only, and combined
    Rashba+Zeeman cases uniformly.

    Note on transverse channels (post-review fix):

    - ``chi_pm`` uses the ``sigma_+`` matrix element
      ``<n,k|sigma_+|m,k+Q>`` and gives the ``S^+ S^-`` static bubble.
    - ``chi_mp`` uses ``sigma_-`` and gives the ``S^- S^+`` bubble.
    - At ``alpha = 0, h_z = 0`` (or under T-invariant Rashba alone)
      ``chi_pm = chi_mp`` to machine precision.
    - Under finite Zeeman the two split (Zeeman gaps the magnon
      branches differently), and the *physical* transverse Stoner
      response is the average ``chi_xx_plus_yy / 2 = (chi_pm + chi_mp) / 2``
      (rotational symmetry about z gives ``chi_xx = chi_yy``).
      :func:`channel_resolved_stoner` exposes this as
      ``chi_transverse``.
    """
    cell1 = MagneticSupercell(1, 1)
    tex = zero_texture(cell1)
    beta = float(params_alpha.beta)

    chi_zz = 0.0
    chi_pm = 0.0
    chi_mp = 0.0
    n_twists = 0
    for shift_u, shift_v in twist_offsets(int(twist_grid)):
        k_pts = _bz_k_points(int(chi0_grid_nk), float(shift_u), float(shift_v))
        # Diagonalize the U=0 Hamiltonian at every k and k+Q.
        n_k = k_pts.shape[0]
        E_k = np.empty((n_k, 2), dtype=float)
        V_k = np.empty((n_k, 2, 2), dtype=complex)
        E_kq = np.empty((n_k, 2), dtype=float)
        V_kq = np.empty((n_k, 2, 2), dtype=complex)
        for idx in range(n_k):
            k = k_pts[idx]
            H_k = build_supercell_hamiltonian(k, tex, params_alpha, cell1)
            E_k[idx], V_k[idx] = np.linalg.eigh(H_k)
            H_kq = build_supercell_hamiltonian(k + q_cart, tex, params_alpha, cell1)
            E_kq[idx], V_kq[idx] = np.linalg.eigh(H_kq)
        f_k = fermi_function(E_k - float(mu), beta)
        f_kq = fermi_function(E_kq - float(mu), beta)
        # Sum the four band-pair channels. The sign and normalization
        # below are chosen so that at alpha=0 both chi_zz and chi_pm
        # collapse to the scalar Lindhard chi_0
        # (per-spin convention from hubbard_nesting.lindhard_chi0:
        # chi_0(q) = -<(f_k - f_kq) / (eps_k - eps_kq)>_k > 0).
        #
        # The (1/2) factor on chi_zz is the longitudinal-channel
        # normalization: a longitudinal probe (sigma_z) couples to both
        # the (up, up) and (down, down) band pairs equally, so the
        # naive sum is 2x the per-channel response. Dividing by 2
        # makes chi_zz the per-channel longitudinal susceptibility,
        # matching the scalar Lindhard at alpha=0 and giving a clean
        # comparison to chi_pm (which has only one (up, down) pair).
        chi_zz_twist = 0.0
        chi_pm_twist = 0.0
        chi_mp_twist = 0.0
        # Degenerate-denominator threshold: match the
        # ``lindhard_chi0_arrays`` convention so the spinor kernel
        # picks up the same Fermi-derivative contribution at the
        # iso-energy crossings (eps_k = eps_kq) the scalar Lindhard
        # would. Without this substitution the spinor path under-
        # estimates chi_zz / chi_pm wherever the iso-energy contour
        # touches the Fermi surface; the discrepancy is silent at
        # h_z = 0 in interior of the Q-grid (where the existing
        # alpha = 0 collapse test sits) but becomes O(10%) at
        # h_z > 0 / alpha > 0.
        scale = max(1.0, float(np.max(np.abs(E_k))), abs(float(mu)))
        deg_tol = 1.0e-10 * scale
        for n in range(2):
            for m in range(2):
                denom = E_k[:, n] - E_kq[:, m]
                degenerate = np.abs(denom) < deg_tol
                df = f_k[:, n] - f_kq[:, m]
                #   <n,k|sigma_z|m,k+Q> = conj(V_k[0,n]) V_kq[0,m]
                #                       - conj(V_k[1,n]) V_kq[1,m]
                #   <n,k|sigma_+|m,k+Q> = conj(V_k[0,n]) V_kq[1,m]
                #   <n,k|sigma_-|m,k+Q> = conj(V_k[1,n]) V_kq[0,m]
                m_zz = (
                    np.conj(V_k[:, 0, n]) * V_kq[:, 0, m]
                    - np.conj(V_k[:, 1, n]) * V_kq[:, 1, m]
                )
                m_pm = np.conj(V_k[:, 0, n]) * V_kq[:, 1, m]
                m_mp = np.conj(V_k[:, 1, n]) * V_kq[:, 0, m]
                # Regular kernel away from degeneracy: -|m|^2 * df / denom.
                # At degenerate denominators substitute -d f / d xi
                # at the mean ``xi`` (since (f_k - f_kq)/(xi_k - xi_kq)
                # → d f / d xi as xi_kq → xi_k, and the chi_0 kernel
                # is *minus* this ratio). Same convention as the parent
                # ``hubbard_nesting.lindhard_chi0_arrays``.
                xi_mean = 0.5 * (
                    (E_k[:, n] - float(mu)) + (E_kq[:, m] - float(mu))
                )
                neg_d_fermi = -fermi_derivative(xi_mean, beta)
                kernel = np.where(
                    degenerate,
                    neg_d_fermi,
                    -df / np.where(degenerate, 1.0, denom),
                )
                weight_zz = np.abs(m_zz) ** 2 * kernel
                weight_pm = np.abs(m_pm) ** 2 * kernel
                weight_mp = np.abs(m_mp) ** 2 * kernel
                chi_zz_twist += float(np.sum(weight_zz))
                chi_pm_twist += float(np.sum(weight_pm))
                chi_mp_twist += float(np.sum(weight_mp))
        # Per-channel longitudinal normalization (see comment above).
        chi_zz += 0.5 * chi_zz_twist / float(n_k)
        chi_pm += chi_pm_twist / float(n_k)
        chi_mp += chi_mp_twist / float(n_k)
        n_twists += 1
    return (
        float(chi_zz / max(n_twists, 1)),
        float(chi_pm / max(n_twists, 1)),
        float(chi_mp / max(n_twists, 1)),
    )


def channel_resolved_stoner(
    *,
    params: TriangularParams,
    alpha: float,
    q_fracs: Sequence[Sequence[float]],
    chi0_grid_nk: int,
    twist_grid: int = 1,
    mu: float,
    h_z: float = 0.0,
) -> dict:
    """Return per-Q longitudinal and transverse Lindhard susceptibility
    plus channel-resolved ``I_c = 1 / (2 chi_max)``.

    Parameters
    ----------
    params
        ``TriangularParams`` with the desired ``t``, ``t2``, ``t3``, and
        ``beta``. ``params.alpha_rashba`` and ``params.h_z`` are
        overridden by the ``alpha`` and ``h_z`` arguments.
    alpha
        Rashba coupling for this pre-flight point.
    q_fracs
        Iterable of ``(q1, q2)`` fractional reciprocal-lattice
        coordinates of the test Q vectors.
    chi0_grid_nk
        Linear resolution of the k-grid on the primitive BZ.
    twist_grid
        Twist-average count along each direction (the ``twist_grid ** 2``
        offsets from ``hubbard_nesting.twist_offsets`` are averaged).
    mu
        Chemical potential. The caller is responsible for picking a mu
        consistent with the target filling on the noninteracting U=0
        bands at this ``(alpha, h_z)``. Use
        :func:`mu_for_target_filling` to compute it.
    h_z
        Uniform Zeeman field. ``h_z = 0`` reproduces the original
        Rashba-only helper. ``alpha = 0, h_z > 0``
        gives the finite-Zeeman Stoner pre-flight:
        the up- and down-bands rigid-shift by ``-h_z/2`` and
        ``+h_z/2``, ``chi_zz`` decomposes into the per-spin Lindhard
        average, and ``chi_pm`` becomes Zeeman-detuned (the ``+-``
        bubble denominator picks up the inter-spin energy splitting).

    Returns
    -------
    dict with:

    - ``alpha``, ``h_z``, ``mu``, ``q_fracs_cartesian`` (parameter echo);
    - ``chi_zz``, ``chi_pm``: lists of per-Q susceptibilities;
    - ``chi_zz_max``, ``chi_pm_max``: max over the Q grid;
    - ``q_max_zz``, ``q_max_pm``: the Q (fractional) where each channel
      peaks;
    - ``I_c_longitudinal``, ``I_c_transverse``: ``1 / (2 chi_max)`` per
      channel; ``inf`` when the channel's chi_max is non-positive.
    """
    params_alpha = TriangularParams(
        t=float(params.t), t2=float(params.t2), t3=float(params.t3),
        beta=float(params.beta),
        alpha_rashba=float(alpha),
        h_z=float(h_z),
    )
    q_fracs_arr = np.asarray(q_fracs, dtype=float)
    if q_fracs_arr.ndim != 2 or q_fracs_arr.shape[1] != 2:
        raise ValueError(
            f"q_fracs must have shape (num_q, 2); got {q_fracs_arr.shape}."
        )
    chi_zz_list: list[float] = []
    chi_pm_list: list[float] = []
    chi_mp_list: list[float] = []
    q_cart_list: list[tuple[float, float]] = []
    for q_frac in q_fracs_arr:
        q_cart = q_frac[0] * RECIPROCAL_B1 + q_frac[1] * RECIPROCAL_B2
        chi_zz, chi_pm, chi_mp = _chi0_at_q(
            q_cart,
            params_alpha=params_alpha,
            mu=float(mu),
            chi0_grid_nk=int(chi0_grid_nk),
            twist_grid=int(twist_grid),
        )
        chi_zz_list.append(chi_zz)
        chi_pm_list.append(chi_pm)
        chi_mp_list.append(chi_mp)
        q_cart_list.append((float(q_cart[0]), float(q_cart[1])))

    chi_zz_arr = np.asarray(chi_zz_list, dtype=float)
    chi_pm_arr = np.asarray(chi_pm_list, dtype=float)
    chi_mp_arr = np.asarray(chi_mp_list, dtype=float)
    # Real transverse Stoner channel: average of the +- and -+ bubbles.
    # Under uniform Zeeman the system retains rotational symmetry about
    # z, so chi_xx = chi_yy = (chi_pm + chi_mp) / 4 and the natural
    # *single-axis* Stoner condition uses this. With the existing
    # convention that chi_zz collapses to scalar Lindhard at alpha=h_z=0,
    # we expose chi_transverse = (chi_pm + chi_mp) / 2 so that BOTH
    # I_c_longitudinal and I_c_transverse collapse to the parent's
    # scalar I_c at the alpha=h_z=0 limit (where chi_pm = chi_mp =
    # chi_zz_collapsed).
    chi_tr_arr = 0.5 * (chi_pm_arr + chi_mp_arr)

    idx_zz = int(np.argmax(chi_zz_arr))
    idx_pm = int(np.argmax(chi_pm_arr))
    idx_mp = int(np.argmax(chi_mp_arr))
    idx_tr = int(np.argmax(chi_tr_arr))
    chi_zz_max = float(chi_zz_arr[idx_zz])
    chi_pm_max = float(chi_pm_arr[idx_pm])
    chi_mp_max = float(chi_mp_arr[idx_mp])
    chi_tr_max = float(chi_tr_arr[idx_tr])
    Ic_zz = (1.0 / (2.0 * chi_zz_max)) if chi_zz_max > 0.0 else float("inf")
    Ic_pm = (1.0 / (2.0 * chi_pm_max)) if chi_pm_max > 0.0 else float("inf")
    Ic_mp = (1.0 / (2.0 * chi_mp_max)) if chi_mp_max > 0.0 else float("inf")
    Ic_tr = (1.0 / (2.0 * chi_tr_max)) if chi_tr_max > 0.0 else float("inf")
    return {
        "alpha": float(alpha),
        "h_z": float(h_z),
        "mu": float(mu),
        "chi0_grid_nk": int(chi0_grid_nk),
        "twist_grid": int(twist_grid),
        "q_fracs": [list(map(float, q)) for q in q_fracs_arr],
        "q_fracs_cartesian": q_cart_list,
        "chi_zz": [float(x) for x in chi_zz_list],
        "chi_pm": [float(x) for x in chi_pm_list],
        "chi_mp": [float(x) for x in chi_mp_list],
        "chi_transverse": [float(x) for x in chi_tr_arr],
        "chi_zz_max": chi_zz_max,
        "chi_pm_max": chi_pm_max,
        "chi_mp_max": chi_mp_max,
        "chi_transverse_max": chi_tr_max,
        "q_max_zz": list(map(float, q_fracs_arr[idx_zz])),
        "q_max_pm": list(map(float, q_fracs_arr[idx_pm])),
        "q_max_mp": list(map(float, q_fracs_arr[idx_mp])),
        "q_max_transverse": list(map(float, q_fracs_arr[idx_tr])),
        # Backward-compatible: I_c_transverse now uses the rotation-
        # symmetric (chi_pm + chi_mp)/2 channel. At alpha=h_z=0 it
        # reduces to the chi_pm-only value (equal collapse). At
        # finite Zeeman it is the physically-correct in-plane Stoner.
        "I_c_longitudinal": float(Ic_zz),
        "I_c_transverse": float(Ic_tr),
        # Diagnostic-only Stoner Ic from each circular channel alone:
        "I_c_pm_only": float(Ic_pm),
        "I_c_mp_only": float(Ic_mp),
        "leading_channel": (
            "transverse" if Ic_tr < Ic_zz else "longitudinal"
        ),
    }


__all__ = ["channel_resolved_stoner", "mu_for_target_filling"]
