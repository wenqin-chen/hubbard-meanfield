"""Visualization helpers for SCF-converged spin textures on the
triangular Hubbard + Rashba magnetic supercell.

Three plot types per texture:

1. ``plot_real_space_texture``: quiver of (S_x, S_y) colored by S_z on
   the Cartesian triangular lattice. Standard SkX visualization.
2. ``plot_moment_magnitudes``: per-site |S_i| as a scatter heatmap.
   Reveals zero-moment cores (e.g., canted_bloch's winding-2 vortex).
   on the magnetic BZ for α ∈ {x, y, z, xy}. Reveals the multi-Q
   triad structure.

All functions return the matplotlib Axes for further customization
and can optionally save to a path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from numpy.typing import NDArray

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hubbard_meanfield import MagneticSupercell  # noqa: E402
from hubbard_nesting import RECIPROCAL_B1, RECIPROCAL_B2  # noqa: E402



def plot_real_space_texture(
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    arrow_target_length: float = 0.7,
    cmap: str = "RdBu_r",
    sz_lim: float | None = None,
    dot_size: float = 60.0,
) -> plt.Axes:
    """Quiver of (S_x, S_y) colored by S_z on the triangular Cartesian lattice.

    arrow_target_length: the longest arrow draws ~this many lattice
    constants; defaults to 0.7 so arrows fit within site separation
    without overlapping.
    """
    arr = np.asarray(texture, dtype=float)
    positions = supercell.site_positions_cartesian()  # (N, 2)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    sz = arr[:, 2]
    if sz_lim is None:
        sz_lim = float(np.max(np.abs(sz))) or 1.0
    norm = TwoSlopeNorm(vmin=-sz_lim, vcenter=0.0, vmax=sz_lim)
    # Site dots colored by S_z, sized by |S|
    site_norms = np.linalg.norm(arr, axis=1)
    ax.scatter(
        positions[:, 0], positions[:, 1],
        c=sz, cmap=cmap, norm=norm,
        s=dot_size,
        edgecolors="black", linewidths=0.4, zorder=2,
    )
    # In-plane arrows on top, scaled so the longest visible arrow is
    # ~arrow_target_length lattice constants
    inplane_max = float(np.max(np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)))
    if inplane_max > 1.0e-9:
        scale_factor = inplane_max / float(arrow_target_length)
    else:
        scale_factor = 1.0
    ax.quiver(
        positions[:, 0], positions[:, 1],
        arr[:, 0], arr[:, 1],
        angles="xy", scale_units="xy",
        scale=scale_factor,
        pivot="middle",
        color="black", width=0.008,
        headwidth=4.0, headlength=5.0, headaxislength=4.5,
        alpha=0.9, zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$  (lattice units)")
    ax.set_ylabel(r"$y$  (lattice units)")
    if title:
        ax.set_title(title)
    # Colorbar for S_z
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(r"$S_z$")
    return ax


def plot_moment_magnitudes(
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Per-site |S_i| as a scatter heatmap, with zero-moment cores
    visibly marked."""
    arr = np.asarray(texture, dtype=float)
    positions = supercell.site_positions_cartesian()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    norms = np.linalg.norm(arr, axis=1)
    sc = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=norms, cmap=cmap, s=120,
        edgecolors="black", linewidths=0.4,
        vmin=0.0, vmax=max(norms.max(), 1e-9),
    )
    # Mark cores (|S_i| below 1e-3) with red ring
    core_mask = norms < 1.0e-3
    if np.any(core_mask):
        ax.scatter(
            positions[core_mask, 0], positions[core_mask, 1],
            facecolors="none", edgecolors="red",
            s=320, linewidths=2.0, label=f"zero-moment core ({core_mask.sum()})",
        )
        ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$  (lattice units)")
    ax.set_ylabel(r"$y$  (lattice units)")
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(r"$|\mathbf{S}_i|$")
    return ax




def make_summary_panel(
    texture: NDArray[np.float64],
    supercell: MagneticSupercell,
    *,
    label: str,
    F_scf: float,
    BL: float | None = None,
    save_path: Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """2-panel figure: real-space + |S_i| for a single texture."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_real_space_texture(
        texture, supercell,
        ax=axes[0],
        title=f"Real-space spin texture",
    )
    plot_moment_magnitudes(
        texture, supercell,
        ax=axes[1],
        title=r"Per-site $|\mathbf{S}_i|$",
    )
    bl_str = (f", $|BL|$ = {int(round(BL))}" if BL is not None else "")
    fig.suptitle(
        f"{label}: F = {F_scf:.6f}{bl_str}",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def make_comparison_panel(
    entries: list[dict],
    *,
    save_path: Path | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Side-by-side real-space texture comparison.

    entries: list of dicts with keys {label, texture, supercell, F, BL}.
    """
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, entry in zip(axes, entries):
        plot_real_space_texture(
            entry["texture"], entry["supercell"],
            ax=ax,
            title=f"{entry['label']}\n$F$ = {entry['F']:.5f}, "
                  f"$|BL|$ = {entry.get('BL', 'undef')}",
        )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


__all__ = [
    "plot_real_space_texture",
    "plot_moment_magnitudes",
    "make_summary_panel",
    "make_comparison_panel",
]
