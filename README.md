# hubbard-meanfield

**A self-consistent Hartree-Fock solver for magnetic textures in the
triangular-lattice Hubbard model with Rashba spin-orbit coupling.**

[![tests](https://github.com/wenqin-chen/hubbard-meanfield/actions/workflows/tests.yml/badge.svg)](https://github.com/wenqin-chen/hubbard-meanfield/actions/workflows/tests.yml)

Pure NumPy research code for deciding which magnetic order an interacting electron
system chooses: a library of competing texture seeds (ferromagnet, spirals, single-Q /
double-Q / triple-Q skyrmion crystals), restricted fixed-texture scans, full
unrestricted self-consistent mean field, and topology diagnostics for converged
states (Berg-Luscher skyrmion number, triple-Q detection). Author: [Wenqin Chen](https://wenqin-chen.github.io/).

## What's in the box

| Module | What it does |
|---|---|
| `hubbard_meanfield.py`, `hubbard_unrestricted_meanfield.py` | supercell Hamiltonian + unrestricted SCF loop (linear and Pulay/DIIS mixing) |
| `hubbard_nesting.py` | susceptibility / nesting diagnostics |
| `texture_seeds_skx.py` | canted triple-Q skyrmion-crystal seeds (Bloch / Neel / tetrahedral / coplanar presets + general mode-spinor ansatz) |
| `restricted_texture_scan.py` | restricted fixed-texture mean-field scans (one-shot diagonalization ranking) |
| `stationarity_check.py` | verifies restricted-scan winners are true SCF stationary points |
| `stoner_channel_resolved.py` | channel-resolved Stoner pre-flight (longitudinal vs transverse instabilities under SOC) |
| `triad_detector.py` | triple-Q detection + normalized skyrmion number |
| `dm_diagnostics.py`, `rashba_band_validator.py` | Dzyaloshinskii-Moriya regime diagnostic; analytic U=0 Rashba band validation |
| `texture_plotter.py` | publication-style texture figures (real-space quiver, moment maps, summary panels) |
| `cell_params.py` | explicit-kwarg parameter builder |

## Scope

This repository is a curated research code sample: the solver core and methodology
modules of a larger private research codebase. Campaign drivers, result datasets, and
components tied to unpublished results are deliberately omitted until the accompanying
paper is published.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build a triple-Q skyrmion-crystal seed and compute its topological charge
python -c "
from hubbard_meanfield import MagneticSupercell, berg_luescher_skyrmion_number
from texture_seeds_skx import canted_tetrahedral_skx
cell = MagneticSupercell(9, 9)
texture = canted_tetrahedral_skx(cell, A=0.0894, S_z0=0.0, p=2)
bl = berg_luescher_skyrmion_number(texture, cell)
print('Berg-Luscher skyrmion number:', bl['number'])"

# test suite (CPU, ~1-2 min)
pytest -q
```

## Validation

The test suite (77 tests, plain `pytest` from the repo root) is oracle-driven:

- the U=0 band structure is validated against the **analytic Rashba dispersion**
- seed textures are checked against their **exact topological charges** (the
  Berg-Luscher number of an analytic triple-Q texture is known mathematics)
- restricted one-shot scans are cross-checked against the full SCF at matched
  amplitudes; converged winners must pass an independent **stationarity check**

## License

MIT
