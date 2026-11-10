# Antisymmetric thermopower without magnetism in three-phase composites

Code and numerical data accompanying J. Nam, *Antisymmetric thermopower without magnetism in three-phase composites*. The computations are organized by physical topic.

## Contents

| Folder | Computations |
|---|---|
| `common/` | Coupled thermoelectric finite-volume and finite-element solvers, material parameters, and shared geometry and enclosure helpers. |
| `planar_cells/` | Pinwheel and design-family cells, axial-density and phase-share decompositions, material sweeps, and searches for enhanced thermopower. |
| `certified_bounds/` | Corner-enriched primal/dual bounds, certified planar designs A–D, high-contrast and insulating phases, and ball-arithmetic certificates with stored integer trial fields. |
| `weak_contrast/` | The cubic term, its geometry and material factors, Fourier-space checks, and comparison with full-contrast solutions. |
| `three_d/` | Layered cells and helices, straight and segmented woodpiles, junction separation, RCSR nets, colour groups, recolouring, and the structure atlas. |
| `devices/` | Rings and winding circuits, finite samples and electrodes, interface resistance, fabrication tolerance, and orthogonal-gradient measurement. |
| `validation/` | Two-phase cancellation, equal-conductivity controls, convergence, variational identities, and independent solver comparisons. |

JSON files store the computed results. The seven `certified_bounds/trials_*.npz` archives store the integer fields needed to recheck the rigorous certificates. Phase labels are defined by each calculation's material list. In the planar design data, A and B are `cases.A_64` and `cases.B_64` in `certified_bounds/design_enclosures.json`; C and D are `material_design_comparison.three` and `.four`. In `three_d/fixed_rasters.json`, `W2` denotes manuscript W1 and `Wopt` denotes manuscript W2; `three_d/junction_geometry.json` records the correspondence.

Results marked **P** are mesh-dependent point estimates. **C** denotes a variational continuum enclosure for the specified fixed raster, evaluated in floating point. **V** denotes an enclosure verified by ball arithmetic using exact decimal material inputs and admissible dyadic trial fields.

## Setup

Use Python 3.12 and install the dependencies from the repository root:

```sh
python -m pip install -r requirements-lock.txt
```

`requirements.txt` provides broader compatible versions. NumPy and SciPy provide the numerical solvers; PyAMG accelerates the three-dimensional solves, gemmi supplies crystallographic operations, and python-flint supplies Arb ball arithmetic. `threadpoolctl`, if available, limits BLAS threads in the larger computations.

## Smoke test

```sh
python smoke_test.py
```

This checks two-phase cancellation, nonzero three-phase thermopower, reversal under reflection, mirror-symmetric zero controls, a three-dimensional helix, and agreement of the closed-form cubic coefficient with its tensor contraction.

## Examples

Run these commands from the repository root:

```sh
python planar_cells/check3.py
python weak_contrast/c3.py
python three_d/atlas3d.py
python certified_bounds/certificate_checks.py
```

Each example takes seconds. The certificate check replays all seven stored proofs without solving the transport equations. The atlas command combines the stored woodpile and net results. Scripts can also be invoked by absolute path. Outputs are written under the release root and replace the stored result files with recomputed ones, so run the commands in a separate copy to keep the archived files untouched.

## Long runs

Material optimization, corner enrichment, three-dimensional refinement, and net ensembles can take minutes to hours. Calculations with stored refinement levels resume from their JSON files. To recompute those levels, use a separate copy and remove only the output being regenerated, retaining its input results. The trial-field certificate generator likewise reuses existing certificates.

The main dependencies are: planar geometry and material searches → planar enclosures; fixed three-dimensional rasters → junction geometry → transport and insulating substitutions; individual results → compact summaries. Geometry preparation uses `three_d/fixed_rasters.py prepare`, followed by **both** `three_d/junction_geometry.py prepare` and `three_d/junction_geometry.py clearance` before transport. The clearance step produces the W2 geometries used in the manuscript. The respective transport commands are `fixed_rasters.py run`, `junction_geometry.py woodpile`, and `junction_geometry.py rcsr`; their `--cases` and `--levels` options select subsets. See each command's `--help`. `certified_bounds/ball_verification.py 2d --k 1` and `3d --k 1` generate new certificates when the corresponding output is absent.

For the complete orthogonal-measurement archive, run `devices/orthogonal_measurement.py --shifted`, then `devices/probe_controls.py`, then repeat the measurement command. The final call applies the refined bulk reference to the stored finite-sample voltages and their ratios. The `--shifted` option includes the additional boundary terminations.

The stored F1 material-search candidate is reproduced by `python planar_cells/bound_de.py F1 40 17`; the final argument is the number of optimization generations. Continuing the search can improve the candidate and therefore change the output. The constrained counterpart is reproduced by `python planar_cells/bound_de_zt1.py F1 40`. Each search archive stores the complete material and geometry parameters needed to re-evaluate its reported ratio.

Three-dimensional finite-volume results use algebraic-multigrid preconditioning; recomputation agrees to solver accuracy rather than necessarily bit for bit. Optimization searches retain the best successfully evaluated candidate. The auxiliary `ukg` calculation can be regenerated with `python three_d/ukg_check.py --recompute --levels 1 2 3`; this requires RCSR input and verifies the reconstructed geometry against its archived voxel-label hash before calling the shared solvers. Retain `three_d/ukg_check.json`, which supplies its material inputs and voxel spacings.

## RCSR input

The Reticular Chemistry Structure Resource (RCSR, <https://rcsr.net>; O'Keeffe et al., *Acc. Chem. Res.* **41**, 1782, 2008) supplies the external three-dimensional net file. It is not redistributed. To regenerate nets or recolourings, obtain the Systre `.cgd` file and set `RCSR_CGD` to its location, or place it at `three_d/rcsr3d_web.cgd`. Stored results, the smoke test, and the examples above do not require this file.

## Citation

Please cite the article and the software archive using `CITATION.cff`. Archive DOI: https://doi.org/10.5281/zenodo.23738657 (version 1.1.0).

MIT License; see `LICENSE`.
