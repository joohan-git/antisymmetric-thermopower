# Antisymmetric thermopower without magnetism in three-phase composites

Code and data for the manuscript

> J. Nam, *Antisymmetric thermopower without magnetism in three-phase composites* (submitted).

The code computes the effective thermoelectric tensors of periodic composites of isotropic phases and the antisymmetric (axial) part of the effective Seebeck tensor, S_a = (S_xy − S_yx)/2. It contains finite-volume and finite-element homogenization in two and three dimensions, certified two-sided variational bounds, the colour-group symmetry test, and the calculations behind every number in the manuscript. The result of each calculation is stored as JSON next to the script that produced it.

## Quick start

```bash
python -m pip install -r requirements-lock.txt   # exact tested versions (Python 3.12)
python smoke_test.py
```

`smoke_test.py` runs in under a second. It checks that a two-phase p4 cell gives S_a = 0, a three-phase p4 cell gives S_a ≠ 0, its mirror image reverses the sign, a mirror-symmetric (p4mm) cell gives zero, that the same holds for a three-dimensional 4₁ helix, and that the closed form of the weak-contrast cubic term equals the general contraction (`analytic/c3.py`) and vanishes or reverses sign in the same symmetry cases.

`requirements-lock.txt` lists the exact versions of a clean environment in which `smoke_test.py` passed; `requirements.txt` gives looser ranges and also installs `python-flint`. Tested with Python 3.12, NumPy 2.5, SciPy 1.18, PyAMG 5.2.1 and gemmi 0.7.5 on Windows 11. `threadpoolctl` is optional; if present, it limits BLAS threads in the 3D runs. `python-flint` 0.9.0 (Arb ball arithmetic) is needed only to rerun the verification `certified/r16_verify.py` and `r16_audit.py`; install it with `python -m pip install python-flint==0.9.0`.

## Layout

| Folder | Content |
|---|---|
| `solvers/fv2d/` | Two-dimensional periodic finite-volume homogenization (`homog2.py`) with the matrix harmonic mean on faces between phases; the reference pinwheel cell (`pinwheel.py`); exact face-wise decomposition of the axial coefficient (`axial_density.py`, `axial_share.py`); material condition D (`dcond_sweep.py`). |
| `solvers/design2d/` | Continuous p4 geometry families (`families.py`), materials (`mats.py`), design searches by differential evolution (`bound_de*.py`, `f3_*.py`) and parameter sweeps (`f2_*.py`). |
| `solvers/fv3d/` | Three-dimensional finite volumes, direct (`homog3d.py`) and conjugate gradients with smoothed-aggregation AMG (`homog3d_it.py`); woodpiles (`woodpile.py`); colour groups (`coloring.py`); RCSR nets from the NODE and EDGE_CENTER records (`rcsr_net.py`) and their coloured voxel structures (`rcsr_color.py`); scans and optimization; the RCSR selection 3,749 → 862 → 231 → 126 → 116 (`rcsr_filter.py`), the 116 structures with voxel and continuous colour point groups (`run_rcsr_all.py`, `rcsr_colour_pg.py`) and the aggregation of the 215-structure atlas (`atlas3d.py`), with the colour groups of the straight and segmented woodpiles (`woodpile_groups.py`, `segwood_groups.py`) and the recoloured nets (`mirror_colour.py`). |
| `certified/` | Independent nodal finite elements (`r5_fem.py`, imported by most later stages) and the laminate counterexample (`r5_laminates.py`), certified continuum intervals, annuli, winding circuits, finite samples, fabrication tolerance and interface resistance, low-symmetry cells and the weak-contrast Taylor coefficients (`r15`), orthogonal-gradient measurement and ball-arithmetic verification (`r16`), insulating-matrix structures (`r17`). Files are grouped by analysis stage (`r5`–`r17`). The trial fields of the `r16` certificates are stored in `r16_trials_*.npz` (about 55 MB in total) for re-verification. |
| `analytic/` | Weak-contrast expansion of the effective transport matrix: the cubic term and its geometry factor (`c3.py`, `c3_checks.py`), and an independent Fourier-space verification with brute-force algebra checks (`c3_fourier_verify.py`, `c3_fourier_algebra.py`, `c3_fourier_checks.py`, `c3_fourier_summary.py`). |

## Where each result comes from

Figure numbers refer to the current manuscript. Each stored result is a JSON file in the same folder as its script, except where a folder is given.

| Manuscript item | Script(s) | Stored result |
|---|---|---|
| Fig. 1a, concept | none (no computed data) | none |
| Fig. 1b, woodpile W2 and its mirror image (drawn before junction separation) | `solvers/fv3d/opt_segwood.py`; `certified/r12_run.py` (geometry) | `solvers/fv3d/opt_segwood.json`, `certified/r12_geometries.json` (`cases.Wopt`) |
| Fig. 2, symmetry test: two-phase, three-phase, mirror-image and p4mm values | Finite-volume zero controls (two-phase and p4mm): `solvers/fv2d/homog2.py` with `pinwheel.py`, also replayed by `smoke_test.py`; three-phase and mirror-image values: `certified/r6_validate.py`, `r6_sector_runs.py` (corner-enriched upper-trial values) | `certified/r6_sector_validation.json`, `r6_summary.json`; `analytic/fig2_values.json` is a hand-collected table of these four values and the interval |
| Fig. 2, certified interval of the pinwheel cell, corner exponents, checkerboard test (SI Note 4) | `certified/r6_validate.py`, `r6_sector.py`, `r6_sector_runs.py`, `r6_singular.py`, `r6_report.py` | `r6_validation.json`, `r6_sector_validation.json`, `r6_analysis.json`, `r6_summary.json` |
| Fig. 3a–c, mechanism in design A: local axial density and its 86% share (128 × 128 raster) | `solvers/fv2d/axial_density.py`, `axial_share.py` | `solvers/fv2d/axial_share.json` (shares; the field maps of Fig. 3b,c are regenerated by `axial_density.py` in seconds) |
| Fig. 3d, rotating squares | `solvers/design2d/f2_sweep.py` | `f2_sweep.json` |
| Fig. 3e, matrix sweep | `solvers/design2d/f2_conv_matrix.py` | `f2_conv_matrix.json` |
| Fig. 3f, certified designs A–D (SI certified table) | `certified/r6_optimize.py`, `r6_followup.py`, `r6_four_phase.py`, `r7.py`, `r7_sensitivity.py` | `r6_designs.json`, `r6_four_phase.json`, `r7_results.json` (half-span-normalized values: `r6_comparison.*.metrics.span_normalized_abs_Sa`; the `normalized_*` fields of `r6_summary.json` divide by the largest constituent \|S\| instead) |
| Idealized materials (SI table) | `certified/r6_followup.py` | `r6_ideal.json` |
| Material condition D (SI figure) | `solvers/fv2d/dcond_sweep.py` | `solvers/fv2d/dcond_sweep.json` |
| Fig. 4, annulus current and field; mirror-symmetric annulus null control (main text) and finite-ring controls | `certified/r5_ring.py`, `r5_additional_checks.py`; closed forms and certified homogeneous value in `r13_methods.py` | `r5_ring_results.json`, `r5_ring_controls.json` (`mirror_symmetric_8`, `_16`), `r5_ring_two_phase_merged.json`, `r5_ring_boundary_refinement.json`, `r13_methods.json` |
| Fig. 4, winding-number voltages | `certified/r8_winding.py`, `r8_finite.py` | `r8_winding.json` |
| Winding law versus the matrix/strip thermal-conductivity ratio (SI Note 8) | `certified/r8_winding_ratio.py` | `certified/r8_winding_ratio.json` |
| Tolerance, interface resistance, finite samples (SI tables; ensemble means, power-factor and probe ratios in `r8_summary.json`) | `certified/r8_tolerance.py`, `r8_tolerance_audit.py`, `r8_manufacturing.py`, `r8_contact.py`, `r8_electrodes.py`, `r8_summary.py` | `r8_*.json` |
| Boundary currents of finite rings, including two-phase p4 rings (SI Note 6) | `certified/r8_ring.py` | `r8_ring.json` |
| Interface resistance in the two-phase p4 pinwheel: a resistive interface acts as a third transport matrix (main text, SI Note 8 table) | `certified/r8_contact_two_phase.py` | `r8_contact_two_phase.json` |
| Three phases with equal σκ: certified S_a contains zero (SI Note 3 table) | `certified/r6_equal_sigma_kappa.py` | `r6_equal_sigma_kappa.json` |
| Third-phase sensitivity and smooth three-phase inclusions (SI checks) | `certified/r5_additional_checks.py` | `r5_third_phase_sensitivity.json`, `r5_smooth_three_phase.json` |
| Two-phase cells of lower symmetry, laminate counterexample (SI Note 2, first SI table) | `certified/r5_laminates.py`, `r15_continuum.py p1` | `r5_laminate_counterexample.json`, `r15_p1.json` |
| Fig. 5, woodpiles: certified intervals | `solvers/fv3d/opt_segwood.py`, `verify_segwood.py`, `converge3d.py`; `certified/r12_run.py`, `r13_core.py`, `r13_run.py` | `r12_periodic.json`, `r13_woodpile.json` |
| Fig. 5, woodpiles with an insulating matrix | `certified/r17_connectivity.py`, `r17_core.py`, `r17_run.py`, `r17_checks.py`, `r17_report.py` | `r17_connectivity.json`, `r17_transport.json`, `r17_checks.json`, `r17_summary.json` |
| Fig. 6, RCSR selection (3,749 → 862 → 231 → 126 → 116) | `solvers/fv3d/rcsr_filter.py` (needs the RCSR file) | `solvers/fv3d/rcsr_filter.json` |
| Fig. 6, 215-structure atlas; segmented-woodpile colour groups (46 × 4, 35 × m, 1 × 422) | `solvers/fv3d/scan_woodpile.py`, `woodpile_groups.py` (colour groups of the straight woodpiles), `segwood_groups.py`, `scan_segwood.py`, `run_rcsr_all.py`, `atlas3d.py` | `solvers/fv3d/scan_*.json`, `woodpile_groups.json`, `segwood_groups.json`, `rcsr_color_N16_all.json`, `atlas3d.json` |
| Main text and SI table, 24 nets recoloured to restore their parent symmetry (three-phase, forbidden, S_a,z below 10^-10 µV/K) | `solvers/fv3d/mirror_colour.py` (needs the RCSR file; `--cgd` or `RCSR_CGD`) | `solvers/fv3d/mirror_colour_N16.json`, `mirror_colour_run.json` |
| Extended Data Fig. 1, convergence of finite volumes and finite elements against the certified interval | `certified/r6_validate.py`, `r6_sector_runs.py` | `r6_validation.json`, `r6_sector_validation.json`, `r6_summary.json` |
| Extended Data Fig. 2, the 49 allowed nets | `solvers/fv3d/run_rcsr_all.py` | `solvers/fv3d/rcsr_color_N16_all.json` (entries with `allowed[2]` true) |
| Extended Data Fig. 3, woodpile W2 before and after junction separation | `certified/r12_run.py`, `r13_run.py` (geometry) | `certified/r12_geometries.json` (`cases.Wopt`), `certified/r13_geometries.json` (`woodpile.W2_split_clear`) |
| Extended Data Fig. 4, net bbr with the handed and the parent-symmetry colouring | `solvers/fv3d/mirror_colour.py` (colour groups and values); strut geometry from `rcsr_net.py` | `solvers/fv3d/mirror_colour_N16.json` (`bbr`), `rcsr_colour_pg.json` |
| Methods, weak-contrast expansion (cubic term, closed form, Fourier-space verification) | `analytic/c3.py`, `c3_checks.py`, `c3_fourier_verify.py`, `c3_fourier_algebra.py`, `c3_fourier_checks.py`, `c3_fourier_summary.py`; `certified/r15_weak.py` | `analytic/c3_results.json`, `c3_checks.json`, `numbers.json`, `c3_fourier_checks.json`, `fft_numbers.json`, `fv_numbers.json`, `geometry_numbers.json`, `exact_fourier_numbers.json`, `algebra_numbers.json`, `units_check.json`; `certified/r15_weak.json` (the numerical checks quoted in SI Note 7 are in `c3_fourier_checks.json`) |
| Supplementary figure, cubic onset versus full contrast along the midpoint path | `analytic/c3_sweep.py` | `analytic/c3_sweep.json` |
| Methods, orthogonal-gradient measurement | `certified/r16_measure.py`, `r16_controls.py` | `r16_measure.json`, `r16_controls.json` |
| Methods, ball-arithmetic verification | `certified/r16_verify.py`, `r16_audit.py` | `r16_verified_*.json`, `r16_trials_*.npz`, `r16_audit.json`, `r16_summary.json` |
| Methods, low-symmetry rod-and-block cells; material symmetry groups of the rasters | `certified/r15_continuum.py`, `r15_sector.py`, `r15_symmetry.py`, `r15_checks.py` | `r15_p1.json`, `r15_symmetry.json`, `r15_checks.json`, `r15_summary.json` |
| Methods, 2D insulating third phase | `certified/r15_continuum.py`, `r15_sector.py` | `r15_insulator.json` |
| SI tables: atlas by family | `solvers/fv3d/atlas3d.py` | `solvers/fv3d/atlas3d.json` |
| SI tables: all 116 nets with colour point groups | `solvers/fv3d/rcsr_colour_pg.py` (needs the RCSR file) | `solvers/fv3d/rcsr_colour_pg.json` |
| SI tables: orthogonal measurement | `certified/r16_measure.py`, `r16_report.py` | `r16_measure.json`, `r16_summary.json` |
| SI tables: insulating matrix | `certified/r17_run.py`, `r17_report.py` | `r17_summary.json` |
| Helices and stacked layers (SI table) | `solvers/fv3d/run3d_helix.py`, `run3d_checks.py`; `certified/r11_run.py`, `r11_fem.py` | `solvers/fv3d/run3d_*.json`, `certified/r11_periodic.json` |
| Net-derived structures, certified (SI tables) | `solvers/fv3d/rcsr_radius.py`; `certified/r12_run.py`, `r13_core.py`, `r13_run.py` | `solvers/fv3d/rcsr_radius.json`, `certified/r12_periodic.json`, `certified/r13_rcsr.json` |
| ukg-derived structure, whose certified intervals contain zero (SI table of all nets, note) | archived results only; the fixed ukg geometry and its generation driver are not included (the r12, r13 and r17 drivers prepare only cyc, unm and bbr) | `certified/ukg_check.json` |
| Material parameters (SI table) | `certified/r6_core.py` | `certified/r6_summary.json` (`realistic_materials`), `certified/r13_geometries.json` (`materials_sigma_S_kappa`) |

## Names and conventions

- **Designs.** Design A of the manuscript is case `A_64` in `certified/r7_results.json`; design B is `B_64`; design C (carbon phase) is `best3` in `r6_designs.json` and `r6_comparison.three` in `r7_results.json`; design D (with copper) is `best` in `r6_four_phase.json` and `r6_comparison.four` in `r7_results.json`.
- **Woodpiles.** In `certified/r12_*` the case `W2` is the manuscript's W1 (the atlas woodpile of 16-voxel cells) and `Wopt` is the manuscript's W2 (optimized, 24-voxel cells); `r13_geometries.json` (`naming`) records this. `W2_split_clear` is W2 with separated junctions (Extended Data Fig. 3).
- **Phase labels.** Three-dimensional voxel maps use 0 = weak phase (or matrix), 1 = n-type, 2 = p-type. Two-dimensional maps index the material list passed to the solver, so the meaning of a label is set by that list in each script.

## Number types

Results are labelled as one of three types, as in the manuscript:

- **C, certified continuum interval** — the effective energy matrix C of the stated fixed pixel or voxel geometry is enclosed between a conforming upper bound and a complementary lower bound in the Loewner order (L ⪯ C ⪯ U), and S_a, S_xx and σ_eff are optimized over that set. The bounds are variational; they are evaluated in floating point and are not interval-arithmetic proofs.
- **P, point estimate** — a single finite-element or finite-volume value on a uniform grid.
- **V, ball-arithmetic verified interval** — the continuum enclosure of S_a for a fixed raster is proved with interval (Arb) arithmetic via python-flint, using exact decimal material values and dyadic conforming trial fields (`certified/r16_verify.py`; `r16_audit.py` rechecks every stored certificate).

## RCSR nets

`solvers/fv3d/rcsr_net.py` reads the three-dimensional net file of the Reticular Chemistry Structure Resource (RCSR, http://rcsr.net; M. O'Keeffe, M. A. Peskov, S. J. Ramsden and O. M. Yaghi, *Acc. Chem. Res.* **41**, 1782 (2008)). The file is not redistributed here. Download the 3D nets in Systre (`.cgd`) format from the RCSR website and either place it at `data/rcsr3d_web.cgd` or set the environment variable `RCSR_CGD` to its path. The stored results do not need this file; it is required only to regenerate the net-derived structures and the recoloured nets.

## Running the analyses

The stored JSON files contain every number in the manuscript. Rerunning takes from seconds (two-dimensional finite volumes) to several hours (certified corner-enriched bounds and 3D refinements at k = 3) on a laptop CPU. Run the scripts from their own folder. Scripts in `certified/` read the results of earlier stages in the same folder; run them in numerical order (`r5` → `r17`). `r16_verify.py` additionally needs `python-flint`. The transport stages skip refinement levels that are already stored in their output JSON, and `r17_run.py` checks that the voxel labels in `r13_geometries.json` are those its stored results were computed from. To recompute a stage rather than reuse stored levels, run it in a copy of the folder from which its output JSON (and those of the later stages) have been removed.

Commands of the staged scripts (run from `certified/`, in this order within each stage):

```
python r5_fem.py            python r5_ring.py          python r5_laminates.py
python r6_validate.py       python r6_sector_runs.py   python r6_report.py      python r6_optimize.py   python r6_followup.py   python r6_four_phase.py
python r7.py                python r7_sensitivity.py       python r6_equal_sigma_kappa.py
python r8_tolerance.py      python r8_tolerance_audit.py   python r8_manufacturing.py   python r8_contact.py   python r8_contact_two_phase.py   python r8_electrodes.py
python r8_winding.py        python r8_winding_ratio.py     python r8_ring.py            python r8_summary.py
python r11_run.py validate|periodic|extrusion|finite|finite_controls [--cases ...] [--levels ...]
python r12_run.py prepare ; python r12_run.py run [--cases ...] [--levels ...] ; python r12_run.py run --mirror ; python r12_run.py sensitivity
python r13_run.py prepare ; python r13_run.py validate ; python r13_run.py woodpile ; python r13_run.py rcsr ; python r13_run.py clearance
python r15_continuum.py p1|insulator [--m ...] [--kappa ...]
python r16_verify.py 2d|3d --k ...  (needs python-flint) ; python r16_measure.py ; python r16_controls.py ; python r16_audit.py ; python r16_report.py
python r17_run.py [--k ...] [--cases ...] [--kappa ...]
```

Steps that prepare the net-derived structures (`r12_run.py prepare`, `r13_run.py prepare`, the RCSR scripts in `solvers/fv3d/`) need the RCSR file; without it they skip the nets and the woodpile stages still run. 

Known limitations: the stored three-dimensional finite-volume values come from runs before the algebraic-multigrid setup in `homog3d_it.py` was seeded, so a rerun reproduces them to the solver tolerance rather than bit for bit; `homog3d_it.py` now stops if a conjugate-gradient solve does not converge. The design searches (`solvers/design2d/bound_de*.py`) score a candidate whose evaluation fails as zero, so they report the best successful candidate only.

Several scripts record SHA-256 hashes of the files they read, keyed by paths relative to this folder; the hash of the external RCSR file is recorded under its name. The records in the stored JSON files have been updated to the files exactly as released, so they agree with one another and with a fresh checkout; a rerun writes the same kind of records. The net-derived structures were rebuilt with `solvers/fv3d/rcsr_net.py`; the entries for `cyc` in `certified/r12_periodic.json` and `r17_transport.json` were computed then, and their `provenance` field says so. The summary scripts `r15_report.py`, `r16_report.py` and `r17_report.py` write only the compact JSON summaries.

## Citation

Please cite the article and this archive (see `CITATION.cff`). Archive DOI: [10.5281/zenodo.23668925](https://doi.org/10.5281/zenodo.23668925).

## License

MIT License (see `LICENSE`).
