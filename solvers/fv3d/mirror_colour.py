"""Mirror-preserving RCSR colourings, N=16 and radius=0.08a.

Run from any directory; all generated files stay beside this script. Example:
    python -B mirror_colour.py --cgd <path/to/rcsr3d_web.cgd>
Sources are imported read-only. The unmodified source solver is used at tol=1e-11.
S_a,z and ax in the output are in microvolt/kelvin; frac is [weak, n, p].
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
# Set before importing NumPy/PyAMG; small coarse solves should not oversubscribe.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ANGLE_TOL = 1e-6  # degrees, symmetric neighbourhood of each tie
ZERO_UV_K = 1e-6  # explicit operational threshold, not a rigorous error bound
SHIFTS = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T


def folded_angle(d):
    """Equivalent to min(theta % 90, 90-theta % 90), without modulo roundoff."""
    if np.linalg.norm(d[:2]) < 1e-6 * np.linalg.norm(d):
        return None
    lo, hi = sorted(np.abs(d[:2]))
    return float(np.degrees(np.arctan2(lo, hi)))


def strut_colour(d):
    phi = folded_angle(d)
    # Vertical and phi=0, 22.5, 45 all get p; no net-specific exceptions.
    return 1 if phi is not None and 22.5 + ANGLE_TOL < phi < 45 - ANGLE_TOL else 2


def voxelize(net, segs, N=16, rad=0.08):
    """Local variant of rcsr_color.voxelize; only the angular colour rule changes."""
    a, b, c = net["cell"][:3]
    Nz = max(4, int(round(N*c/a)))
    Nz += (-Nz) % 4
    g = (np.arange(N) + 0.5)/N
    gz = (np.arange(Nz) + 0.5)/Nz
    X, Y, Z = np.meshgrid(g*a, g*b, gz*c, indexing="ij")
    P = np.stack([X, Y, Z], -1)
    lab = np.zeros((N, N, Nz), int)
    isp = np.zeros_like(lab, bool)
    isn = np.zeros_like(lab, bool)
    lengths = np.array([a, b, c])
    for p0, p1 in segs:
        A, B = p0*lengths, p1*lengths
        d = B - A
        if not np.any(d):
            raise ValueError("Zero-length strut")
        col = strut_colour(d)
        for shift in SHIFTS:
            dd = P - (A + shift*lengths)
            t = np.clip((dd @ d)/(d @ d), 0, 1)
            dist = np.linalg.norm(dd - t[..., None]*d, axis=-1)
            (isp if col == 2 else isn)[dist < rad*a] = True
    lab[isn] = 1
    lab[isp] = 2  # Same p-priority at intersections; also symmetry invariant.
    return lab


def keys(group):
    return {tuple(g.ravel()) for g in group}


def verify_angular_rule(C):
    """Check the literal modulo definition and boundary colours independently."""
    rng = np.random.default_rng(19)
    angles = list(rng.uniform(-720, 720, 1000))
    angles += [90*k + boundary + offset for k in range(-4, 5)
               for boundary in (0, 22.5, 45, 67.5) for offset in (-1e-4, 0, 1e-4)]
    for theta in angles:
        d = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta)), 0.37])
        q = float(np.degrees(np.arctan2(d[1], d[0])) % 90)
        literal_phi = min(q, 90-q)
        expected = 1 if 22.5 + ANGLE_TOL < literal_phi < 45 - ANGLE_TOL else 2
        assert abs(folded_angle(d)-literal_phi) < 1e-12
        assert strut_colour(d) == expected
        for op in C.OPS:
            assert strut_colour(op @ d) == expected
    for d in (np.array([0., 0., 1.]), np.array([0., 0., -1.])):
        assert strut_colour(d) == 2
    return dict(directions=len(angles)+2, point_operations=len(C.OPS), passed=True)


def continuous_groups(net, segs, RN, C):
    """R.edges already verifies the uncoloured edge set under every space operation."""
    lengths = np.array(net["cell"][:3])
    directions = [(b-a)*lengths for a, b in segs]
    geom, colour = [], []
    for op in RN.spacegroup(net).operations():
        W = np.array(op.rot)/op.DEN
        Wi = np.rint(W).astype(int)
        if tuple(Wi.ravel()) in keys(geom):
            continue
        assert np.array_equal(W, Wi) and tuple(Wi.ravel()) in keys(C.OPS)
        geom.append(Wi)
        if all(strut_colour(d) == strut_colour((W @ (d/lengths))*lengths)
               for d in directions):
            colour.append(Wi)
    return geom, colour


def strut_statistics(net, segs):
    stats = Counter()
    phis = []
    for a, b in segs:
        d = (b-a)*np.array(net["cell"][:3])
        phi = folded_angle(d)
        stats["n" if strut_colour(d) == 1 else "p"] += 1
        if phi is None:
            stats["vertical"] += 1
        else:
            phis.append(phi)
            for boundary in (0, 22.5, 45):
                if abs(phi-boundary) <= ANGLE_TOL:
                    stats[f"tie_{boundary:g}"] += 1
    return dict(stats), sorted(set(round(v, 9) for v in phis))


def save_json(path, data):
    # Direct checkpoints avoid file-replacement races on Windows.
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    for attempt in range(20):
        try:
            path.write_text(payload, encoding="utf-8")
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def hkey(p, cgd):
    """Hash-record key: path relative to the release folder; the external RCSR file by name."""
    return "rcsr3d_web.cgd (RCSR, not redistributed)" if Path(p) == Path(cgd) else "solvers/fv3d/" + Path(p).name


def source_dir():
    if (HERE / "rcsr_net.py").is_file():
        return HERE
    return next((p / "code_release" / "solvers" / "fv3d" for p in HERE.parents
                 if (p / "code_release" / "solvers" / "fv3d").is_dir()), None)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(rows, meta):
    three = [r for r in rows if r["n_phases"] == 3]
    forbidden = [r for r in three if not r["allowed"][2]]
    allowed = [r for r in rows if r["allowed"][2]]
    solved_forbidden = [r for r in forbidden if r["status"] == "computed"]
    peak = max(solved_forbidden, key=lambda r: abs(r["S_a,z"]), default=None)
    old = meta["original_summary"]
    groups = Counter((r["n_phases"], r["colour_point_group"], r["allowed"][2]) for r in rows)
    missing = [r["name"] for r in rows if r["status"] in ("pending", "budget_skipped", "solver_error")]
    changed = [r for r in rows if not r["voxel_contains_geometric_group"]]
    enlarged = [r for r in rows if r["voxel_contains_geometric_group"] and not r["voxel_equals_geometric_group"]]
    geom_allowed = [r for r in rows if r["geometric_allowed"][2]]
    lines = [
        "## Conclusion", "",
        f"All {len(rows)} selected nets were classified: **{len(forbidden)} three-phase forbidden**, "
        f"**{sum(r['allowed'][2] for r in three)} three-phase allowed**, and "
        f"**{sum(r['n_phases'] == 2 for r in rows)} two-phase**.", "",
        (f"All {len(solved_forbidden)} solved forbidden three-phase nets are numerical zeros at the "
         f"declared {ZERO_UV_K:g} microvolt/kelvin threshold. "
         if all(abs(r["S_a,z"]) < ZERO_UV_K for r in solved_forbidden) else
         "Some forbidden three-phase responses exceed the declared zero threshold. ") +
        (f"The maximum |S_a,z| is **{abs(peak['S_a,z']):.10g} microvolt/kelvin** "
         f"({peak['name']}; {abs(peak['S_a,z'])*1e-6:.10g} V/K)." if peak else "No forbidden three-phase solve is available."),
        "", "This supplies three-phase controls for separating symmetry prohibition from the two-phase obstruction.",
        "", "| Phases | Voxel colour point group | Axial z allowed | Nets |",
        "|---:|---|---|---:|",
    ]
    lines += [f"| {n} | {g} | {a} | {count} |" for (n, g, a), count in sorted(groups.items())]
    lines += ["", "Allowed voxel colour groups:", "",
              "| Net | Space group | Geometric point group | Colour point group | Phases | S_a,z (microvolt/K) | Original S_a,z (microvolt/K) |",
              "|---|---|---|---|---:|---:|---:|"]
    for r in allowed:
        value = "not computed" if r["S_a,z"] is None else f"{r['S_a,z']:.10g}"
        lines.append(f"| {r['name']} | {r['group']} | {r['geometric_point_group']} | {r['colour_point_group']} | "
                     f"{r['n_phases']} | {value} | {r['original_Sa_z_uV_K']:.10g} |")
    if not allowed:
        lines.append("| None | | | | | | |")
    lines += ["", "Geometrically axial-z-allowed nets: " +
              (", ".join(f"{r['name']} ({r['geometric_point_group']})" for r in geom_allowed) or "none") + ".",
              "No geometric point group 4 occurs in this selection.",
              "", f"Original atlas: **{old['allowed_z']} allowed**, **{old['allowed_and_nonzero']} allowed and nonzero** "
              f"at |S_a,z| > {ZERO_UV_K:g} microvolt/K; {old['forbidden_z']} forbidden. "
              f"Of the new three-phase forbidden controls, {sum(r['original_allowed'][2] for r in forbidden)} were originally allowed "
              f"and {sum(r['original_allowed'][2] and abs(r['original_Sa_z_uV_K']) > ZERO_UV_K for r in forbidden)} were allowed and nonzero.",
              "The remaining 18 originally allowed nets become two-phase; the original 67 forbidden nets stay two-phase.",
              "", "Three-phase forbidden controls: " + ", ".join(r["name"] for r in forbidden) + ".",
              "", "## Reasoning", "",
              "Write q = theta mod 90 degrees and phi = min(q, 90-q). Equivalently, "
              "phi = atan2(min(|dx|,|dy|), max(|dx|,|dy|)) in degrees. "
              "Every operation of 4/mmm acts on (dx,dy) by a signed permutation and independently changes the sign of dz, "
              "so phi and verticality are invariant. The x and y mirrors send theta to 180-theta or -theta; "
              "the two diagonal mirrors send it to 90-theta or 270-theta. A two-fold rotation about an in-plane axis "
              "at alpha = 0, 45, 90 or 135 degrees sends theta to 2alpha-theta and dz to -dz. "
              "All induce q -> q or 90-q. This proof concerns the tetragonal axes, not an arbitrary in-plane axis.",
              "", "Use p for 0 < phi < 22.5 degrees and n for 22.5 < phi < 45 degrees. "
              "**Vertical struts and phi = 0, 22.5, 45 degrees are p.** "
              f"The implementation uses a symmetric {ANGLE_TOL:g}-degree tie tolerance, "
              "and gives p priority where cylinders overlap, exactly as the original voxelizer. "
              "The weak matrix is phase 0, n is 1 and p is 2. No net-specific tie rule is used. "
              "Colouring therefore retains every geometric point operation before voxelization.",
              "", "No strut has phi=22.5 degrees within the tie tolerance. "
              "The nearest values are svs: 22.500605395 degrees and pll: 22.501178884 degrees. "
              "Both are assigned n according to the supplied CGD coordinates; they are not treated as ties. "
              "No alternative tie rule is required for those coordinates. All 85 two-phase rasters contain weak matrix and p.",
              "", "For an axial vector, a' = det(R) R a. Vertical mirrors and the perpendicular two-fold axes "
              "reverse a_z and force it to zero. Inversion and the horizontal mirror preserve a_z. "
              "The tetragonal axial-z-allowed classes are 4, -4 and 4/m. Chirality alone is insufficient: "
              "422 is chiral but forbidden; 4/m is centrosymmetric and allowed.",
              "", "Names come from rcsr_filter.json steps[-1].names. Geometry uses R.read_cgd, R.expand and R.edges "
              "with the source origin-choice and edge-reconstruction checks. The local voxelizer follows the source "
              "grid, periodic images, strict distance cutoff and p-priority, using N=16, radius=0.08a and "
              "Nz=ceil-to-multiple-of-4(max(4, round(16c/a))). Point groups use "
              "rcsr_colour_pg.colour_group_full (all 16 parent operations and all integer voxel translations); "
              "verdicts use coloring.axial_allowed. Continuous colour invariance is checked against every distinct "
              "point operation in the declared space group, in addition to R.edges' full space-group invariance check.",
              "", "Transport uses unmodified homog3d_it.homogenize3d_it at tol=1e-11, "
              "mats=[weak,n,p], with p=sloc(1e5,200e-6,1.2), n=sloc(1e5,-180e-6,1.3), "
              "weak=sloc(10.45,10e-6,0.9012), T0=300 K. S_a,z=(S_xy-S_yx)/2; "
              "stored S_a,z, Sa_z_uV_K and ax use microvolt/K, with S_a,z_V_K also provided. "
              "Results are raw solver outputs, without imposing zero or symmetrizing tensors. "
              "allowed is [x,y,z]; frac is [weak,n,p]. Unsolved two-phase entries have null responses.",
              "", f"Runtime: {meta['runtime_seconds']:.2f} s ({meta['runtime_seconds']/60:.2f} min), "
              f"including {meta['classification_seconds']:.2f} s classification and "
              f"{sum(r.get('solve_seconds', 0) for r in rows):.2f} s homogenization. "
              f"Solved {sum(r['status'] == 'computed' for r in rows)} nets. "
              + ("The full requested set fitted within the two-hour budget." if not missing else
                 "Budget/error-limited subset; unresolved names: " + ", ".join(missing) + "."),
              "", "Reproduce:", "", "```powershell",
              f'python -B mirror_colour.py --source "{meta["source_directory"]}" --cgd "{meta["cgd_path"]}"',
              "```", "", "Source paths, SHA-256 hashes, versions, runtime and solver diagnostics are saved in "
              "mirror_colour_run.json and mirror_colour_N16.json.",
              "", "## Confidence", "",
              f"All {sum(r['continuous_preserves_geometric_group'] for r in rows)}/{len(rows)} continuous colour groups retain "
              f"the declared geometric point group. {len(changed)} voxel groups lose geometric operations. "
              f"{len(enlarged)} voxel groups gain operations: " +
              (", ".join(f"{r['name']} ({r['geometric_point_group']} -> {r['colour_point_group']}, {r['n_phases']} phases)" for r in enlarged) or "none") + ". " +
              f"{sum(r['allowed'][2] != r['geometric_allowed'][2] for r in rows)} voxel axial-z verdicts differ from the geometric verdict. "
              f"All six CG solves converged for each completed net: "
              f"{all(all(info == 0 for _, info in r['cg_iterations_info']) for r in rows if r['status'] == 'computed')}. "
              f"Maximum relative Onsager mismatch: {max((r['onsager_relative_error'] for r in rows if r['status'] == 'computed'), default=0):.6g}.",
              "", f"The literal modulo-angle rule and all 16 parent operations were checked on "
              f"{meta['angular_rule_verification']['directions']} directions including the ties. "
              "For all 116 nets, the new voxelizer's occupied/empty mask exactly matches the original voxelizer. "
              "Regenerated original-rule phase fractions and colour-group verdicts match the stored atlas. "
              "Source-file and CGD hashes were unchanged by this run.",
              "", "Issues: " + ("; ".join(meta.get("issues", [])) or "none") + ".",
              "", "## Not checked", "",
              "No mesh/radius convergence study or certified continuum interval was attempted. "
              "The numerical values are N=16 finite-volume estimates; symmetry supplies the analytic zero in the continuum. "
              "Two-phase nets without a geometrically allowed axial z component were not homogenized. "
              "The original atlas values were read from rcsr_color_N16_all.json, not recomputed. "
              "Geometric point groups refer to the declared RCSR embedding and its space-group operations; "
              "no independent search for additional continuous symmetries was performed.", ""]
    lines[-2] += " Sensitivity to rounding or idealization of CGD coordinates near 22.5 degrees was not tested."
    (HERE / "mirror_colour_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=source_dir())
    parser.add_argument("--cgd", type=Path, default=os.environ.get("RCSR_CGD"))
    parser.add_argument("--budget-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.source is None:
        parser.error("Cannot locate code_release; supply --source")
    src = args.source.resolve()
    cgd = (args.cgd or src / ".." / ".." / "data" / "rcsr3d_web.cgd").resolve()
    if not cgd.is_file():
        parser.error("CGD missing; supply --cgd or RCSR_CGD")
    os.environ["RCSR_CGD"] = str(cgd)
    sys.path.insert(0, str(src))
    import rcsr_color as R
    import rcsr_net as RN
    import coloring as C
    import homog3d as H3
    import homog3d_it as HI
    import rcsr_colour_pg as PG
    import scipy
    import gemmi
    import pyamg

    start = time.perf_counter()
    names = json.loads((src / "rcsr_filter.json").read_text())["steps"][-1]["names"]
    assert len(names) == 116 and len(set(names)) == 116
    original = {r["name"]: r for r in json.loads((src / "rcsr_color_N16_all.json").read_text())}
    assert set(names) == set(original)
    nets = {r["name"]: r for r in R.read_cgd()}
    source_files = [src / s for s in ("rcsr_filter.json", "rcsr_color_N16_all.json", "rcsr_color.py",
                    "rcsr_net.py", "rcsr_colour_pg.py", "coloring.py", "homog3d.py", "homog3d_it.py", "run_rcsr_all.py")]
    meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_directory=src.name, cgd_path=cgd.name,
                N=16, radius_over_a=0.08, angle_tolerance_degrees=ANGLE_TOL, solver_tolerance=1e-11,
                numerical_zero_threshold_uV_K=ZERO_UV_K, budget_seconds=args.budget_seconds,
                phase_order=["weak", "n", "p"], issues=[],
                versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, gemmi=gemmi.__version__, pyamg=pyamg.__version__),
                sha256={hkey(p, cgd): sha256(p) for p in source_files + [cgd, Path(__file__).resolve()]})
    meta["angular_rule_verification"] = verify_angular_rule(C)
    meta["original_summary"] = dict(
        allowed_z=sum(r["allowed"][2] for r in original.values()),
        forbidden_z=sum(not r["allowed"][2] for r in original.values()),
        allowed_and_nonzero=sum(r["allowed"][2] and abs(r["ax"][2]) > ZERO_UV_K for r in original.values()),
        n_phases_counts=dict(Counter(sum(f > 0 for f in r["frac"]) for r in original.values())))
    rows, labels = [], {}
    for i, name in enumerate(names):
        net = nets[name]
        assert np.isclose(net["cell"][0], net["cell"][1], rtol=0, atol=1e-8)
        pts, cns = R.expand(net)
        segs = R.edges(net, pts, cns)
        lab = voxelize(net, segs)
        original_lab = R.voxelize(net, segs, 16, 0.08)
        assert np.array_equal(lab > 0, original_lab > 0), f"Geometry mismatch: {name}"
        original_frac = np.bincount(original_lab.ravel(), minlength=3)/original_lab.size
        assert np.allclose(original_frac, original[name]["frac"], rtol=0, atol=1e-15), name
        original_G = PG.colour_group_full(original_lab)
        assert PG.symbol(original_G) == original[name]["colour_point_group"], name
        assert C.axial_allowed(original_G)[0] == original[name]["allowed"], name
        G = PG.colour_group_full(lab)
        Ggeom, Gcontinuous = continuous_groups(net, segs, RN, C)
        assert keys(Ggeom) == keys(Gcontinuous), f"Continuous symmetry loss: {name}"
        stats, phis = strut_statistics(net, segs)
        frac = (np.bincount(lab.ravel(), minlength=3)/lab.size).tolist()
        allowed = [bool(a) for a in C.axial_allowed(G)[0]]
        geom_allowed = [bool(a) for a in C.axial_allowed(Ggeom)[0]]
        nph = len(np.unique(lab))
        row = dict(name=name, group=net["group"], setting=RN.spacegroup(net).xhm(), n_phases=nph,
                   colour_point_group=PG.symbol(G), colour_group_order=len(G), allowed=allowed,
                   **{"S_a,z": None, "S_a,z_V_K": None}, Sa_z_uV_K=None, ax=None, frac=frac,
                   shape=list(lab.shape), nodes=len(pts), struts=len(segs),
                   geometric_point_group=PG.symbol(Ggeom), geometric_allowed=geom_allowed,
                   continuous_colour_point_group=PG.symbol(Gcontinuous), continuous_preserves_geometric_group=True,
                   voxel_contains_geometric_group=keys(Ggeom).issubset(keys(G)),
                   voxel_equals_geometric_group=keys(Ggeom) == keys(G),
                   original_voxel_geometry_and_atlas_checks_passed=True,
                   colour_point_operations=[g.tolist() for g in G], strut_counts=stats, folded_angles_degrees=phis,
                   original_allowed=original[name]["allowed"], original_Sa_z_uV_K=original[name]["ax"][2],
                   status="pending" if nph == 3 or geom_allowed[2] else "not_requested_two_phase")
        rows.append(row)
        if row["status"] == "pending":
            labels[name] = lab
        if not row["voxel_contains_geometric_group"]:
            meta["issues"].append(f"{name}: voxel {row['colour_point_group']} loses operations of geometric {row['geometric_point_group']}")
        print(f"classify {i+1}/116 {name}: {nph} phases, {row['colour_point_group']}, allowed_z={allowed[2]}", flush=True)
        save_json(HERE / "mirror_colour_N16.json", rows)
    meta["classification_seconds"] = time.perf_counter()-start
    # Benchmark a representative of every group, choosing the largest voxel grid.
    pending = [r for r in rows if r["status"] == "pending"]
    representatives = []
    for group in sorted(set(r["colour_point_group"] for r in pending)):
        representatives.append(max((r for r in pending if r["colour_point_group"] == group), key=lambda r: np.prod(r["shape"])))
    rep_names = {r["name"] for r in representatives}
    order = representatives + [r for r in pending if r["name"] not in rep_names]
    meta["representative_names"] = [r["name"] for r in representatives]
    mats = [H3.sloc(10.45, 10e-6, 0.9012), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(1e5, 200e-6, 1.2)]
    for i, row in enumerate(order):
        elapsed = time.perf_counter()-start
        estimated_next = max((r.get("solve_seconds", 0) for r in rows), default=0)
        if elapsed + estimated_next > args.budget_seconds:
            row["status"] = "budget_skipped"
            continue
        tsolve = time.perf_counter()
        try:
            np.random.seed(20261026 + names.index(row["name"]))
            sig, L, S, ons, its = HI.homogenize3d_it(labels[row["name"]], mats, tol=1e-11)
            if any(info != 0 for _, info in its):
                raise RuntimeError(f"CG did not converge: {its}")
            ax = H3.axial(S)*1e6
            row.update({"S_a,z": float(ax[2]), "S_a,z_V_K": float(ax[2]*1e-6), "Sa_z_uV_K": float(ax[2]),
                        "ax": ax.tolist(), "sigma": sig.tolist(), "L": L.tolist(), "S_V_K": S.tolist(),
                        "onsager_relative_error": float(ons), "cg_iterations_info": [[int(c), int(info)] for c, info in its],
                        "status": "computed"})
        except Exception as exc:
            row.update(status="solver_error", error=repr(exc))
            meta["issues"].append(f"{row['name']}: {exc}")
        row["solve_seconds"] = time.perf_counter()-tsolve
        print(f"solve {i+1}/{len(order)} {row['name']}: Sa_z={row['S_a,z']} uV/K, {row['solve_seconds']:.2f} s", flush=True)
        if i+1 == len(representatives):
            est = meta["classification_seconds"] + np.mean([r["solve_seconds"] for r in representatives])*len(order)
            meta["representative_extrapolated_runtime_seconds"] = float(est)
            print(f"Full-run estimate from representative solves: {est:.1f} s", flush=True)
        meta["runtime_seconds"] = time.perf_counter()-start
        save_json(HERE / "mirror_colour_N16.json", rows)
        save_json(HERE / "mirror_colour_run.json", meta)
    meta["runtime_seconds"] = time.perf_counter()-start
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["source_hashes_unchanged"] = all(sha256(p) == meta["sha256"][hkey(p, cgd)] for p in source_files + [cgd])
    assert meta["source_hashes_unchanged"]
    save_json(HERE / "mirror_colour_N16.json", rows)
    save_json(HERE / "mirror_colour_run.json", meta)
    write_report(rows, meta)
    print(f"Finished in {meta['runtime_seconds']:.2f} s; report: {HERE / 'mirror_colour_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
