"""Mirror-preserving RCSR colourings, N=16 and radius=0.08a.

Run from any directory; all generated files stay beside this script. Example:
    python -B three_d/mirror_colour.py --cgd <path/to/rcsr3d_web.cgd>
Transport is computed with the shared solver at tol=1e-11.
S_a,z and ax in the output are in microvolt/kelvin; frac is [weak, n, p].
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)

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

HERE = RELEASE_ROOT
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
    return "rcsr3d_web.cgd (RCSR, not redistributed)" if Path(p) == Path(cgd) else Path(p).resolve().relative_to(RELEASE_ROOT).as_posix()


def source_dir():
    return RELEASE_ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=source_dir())
    parser.add_argument("--cgd", type=Path, default=os.environ.get("RCSR_CGD"))
    parser.add_argument("--time-limit-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.source is None:
        parser.error("Cannot locate the release; supply --source")
    src = args.source.resolve()
    cgd = (args.cgd or src / "three_d" / "rcsr3d_web.cgd").resolve()
    if not cgd.is_file():
        parser.error("CGD missing; supply --cgd or RCSR_CGD")
    os.environ["RCSR_CGD"] = str(cgd)
    from three_d import rcsr_color as R
    from three_d import rcsr_net as RN
    from three_d import coloring as C
    from common import homog3d as H3
    from common import homog3d_it as HI
    from three_d import rcsr_colour_pg as PG
    import scipy
    import gemmi
    import pyamg

    start = time.perf_counter()
    names = json.loads((src / "three_d/rcsr_filter.json").read_text())["steps"][-1]["names"]
    assert len(names) == 116 and len(set(names)) == 116
    original = {r["name"]: r for r in json.loads((src / "three_d/rcsr_color_N16_all.json").read_text())}
    assert set(names) == set(original)
    nets = {r["name"]: r for r in R.read_cgd()}
    source_files = [src / s for s in ("three_d/rcsr_filter.json", "three_d/rcsr_color_N16_all.json", "three_d/rcsr_color.py",
                    "three_d/rcsr_net.py", "three_d/rcsr_colour_pg.py", "three_d/coloring.py", "common/homog3d.py", "common/homog3d_it.py", "three_d/run_rcsr_all.py")]
    meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_tree='.', cgd_path=cgd.name,
                N=16, radius_over_a=0.08, angle_tolerance_degrees=ANGLE_TOL, solver_tolerance=1e-11,
                numerical_zero_threshold_uV_K=ZERO_UV_K, time_limit_seconds=args.time_limit_seconds,
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
                   status="pending" if nph == 3 or geom_allowed[2] else "not_selected_two_phase")
        rows.append(row)
        if row["status"] == "pending":
            labels[name] = lab
        if not row["voxel_contains_geometric_group"]:
            meta["issues"].append(f"{name}: voxel {row['colour_point_group']} loses operations of geometric {row['geometric_point_group']}")
        print(f"classify {i+1}/116 {name}: {nph} phases, {row['colour_point_group']}, allowed_z={allowed[2]}", flush=True)
        save_json(HERE / "three_d/mirror_colour_N16.json", rows)
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
        if elapsed + estimated_next > args.time_limit_seconds:
            row["status"] = "time_limit_skipped"
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
        save_json(HERE / "three_d/mirror_colour_N16.json", rows)
        save_json(HERE / "three_d/mirror_colour_run.json", meta)
    meta["runtime_seconds"] = time.perf_counter()-start
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["source_hashes_unchanged"] = all(sha256(p) == meta["sha256"][hkey(p, cgd)] for p in source_files + [cgd])
    assert meta["source_hashes_unchanged"]
    save_json(HERE / "three_d/mirror_colour_N16.json", rows)
    save_json(HERE / "three_d/mirror_colour_run.json", meta)
    print("Completed recolouring calculations.", flush=True)


if __name__ == "__main__":
    main()
