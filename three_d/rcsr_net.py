"""Nets of the RCSR 3D file: nodes and struts reconstructed from the NODE and EDGE_CENTER records.

expand(net) returns the expanded nodes and their coordination numbers; edges(net, pts, cns) returns the struts as
fractional (start, end) pairs, one per undirected periodic edge. Both are used by rcsr_color.py.

Origin choice 2 is used for the thirteen tetragonal groups with two origin choices: in it the EDGE_CENTER points are
midpoints of node pairs and the coordination numbers are reproduced. For each expanded edge centre one
midpoint-compatible node pair is chosen, minimizing the total squared strut length subject to every node CN;
records without a unique such assignment, or with centres that join no node pair, raise ValueError.
The edge set is checked to be invariant under the space group. No record is edited.
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

import os

from itertools import product
from pathlib import Path
import numpy as np
import gemmi

CGD = os.environ.get('RCSR_CGD', str(RELEASE_ROOT / 'three_d' / 'rcsr3d_web.cgd'))   # see README
ORIGIN_TWO = {85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142}
POINT_TOL = 1e-4
MIDPOINT_TOL = 5e-4  # fractional residual p + q - 2m; sfo needs 4e-4
SHIFTS = np.array(list(product((-1, 0, 1), repeat=3)), dtype=int)


def read_cgd(path=CGD):
    out, net = [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.split()
        if not s or s[0].startswith("#"):
            continue
        if s[0] == "CRYSTAL":
            net = {"nodes": [], "edge_centers": []}
        elif net is None:
            continue
        elif s[0] == "NAME":
            net["name"] = s[1]
        elif s[0] == "GROUP":
            net["group"] = s[1]
        elif s[0] == "CELL":
            net["cell"] = list(map(float, s[1:7]))
        elif s[0] in ("NODE", "EDGE_CENTER"):
            key = "nodes" if s[0] == "NODE" else "edge_centers"
            net[key].append((int(s[2]), np.array(list(map(float, s[3:6])))))
        elif s[0] == "END":
            out.append(net)
            net = None
    return out


def spacegroup(net, choice=None):
    sg = gemmi.find_spacegroup_by_name(net["group"])
    if sg is None:
        raise ValueError(f"Unknown space group: {net['group']}")
    if sg.number in ORIGIN_TWO:
        choice = 2 if choice is None else choice
        sg = gemmi.find_spacegroup_by_name(net["group"].split(":")[0] + f":{choice}")
    return sg


def expand_sites(sites, sg):
    pts, cns = [], []
    for cn, xyz in sites:
        for op in sg.operations():
            p = np.array(op.apply_to_xyz(list(xyz))) % 1.0
            matches = [i for i, q in enumerate(pts)
                       if np.linalg.norm((p - q + .5) % 1 - .5) < POINT_TOL]
            if not matches:
                pts.append(p)
                cns.append(cn)
            elif any(cns[i] != cn for i in matches):
                raise ValueError("Overlapping node sites have inconsistent CN")
    return np.array(pts).reshape((-1, 3)), np.array(cns, dtype=int)


def expand(net):
    return expand_sites(net["nodes"], spacegroup(net))


def canonical_edge(i, j, shift):
    s = tuple(map(int, shift))
    return min((int(i), int(j), s), (int(j), int(i), tuple(-x for x in s)))


def orthogonal_cell(net):
    if not np.allclose(net["cell"][3:], [90, 90, 90], atol=1e-8, rtol=0):
        raise ValueError("Only orthogonal cells are supported")
    return np.array(net["cell"][:3])


def centre_candidates(net, pts, m, tolerance=MIDPOINT_TOL):
    """All shortest image candidates, plus next image shells, at a centre.

    With endpoints p+n and q+t-n, t=round(2m-p-q); the periodic edge
    key has shift t-2n. For an orthogonal cell its shortest image lies
    at n=round((q+t-p)/2). Neighbouring n include all tied minima.
    """
    lengths = orthogonal_cell(net)
    residual = 2*m - pts[:, None, :] - pts[None, :, :]
    translations = np.rint(residual).astype(int)
    errors = np.max(np.abs(residual-translations), axis=2)
    pairs = np.argwhere(errors <= tolerance)
    candidates = {}
    for i, j in pairs:
        t = translations[i, j]
        n0 = np.rint((pts[j] + t - pts[i])/2).astype(int)
        ns = n0 + SHIFTS
        shifts = t - 2*ns
        lens = np.linalg.norm((pts[j] + shifts - pts[i])*lengths, axis=1)
        for shift, length in zip(shifts, lens):
            if length <= 1e-6:
                continue
            key = canonical_edge(i, j, shift)
            candidates[key] = (float(length), float(errors[i, j]))
    return sorted((length, key, error) for key, (length, error) in candidates.items())


def pair_options(candidates):
    """Keep shortest image(s) for each unordered node-index pair.

    Longer images of the same pair have identical degree contributions and
    cannot improve the length objective. Retain tied images to detect ambiguity.
    """
    best = {}
    for length, key, error in candidates:
        pair = key[:2]
        if pair not in best:
            best[pair] = length
    return [item for item in candidates if item[0] <= best[item[1][:2]] + 1e-7]


def solve_centres(options, cns, limit=2):
    """Minimum total squared length, one pair per centre, with exact node CN.

    Retain up to two optimum solutions: arbitrary tie breaks are not allowed.
    Each option is an undirected periodic edge, and a self-edge contributes 2.
    """
    degree = np.zeros(len(cns), dtype=int)
    solutions, chosen, used = [], {}, set()
    best_cost = float("inf")
    cost_tolerance = 1e-9

    def contribution(edge):
        inc = np.zeros(len(cns), dtype=int)
        inc[edge[0]] += 1
        inc[edge[1]] += 1
        return inc

    increments = {key: contribution(key) for opts in options for _, key, _ in opts}
    costs = [{key: length**2 for length, key, _ in opts} for opts in options]

    def search(remaining, cost):
        nonlocal best_cost
        if not remaining:
            if np.array_equal(degree, cns):
                if cost < best_cost - cost_tolerance:
                    best_cost = cost
                    solutions.clear()
                if cost <= best_cost + cost_tolerance and len(solutions) < limit:
                    solutions.append([chosen[k] for k in range(len(options))])
            return
        viable = {k: [e for _, e, _ in options[k] if e not in used and
                      np.all(degree + increments[e] <= cns)] for k in remaining}
        if any(not opts for opts in viable.values()):
            return
        bound = cost + sum(min(costs[k][e] for e in opts) for k, opts in viable.items())
        if bound > best_cost + cost_tolerance:
            return
        if len(solutions) >= limit and bound >= best_cost - cost_tolerance:
            return
        lower = sum((np.min([increments[e] for e in opts], axis=0)
                     for opts in viable.values()), np.zeros(len(cns), dtype=int))
        upper = sum((np.max([increments[e] for e in opts], axis=0)
                     for opts in viable.values()), np.zeros(len(cns), dtype=int))
        if np.any(degree + lower > cns) or np.any(degree + upper < cns):
            return
        k = min(remaining, key=lambda x: len(viable[x]))
        rest = [x for x in remaining if x != k]
        for edge in viable[k]:
            chosen[k] = edge
            used.add(edge)
            degree[:] += increments[edge]
            search(rest, cost + costs[k][edge])
            degree[:] -= increments[edge]
            used.remove(edge)
            del chosen[k]

    if len(options)*2 == sum(cns) and all(options):
        search(list(range(len(options))), 0.)
    return solutions, best_cost


def reconstruct(net, pts=None, cns=None, choice=None, tolerance=MIDPOINT_TOL):
    sg = spacegroup(net, choice)
    if pts is None:
        pts, cns = expand_sites(net["nodes"], sg)
    if not net.get("edge_centers"):
        raise ValueError("EDGE_CENTER records required; use this module's read_cgd")
    if any(cn != 2 for cn, _ in net["edge_centers"]):
        raise ValueError("An EDGE_CENTER must have degree two")
    centres, _ = expand_sites(net["edge_centers"], sg)
    keys, info, failures, options = [], [], [], []
    for k, m in enumerate(centres):
        candidates = centre_candidates(net, pts, m, tolerance)
        if not candidates:
            failures.append({"centre": k, "reason": "no midpoint pair"})
            options.append([])
            continue
        best = candidates[0][0]
        shortest = [x for x in candidates if x[0] <= best + 1e-7]
        options.append(pair_options(candidates))
        if len(shortest) == 1:
            keys.append(shortest[0][1])
        info.append({"centre": k, "length": best,
                     "residual": max(x[2] for x in shortest),
                     "shortest_pair_count": len(shortest),
                     "next_length": candidates[len(shortest)][0]
                                    if len(candidates)>len(shortest) else None})
    if 2*len(centres) != sum(cns):
        failures.append({"reason": "centre count differs from sum(CN)/2",
                         "centres": len(centres), "expected": int(sum(cns))/2})
    solutions, best_cost = solve_centres(options, cns)
    if len(solutions) == 1:
        keys = solutions[0]
        for row in info:
            k = row["centre"]
            selected = next(x for x in options[k] if x[1] == keys[k])
            row["selected_length"] = selected[0]
            row["selected_residual"] = selected[2]
            row["selected_longer_than_shortest"] = selected[0] > row["length"] + 1e-7
    else:
        failures.append({"reason": "no CN-consistent pair assignment"
                         if not solutions else "multiple minimum-length CN-consistent assignments"})
    degree = np.zeros(len(pts), dtype=int)
    for i, j, shift in set(keys):
        degree[i] += 1
        degree[j] += 1
    valid = not failures and len(set(keys)) == len(centres) and np.array_equal(degree, cns)
    invariant = symmetry_invariant(pts, keys, sg) if valid else None
    if valid and not invariant:
        failures.append({"reason": "edge set violates space-group symmetry"})
        valid = False
    return {"pts": pts, "cns": cns, "centres": centres, "keys": sorted(set(keys)),
            "degree": degree, "info": info, "failures": failures, "valid": valid,
            "setting": sg.xhm(), "options": options,
            "assignment": solutions[0] if len(solutions) == 1 else None,
            "spacegroup_invariant": invariant,
            "objective_sum_squared_lengths": best_cost if solutions else None,
            "solutions_found_capped_at_2": len(solutions)}


def symmetry_invariant(pts, keys, sg):
    keys = set(keys)
    for op in sg.operations():
        for i, j, shift in keys:
            a = np.array(op.apply_to_xyz(pts[i].tolist()))
            b = np.array(op.apply_to_xyz((pts[j]+shift).tolist()))
            da = np.linalg.norm((pts-a+.5) % 1-.5, axis=1)
            db = np.linalg.norm((pts-b+.5) % 1-.5, axis=1)
            ki, kj = int(np.argmin(da)), int(np.argmin(db))
            if da[ki] > POINT_TOL or db[kj] > POINT_TOL:
                return False
            s = np.rint((b-pts[kj])-(a-pts[ki])).astype(int)
            if canonical_edge(ki, kj, s) not in keys:
                return False
    return True


def edges(net, pts, cns):
    result = reconstruct(net, pts, cns)
    if not result["valid"]:
        raise ValueError(f"{net['name']}: midpoint reconstruction failed checks: "
                         f"{result['failures']}; degree={result['degree'].tolist()}, "
                         f"CN={np.asarray(cns).tolist()}")
    return [(pts[i].copy(), pts[j] + np.array(shift)) for i, j, shift in result["keys"]]
