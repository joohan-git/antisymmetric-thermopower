"""Colour-preserving point groups of every computed RCSR-derived structure, as Hermann-Mauguin symbols.
Voxel group: the 16-voxel map of three_d/run_rcsr_all.py (strut radius 0.08a); every point operation of 4/mmm is tested against
every voxel translation (FFT cross-correlation of the phase indicators), so screw axes such as 4_1 are found for any
number of voxels along z. Continuous group: the space-group operations of the net that map every strut onto a strut of
the same colour (colouring rule of rcsr_color.voxelize), i.e. the group of the coloured-strut structure before
voxelization. Checks that the voxel verdict reproduces rcsr_color_N16_all.json. Output rcsr_colour_pg.json."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import os; import sys; import json; import numpy as np; import gemmi
from three_d import rcsr_color as R; from three_d import coloring as C; from three_d import rcsr_net as RN


def axis(v):
    v = np.round(v / np.max(np.abs(v)), 3)
    names = {(0, 0, 1): 'z', (1, 0, 0): 'x', (0, 1, 0): 'y', (1, 1, 0): '[110]', (1, -1, 0): '[1-10]'}
    for k, s in names.items():
        if np.allclose(np.abs(v), np.abs(k)) and (s not in ('[110]', '[1-10]') or np.sign(v[0] * v[1]) == np.sign(k[0] * k[1])):
            return s
    return str(v.tolist())


def eigvec(Rm, lam):
    w, V = np.linalg.eig(Rm.astype(float)); return np.real(V[:, np.argmin(np.abs(w - lam))])


def symbol(G):
    det = [round(np.linalg.det(g)) for g in G]; tr = [int(np.trace(g)) for g in G]
    n4 = sum(1 for d, t in zip(det, tr) if d == 1 and t == 1)
    n2 = [g for g, d, t in zip(G, det, tr) if d == 1 and t == -1]
    nm = [g for g, d, t in zip(G, det, tr) if d == -1 and t == 1]
    n4b = sum(1 for d, t in zip(det, tr) if d == -1 and t == -1)
    inv = any(t == -3 for t in tr); o = len(G)
    if o == 16: return '4/mmm'
    if o == 8:
        if n4 == 2 and inv: return '4/m'
        if n4 == 2 and len(n2) == 5: return '422'
        if n4 == 2 and len(nm) == 4: return '4mm'
        if n4b == 2: return '-42m'
        if inv and len(n2) == 3: return 'mmm'
    if o == 4:
        if n4 == 2: return '4'
        if n4b == 2: return '-4'
        if len(n2) == 3: return '222'
        if inv: return f'2/m (2 || {axis(eigvec(n2[0], 1))})'
        if len(nm) == 2: return f'mm2 (2 || {axis(eigvec(n2[0], 1))})'
    if o == 2:
        if inv: return '-1'
        if n2: return f'2 (|| {axis(eigvec(n2[0], 1))})'
        if nm: return f'm (normal {axis(eigvec(nm[0], -1))})'
    if o == 1: return '1'
    return f'order {o}'


def has_translation(A, B):
    """True if B equals A shifted by some translation of the voxel grid, with every phase matched."""
    tot = None
    for ph in np.unique(A):
        a = (A == ph).astype(float); b = (B == ph).astype(float)
        if a.sum() != b.sum(): return False
        corr = np.real(np.fft.ifftn(np.fft.fftn(b) * np.conj(np.fft.fftn(a))))
        tot = corr if tot is None else tot + corr
    return bool(np.max(tot) > A.size - 0.5)


def colour_group_full(T):
    return [Rm for Rm in C.OPS if has_translation(T, C.apply(T, Rm))]


def strut_colour(d):
    """Colour of a strut with Cartesian direction d, as in rcsr_color.voxelize."""
    if np.linalg.norm(d[:2]) < 1e-6*np.linalg.norm(d): return 'p'
    th = round(float(np.degrees(np.arctan2(d[1], d[0])) % 90.0), 6) % 90.0
    return 'p' if 1e-6 < th < 45 - 1e-6 else ('n' if th > 45 + 1e-6 else 'p')


def continuous_group(net, segs):
    L = np.array(net['cell'][:3]); S = [(((a + b)/2) % 1.0, (b - a)*L) for a, b in segs]; G = []
    for op in RN.spacegroup(net).operations():
        W = np.array(op.rot)/op.DEN; w = np.array(op.tran)/op.DEN; ok = True
        for m, d in S:
            m2 = (W @ m + w) % 1.0; d2 = (W @ (d/L))*L
            assert any(np.linalg.norm(((x - m2 + 0.5) % 1) - 0.5) < 1e-4 for x, _ in S)          # the uncoloured net is invariant
            if strut_colour(d2) != strut_colour(d): ok = False; break
        Wi = np.rint(W).astype(int)
        if ok and not any(np.array_equal(Wi, g) for g in G): G.append(Wi)
    return G, sum(strut_colour(d) == 'n' for _, d in S)


if __name__ == '__main__':
    here = str(RELEASE_ROOT)
    ref = {r['name']: r for r in json.load(open(os.path.join(here, 'three_d/rcsr_color_N16_all.json')))}
    nets = {x['name']: x for x in R.read_cgd()}; out = []
    for name, r in ref.items():
        net = nets[name]; pts, cns = R.expand(net); segs = R.edges(net, pts, cns); lab = R.voxelize(net, segs, 16, 0.08)
        G = colour_group_full(lab); allowed = C.axial_allowed(G)[0]; Gc, n_struts_n = continuous_group(net, segs)
        out.append(dict(name=name, parent_space_group=net['group'], setting=RN.spacegroup(net).xhm(), shape=list(lab.shape),
                        colour_point_group=symbol(G), order=len(G), axial_z_allowed=bool(allowed[2]),
                        continuous_colour_point_group=symbol(Gc), continuous_axial_z_allowed=bool(C.axial_allowed(Gc)[0][2]),
                        n_type_struts=int(n_struts_n), struts=len(segs), Sa_z_uV_K=r['ax'][2],
                        matches_atlas_verdict=bool(allowed[2]) == bool(r['allowed'][2])))
        print(name, net['group'], out[-1]['colour_point_group'], out[-1]['continuous_colour_point_group'], 'allowed_z', allowed[2], flush=True)
    json.dump(out, open(os.path.join(here, 'three_d/rcsr_colour_pg.json'), 'w'), indent=1)
    print(len(out), 'nets;', sum(not o['matches_atlas_verdict'] for o in out), 'verdict mismatches;',
          sum(o['colour_point_group'] != o['continuous_colour_point_group'] for o in out), 'voxel groups differ from the continuous group;',
          sum(o['axial_z_allowed'] != o['continuous_axial_z_allowed'] for o in out), 'verdicts differ')
