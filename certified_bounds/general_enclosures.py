"""Fixed-raster corner enrichment and primal/dual energy enclosures for contrast."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import ast
import hashlib
import inspect
import json
import time
import numpy as np
from common.transport2d import ROOT, save, solve, inverse_phases, BETA
from certified_bounds.high_contrast_sectors import sector_homogenize, sector_pair
from certified_bounds.design_enclosures import metrics


def generic_enclosure(primal, dual):
    """No rotational symmetry assumed. Linear matrix support plus rigorous inverse remainder."""
    rot = np.kron(np.eye(2), np.array([[0., -1.], [1., 0.]]))
    upper = np.array(primal['scaled_effective_energy'])
    lower = rot.T @ np.linalg.inv(dual['scaled_effective_energy']) @ rot
    upper = (upper + upper.T)/2
    lower = (lower + lower.T)/2
    gap = upper - lower
    w, v = np.linalg.eigh(gap)
    assert min(w) > -1e-8
    root = (v * np.sqrt(np.maximum(w, 0))) @ v.T
    center = (upper + lower)/2
    a, b = center[:2, :2], center[:2, 2:]
    sm = np.linalg.solve(a, b)
    wa, va = np.linalg.eigh(a)
    ah = (va / np.sqrt(wa)) @ va.T
    de = np.linalg.norm(ah @ gap[:2, :2] @ ah, 2)
    q = de/2
    assert q < 1
    y = np.vstack((-sm, np.eye(2)))
    dy = np.linalg.norm(y.T @ gap @ y, 2)
    remainder = np.linalg.norm(ah, 2)*q/(1-q)*np.sqrt(de*dy)/2
    def scalar(s, kind):
        return (s[0, 1] - s[1, 0])/2 if kind == 'Sa' else s[0, 0]
    out = {}
    for kind in ('Sa', 'Sxx'):
        h = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                e = np.zeros((4, 4)); e[i, j] = e[j, i] = 1
                f = scalar(np.linalg.solve(a, e[:2, 2:] - e[:2, :2] @ sm), kind)
                h[i, j] = h[j, i] = f if i == j else f/2
        radius = np.abs(np.linalg.eigvalsh(root @ h @ root)).sum()/2 + remainder
        mid = scalar(sm, kind)
        out[kind + '_interval_uV_K'] = [(mid-radius)*BETA*1e6, (mid+radius)*BETA*1e6]
    out.update(lower_energy=lower.tolist(), upper_energy=upper.tolist(), gap_eigenvalues=w.tolist(), inverse_relative_radius=q,
               nonlinear_remainder_uV_K=remainder*BETA*1e6, method='C: Loewner interval support function with norm-bounded nonlinear inverse remainder; no p4 assumption')
    return out


def rot_rect(N, cx, cy, a, b, ang, ph, val):
    ii, jj = np.meshgrid((np.arange(N)+0.5)/N, (np.arange(N)+0.5)/N, indexing='ij')
    x = (ii-cx+0.5) % 1-0.5; y = (jj-cy+0.5) % 1-0.5
    c, s = np.cos(ang), np.sin(ang); u = c*x + s*y; v = -s*x + c*y
    ph[(abs(u) < a/2) & (abs(v) < b/2)] = val


def original_p1(n):
    """Long thin bar at 0 degrees plus short fat block at 60 degrees: no mirror, no rotation axis."""
    ph = np.zeros((n, n), int)
    rot_rect(n, .5, .25, .92, .06, 0.0, ph, 1)
    rot_rect(n, .5, .70, .34, .22, np.deg2rad(60), ph, 1)
    return ph


def two_phase_axial_enclosure(enc):
    """Exact two-phase reduction makes the cross block B spatially symmetric.

    Sa/BETA = ((A11-A00)*B01 + A01*(B00-B11))/(2 det(A)).
    Linear matrix support is exact; the quadratic remainder is bounded by
    separate linear-functional radii. Determinant monotonicity bounds the
    positive denominator. This sharpens, never replaces, the generic check.
    """
    lower, upper = [np.array(enc[k]) for k in ('lower_energy', 'upper_energy')]
    center = (lower+upper)/2; gap = upper-lower
    w, v = np.linalg.eigh(gap); root = (v*np.sqrt(np.maximum(w, 0))) @ v.T
    def functionals(c):
        return np.array([c[1, 1]-c[0, 0], (c[0, 3]+c[1, 2])/2, c[0, 1], c[0, 2]-c[1, 3]])
    hs = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(i, 4):
            e = np.zeros((4, 4)); e[i, j] = e[j, i] = 1
            hs[:, i, j] = hs[:, j, i] = functionals(e)/(1 if i == j else 2)
    radius = lambda h: np.abs(np.linalg.eigvalsh(root @ h @ root)).sum()/2
    x, y, z, w0 = functionals(center)
    hlin = (y*hs[0]+x*hs[1]+w0*hs[2]+z*hs[3])/2
    radii = [radius(h) for h in hs]
    nonlinear = (radii[0]*radii[1]+radii[2]*radii[3])/2
    mid = (x*y+z*w0)/2; error = radius(hlin)+nonlinear
    determinant = [np.linalg.det(c[:2, :2]) for c in (lower, upper)]
    ratios = [BETA*1e6*a/b for a in (mid-error, mid+error) for b in determinant]
    return {'Sa_interval_uV_K': [float(min(ratios)), float(max(ratios))],
        'numerator_center': float(mid), 'linear_support_radius': float(radius(hlin)), 'quadratic_remainder': float(nonlinear),
        'determinant_interval': [float(x) for x in determinant],
        'method': 'C: exact spatial symmetry of two-phase cross block, quadratic numerator with analytic remainder, monotone positive determinant denominator'}


def p1_group(labels):
    valid = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                t = labels.T if swap else labels.copy()
                if sx < 0: t = t[::-1, :]
                if sy < 0: t = t[:, ::-1]
                corr = np.fft.ifft2(np.fft.fft2(labels)*np.conj(np.fft.fft2(t))).real
                for shift in np.argwhere(abs(corr - np.sum(labels**2)) < 1e-7):
                    if np.array_equal(np.roll(t, tuple(shift), axis=(0, 1)), labels):
                        valid.append({'swap': swap, 'sx': sx, 'sy': sy, 'shift': shift.tolist()}); break
    return valid


def run_p1(ms):
    name = 'certified_bounds/rod_block_enclosures.json'
    path = ROOT/name
    phases = [(1e5, 200e-6, 1.5), (1e7, 2e-6, 50.)]
    labels = original_p1(64)
    out = json.loads(path.read_text()) if path.exists() else {
        'source': 'certified_bounds/general_enclosures.py: rot_rect and original_p1 (long thin bar plus tilted block; fixed reference geometry)',
        'source_sha256': hashlib.sha256((inspect.getsource(rot_rect)+inspect.getsource(original_p1)).encode()).hexdigest(),
        'base_N': 64, 'labels': labels.tolist(), 'phases_sigma_S_kappa': phases,
        'geometry': 'Periodic metal rectangles (cx,cy,a,b,angle_deg): (0.5,0.25,0.92,0.06,0), (0.5,0.70,0.34,0.22,60), in p-type matrix.',
        'material_preserving_group': p1_group(labels), 'sector': [], 'uniform': [], 'regenerated_original_P': []}
    for m in ms:
        if any(x['m'] == m for x in out['sector']): continue
        primal = sector_homogenize(labels, phases, m, m)
        dual = sector_homogenize(labels, inverse_phases(phases), m, m)
        row = {'m': m, 'fem_P': primal, 'dual_P': dual, 'enclosure_C': generic_enclosure(primal, dual)}
        row['two_phase_C'] = two_phase_axial_enclosure(row['enclosure_C'])
        out['sector'].append(row); save(name, out)
        print('p1 sector', m, row['two_phase_C']['Sa_interval_uV_K'], flush=True)
    for row in out['sector']:
        row['two_phase_C'] = two_phase_axial_enclosure(row['enclosure_C'])
    save(name, out)
    for k in (1, 2, 4):
        if any(x['k'] == k for x in out['uniform']): continue
        raster = np.kron(labels, np.ones((k, k), int))
        row = {'k': k, 'fem_P': solve(raster, phases), 'fv_P': solve(raster, phases, 'fv')}
        out['uniform'].append(row); save(name, out)
        print('p1 uniform', k, row['fem_P']['Sa'], row['fv_P']['Sa'], flush=True)
    for n in (64, 128, 192):
        if any(x['N'] == n for x in out['regenerated_original_P']): continue
        out['regenerated_original_P'].append(solve(original_p1(n), phases, 'fv')); save(name, out)
    out['mirror_fv_P'] = solve(labels[:, ::-1].copy(), phases, 'fv')
    save(name, out)


def run_insulator(ms, kappas):
    old = json.loads((ROOT/'certified_bounds/design_enclosures.json').read_text())['cases']['A_64']
    labels = np.array(old['labels'])
    name = 'certified_bounds/insulating_planar.json'; path = ROOT/name
    out = json.loads(path.read_text()) if path.exists() else {
        'source': 'certified_bounds/design_enclosures.json cases.A_64 exact fixed raster', 'labels': labels.tolist(), 'N_base': 64,
        'original_phases': old['phases'], 'original_sector': old['sector'],
        'classification': 'C = variational discretization enclosure, floating-point arithmetic and quadrature are not interval certified. P = point estimate.', 'cases': {}}
    for kap in kappas:
        phases = [(1e-6, 0., kap), (1e5, -180e-6, 1.3), (1e5, 200e-6, 1.2)]
        case = out['cases'].setdefault(str(kap), {'phases_sigma_S_kappa': phases, 'sector': [], 'uniform': []})
        legacy = [r for r in case['sector'] if 'method' not in r['fem']]
        if legacy:
            case['rejected_unstable_legacy_P'] = [{'m': r['fem']['m'], 'gap_eigenvalues': r['enclosure']['gap_eigenvalues'],
                'Sa_point': r['fem']['Sa'], 'reason': 'Extreme-contrast squared eigensolve / subtractive energy not accepted as C; m=2 exhibited a negative gap.'} for r in legacy]
            case['sector'] = [r for r in case['sector'] if 'method' in r['fem']]
            save(name, out)
        for m in ms:
            if any(x['fem']['m'] == m for x in case['sector']): continue
            result = sector_pair(labels, phases, m, m)
            result['metrics_C'] = metrics(result, phases)
            case['sector'].append(result); save(name, out)
            print('insulator', kap, m, result['metrics_C'], flush=True)
    save(name, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=('p1', 'insulator'))
    parser.add_argument('--m', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--kappa', type=float, nargs='+', default=[.3, 1.])
    args = parser.parse_args()
    if args.task == 'p1': run_p1(args.m)
    else: run_insulator(args.m, args.kappa)
