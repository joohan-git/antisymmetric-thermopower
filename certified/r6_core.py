"""Two-dimensional reproducible calculations. The r5 files and the solvers/fv2d sources are read only."""
from pathlib import Path
import ast
import hashlib
import json
import time
import types
import numpy as np
from scipy.optimize import brentq, least_squares
from r5_fem import homogenize, pinwheel, PHASES, material_energy, BETA

ROOT = Path(__file__).resolve().parent

def rel(p):
    """Path relative to the release folder, for hash records (never an absolute local path)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        return p.name
MATERIALS = {
    'p': (1e5, 200e-6, 1.2),
    'n': (1e5, -180e-6, 1.3),
    'Cu': (6e7, 1.8e-6, 400.),
    'Ni': (1.4e7, -19e-6, 90.),
    'PEDOT': (1e5, 20e-6, .3),
    'carbon': (1e3, 10e-6, .3),
    'polymer_void': (1e-6, 0., .2),
}


def save(name, value):
    path = ROOT / name
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + chr(10)
    temporary = path.with_name(path.name+'.'+str(time.time_ns())+'.tmp')
    assert temporary.resolve().parent == ROOT and path.resolve().parent == ROOT
    temporary.write_text(payload, encoding='utf-8')
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except OSError:
            if attempt == 11:
                raise
            time.sleep(.1*(attempt+1))


def sources():
    paths = [ROOT/'r5_fem.py', ROOT.parent/'solvers/fv2d/homog2.py', ROOT.parent/'solvers/fv2d/check3.py']
    return {rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_fv():
    src = (ROOT.parent/'solvers/fv2d/homog2.py').read_text(encoding='utf-8')
    # Preserve a byte-equivalent text snapshot in our workspace; add full tensor return in memory only.
    snapshot = ROOT/'r6_fv_original_snapshot.py'
    if not snapshot.exists() or snapshot.read_text(encoding='utf-8') != src:
        snapshot.write_text(src, encoding='utf-8')
    mod = types.ModuleType('r6_original_fv')
    src = src.replace('return sig, L, np.linalg.solve(sig, L)',
                      'return sig, L, np.linalg.solve(sig, L), Meff')
    exec(compile(src, 'r6_fv_original_snapshot.py', 'exec'), mod.__dict__)
    return mod


FV = load_fv()


def original_cell3(n=24, angle=15, period=6):
    src = (ROOT.parent/'solvers/fv2d/check3.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'cell3')
    ns = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'original_cell3_readonly', 'exec'), ns)
    return ns['cell3'](n, angle, period)


def compact(result):
    r = dict(result)
    r['Sa'] = r['Santi_main_uV_K']
    r['Sxx'] = r['S_uV_K'][0][0]
    r['sigma_iso'] = float(np.trace(r['sigma'])/2)
    return r


def solve(labels, phases, method='fem'):
    if method == 'fem':
        return compact(homogenize(labels, phases))
    started = time.perf_counter()
    sig, alpha, seebeck, m = FV.homogenize(labels, [FV.mloc(*x) for x in phases])
    energy = m.reshape(4, 4).copy()
    energy[2:] /= 300
    factors = np.array([BETA, BETA, 1., 1.])
    energy *= factors[:, None]*factors[None, :]/.004
    return {'N': len(labels), 'sigma': sig.tolist(), 'sigma_iso': float(np.trace(sig)/2),
            'alpha': alpha.tolist(), 'S_uV_K': (seebeck*1e6).tolist(),
            'Sa': float((seebeck[0, 1]-seebeck[1, 0])*5e5), 'Sxx': float(seebeck[0, 0]*1e6),
            'scaled_effective_energy': energy.tolist(),
            'onsager_relative': float(np.linalg.norm(energy-energy.T)/np.linalg.norm(energy)),
            'seconds': time.perf_counter()-started}


def richardson(rows, key):
    k = np.array([r['k'] for r in rows], float)
    v = np.array([r[key] for r in rows], float)
    triplets = []
    for i in range(len(v)-2):
        d1, d2 = v[i+1]-v[i], v[i+2]-v[i+1]
        item = {'k': k[i:i+3].tolist()}
        if d1*d2 > 0 and abs(d1) > abs(d2) and abs(d2) > 1e-12:
            p = np.log2(d1/d2)
            item.update(p=float(p), limit=float(v[i+2]+d2/(2**p-1)))
        else:
            item['status'] = 'not a decaying single power on this triplet'
        triplets.append(item)
    return triplets


def inverse_phases(phases):
    # Invert the dimensionless symmetric 2x2 coefficient, then encode it using r5 API.
    result = []
    for phase in phases:
        a = np.linalg.inv(material_energy(phase))
        sigma = a[0, 0]*.004/BETA**2
        seebeck = a[0, 1]/a[0, 0]*BETA
        kappa = 300*.004*(a[1, 1]-a[0, 1]**2/a[0, 0])
        result.append((float(sigma), float(seebeck), float(kappa)))
    return result


def energy_enclosure(labels, phases, primal=None):
    """Primal conforming Q1 upper and rotated dual Q1 lower energy matrices.

    Exact in 2D for any number of coupled scalar potential fields. This is a
    discretization enclosure, not a rigorous floating point interval computation.
    """
    primal = solve(labels, phases) if primal is None else primal
    dual = solve(labels, inverse_phases(phases))
    assert np.array_equal(labels, np.rot90(labels))
    return enclosure_from_pair(primal, dual)


def enclosure_from_pair(primal, dual):
    rot = np.kron(np.eye(2), np.array([[0., -1.], [1., 0.]]))
    upper = np.array(primal['scaled_effective_energy'])
    lower = rot.T @ np.linalg.inv(dual['scaled_effective_energy']) @ rot
    upper = (upper+upper.T)/2
    lower = (lower+lower.T)/2
    gap = upper-lower
    w, v = np.linalg.eigh(gap)
    root = (v*np.sqrt(np.maximum(w, 0))) @ v.T
    # p4 gives sigma=a I and Sa/BETA=(C03-C12)/(2a).
    ha = np.diag([.5, .5, 0., 0.])
    hc = np.zeros((4, 4)); hc[0, 3] = hc[3, 0] = .25; hc[1, 2] = hc[2, 1] = -.25
    hb = np.zeros((4, 4)); hb[0, 2] = hb[2, 0] = .25; hb[1, 3] = hb[3, 1] = .25
    def maximum(h):
        return float(np.sum(h*lower)+np.maximum(np.linalg.eigvalsh(root@h@root), 0).sum())
    def ratio_bounds(h):
        f_upper = lambda t: maximum(h-t*ha)
        f_lower = lambda t: -maximum(-h+t*ha)
        radius = 1.
        while f_upper(radius) > 0 or f_lower(-radius) < 0:
            radius *= 2
        return [float(brentq(f_lower, -radius, radius)*BETA*1e6),
                float(brentq(f_upper, -radius, radius)*BETA*1e6)]
    return {'lower_energy': lower.tolist(), 'upper_energy': upper.tolist(),
            'gap_eigenvalues': w.tolist(), 'Sa_interval_uV_K': ratio_bounds(hc),
            'Sxx_interval_uV_K': ratio_bounds(hb),
            'sigma_interval': [float(np.trace(lower[:2,:2])/2*1e5),float(np.trace(upper[:2,:2])/2*1e5)],
            'dual_seconds': dual['seconds'], 'dual_residual': dual.get('residual_relative')}
