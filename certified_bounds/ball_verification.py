"""Verified variational bounds from exact dyadic Q1 and RT0 trial fields.

Integer Gram matrices integrate polynomials exactly. Local material matrices,
small inverses, and semidefinite dual certificates use python-flint Arb balls.
Floating eigenvectors only propose witnesses; Arb LDL validates every witness.
"""

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
from fractions import Fraction
import hashlib
import inspect
import itertools
import json
import time
import numpy as np
from flint import arb, arb_mat, fmpq, fmpz_mat, ctx
from common.transport2d import ROOT, save, PHASES, inverse_phases
from weak_contrast.total_matrix_contrast import cell
from common import fem2d
from common import rectangular_voxels

ctx.prec = 192
BITS = 40
DEN = 1 << BITS


def ball(x):
    if isinstance(x, arb): return x
    if isinstance(x, (float, np.floating)):
        q = Fraction.from_float(float(x))
    else:
        q = Fraction(x)
    return arb(fmpq(q.numerator, q.denominator))


def decimal_materials(phases):
    matrices = []
    beta = ball('0.0002'); escale = ball('0.004'); temp = arb(300)
    for phase in phases:
        sigma, seebeck, kap = [ball(str(x)) for x in phase]
        a = arb_mat([[sigma*beta**2/escale, sigma*beta*seebeck/escale],
                     [sigma*beta*seebeck/escale, (kap/temp+sigma*seebeck**2)/escale]])
        assert spd(a)[0]
        matrices.append(a)
    return matrices


def dump_matrix(a):
    return [[a[i, j].str(60, more=True) for j in range(a.ncols())] for i in range(a.nrows())]


def mid_array(a):
    return np.array([[float(a[i, j].mid()) for j in range(a.ncols())] for i in range(a.nrows())])


def spd(a):
    n = a.nrows(); lower = [[arb(0) for unused in range(n)] for unused in range(n)]; pivots = []
    for i in range(n):
        # In python-flint 0.9, generic ** may return nan for zero-containing
        # balls even at exponent two. Explicit multiplication is an enclosure.
        pivot = a[i, i]-sum((lower[i][k]*lower[i][k]*pivots[k] for k in range(i)), arb(0))
        if not pivot > 0: return False, [x.str(50, more=True) for x in pivots]+[pivot.str(50, more=True)]
        pivots.append(pivot); lower[i][i] = arb(1)
        for j in range(i+1, n):
            lower[j][i] = (a[j, i]-sum((lower[j][k]*lower[i][k]*pivots[k] for k in range(i)), arb(0)))/pivot
    return True, [x.str(50, more=True) for x in pivots]


def expose_trial(function, extra):
    tree = ast.parse(inspect.getsource(function))
    class AddTrial(ast.NodeTransformer):
        def visit_Return(self, node):
            return ast.copy_location(ast.Return(ast.Tuple([node.value, ast.Name(extra, ast.Load())], ast.Load())), node)
    tree = AddTrial().visit(tree); ast.fix_missing_locations(tree)
    namespace = function.__globals__.copy()
    exec(compile(tree, 'dyadic_trial_extraction', 'exec'), namespace)
    return namespace[function.__name__]


def quantize(values):
    assert np.isfinite(values).all()
    assert np.max(np.abs(values))*DEN < 2**59
    return np.rint(values*DEN).astype(np.int64)


def gram(rows):
    a = fmpz_mat(rows.tolist())
    return a.transpose()*a


def combine(grams, materials, denominator, loads):
    out = arb_mat(loads, loads)
    for i in range(loads):
        for j in range(i, loads):
            x = arb(0)
            for g, s in zip(grams, materials):
                for a in range(2):
                    for b in range(2):
                        x += s[a, b]*arb(g[a*loads+i, b*loads+j])
            out[i, j] = out[j, i] = x/ball(denominator)
    return out


def q1_energy(labels, nodal, materials):
    """Tensor-product Legendre moments of the exact multilinear gradient."""
    dim = labels.ndim; loads = 2*dim; count = labels.size
    u = nodal.reshape(tuple(labels.shape)+(2, loads))
    grams = [fmpz_mat(2*loads, 2*loads) for unused in materials]
    corners = list(itertools.product((0, 1), repeat=dim-1))
    for d in range(dim):
        others = [k for k in range(dim) if k != d]
        edges = []
        for corner in corners:
            shifts = [0]*dim
            for ax, value in zip(others, corner): shifts[ax] = -value
            left = np.roll(u, tuple(shifts), axis=tuple(range(dim)))
            shifts[d] = -1
            right = np.roll(u, tuple(shifts), axis=tuple(range(dim)))
            g = (right-left).reshape(count, 2, loads).copy()
            for field in range(2): g[:, field, field*dim+d] += DEN
            edges.append(g)
        for mode in corners:
            values = sum(((-1)**sum(a*b for a, b in zip(mode, corner))*g for corner, g in zip(corners, edges)), start=np.zeros_like(edges[0]))
            weight = 3**(dim-1-sum(mode))
            values = values.reshape(count, 2*loads)
            for p in range(len(materials)):
                grams[p] += weight*gram(values[labels.ravel() == p])
    denominator = count*DEN**2*4**(dim-1)*3**(dim-1)
    return combine(grams, materials, denominator, loads), grams, denominator


def integer_divergence(f):
    dim = f.ndim-3
    return sum((f[..., d, :]-np.roll(f[..., d, :], 1, axis=d) for d in range(dim)), start=np.zeros(f.shape[:dim]+(2, f.shape[-1]), dtype=np.int64))


def equilibrate(face):
    """Exact integer tree correction on the periodic cell graph, unit voxels."""
    f = face.copy(); residual = integer_divergence(f)
    before = int(np.abs(residual).max())
    shape = f.shape[:3]
    assert int(np.abs(f).max())+int(np.prod(shape))*before < 2**60
    for axis in range(3):
        for location in range(shape[axis]-1, 0, -1):
            child = [slice(None)]*3
            for previous in range(axis): child[previous] = 0
            child[axis] = location
            parent = child.copy(); parent[axis] = location-1
            r = residual[tuple(child)].copy()
            f[tuple(parent)+(slice(None), axis, slice(None))] += r
            residual[tuple(parent)] += r
            residual[tuple(child)] = 0
    final = integer_divergence(f)
    assert np.count_nonzero(final) == 0
    return f, {'max_integer_divergence_before': before, 'max_integer_divergence_after': int(np.abs(final).max()),
        'max_face_change_integer': int(np.abs(f-face).max()), 'denominator': str(DEN),
        'proof': 'All operations are integer additions without overflow; every face is single valued periodically, and every cell divergence is identically zero.'}


def rt0_energy(labels, flux, materials):
    dim = 3; loads = 6; count = labels.size
    grams = [fmpz_mat(12, 12) for unused in materials]
    for d in range(dim):
        plus = flux[..., d, :].reshape(count, 12)
        minus = np.roll(flux[..., d, :], 1, axis=d).reshape(count, 12)
        for values, weight in ((plus+minus, 3), (plus-minus, 1)):
            for p in range(len(materials)):
                grams[p] += weight*gram(values[labels.ravel() == p])
    denominator = 12*count*DEN**2
    h = combine(grams, [x.inv() for x in materials], denominator, loads)
    assert int(np.abs(flux).max())*count < 2**62
    sums = flux.sum(axis=(0, 1, 2), dtype=np.int64).reshape(6, 6)
    mean = arb_mat([[ball(int(sums[i, j]))/(count*DEN) for j in range(6)] for i in range(6)])
    assert not mean.det().contains(0)
    assert spd(h)[0]
    lower = mean*h.inv()*mean.transpose()
    return lower, h, mean, grams, denominator


def p4_symmetry(matrix, dim):
    rotation = np.eye(dim, dtype=int); rotation[:2, :2] = [[0, -1], [1, 0]]
    r = arb_mat(np.kron(np.eye(2, dtype=int), rotation).tolist())
    result = arb_mat(matrix.nrows(), matrix.ncols()); work = matrix
    for unused in range(4):
        result += work; work = r*work*r.transpose()
    return result/4


def trace_product(a, b):
    return sum((a[i, j]*b[j, i] for i in range(a.nrows()) for j in range(a.ncols())), arb(0))


def certificate_ratio(lower, upper, dim):
    """Verified SDP-dual upper witnesses for c/a and -c/a; no trusted float eigensolve."""
    n = 2*dim; gap = mid_array(upper-lower)
    eig, vec = np.linalg.eigh(gap)
    assert min(eig) > 0
    root = (vec*np.sqrt(eig)) @ vec.T
    invroot = (vec/np.sqrt(eig)) @ vec.T
    ha = arb_mat(n, n); ha[0, 0] = ha[1, 1] = ball(Fraction(1, 2))
    hc = arb_mat(n, n)
    hc[0, dim+1] = hc[dim+1, 0] = ball(Fraction(1, 4))
    hc[1, dim] = hc[dim, 1] = ball(Fraction(-1, 4))
    assert trace_product(ha, lower) > 0
    def witness(t, sign):
        h = hc*sign-ha*ball(t)
        ev, vv = np.linalg.eigh(root @ mid_array(h) @ root)
        p = invroot @ ((vv*np.maximum(ev, 0)) @ vv.T) @ invroot
        p = (p+p.T)/2
        for exponent in (40, 34, 28, 22):
            guard = max(float(np.linalg.norm(p, 2)), 1.)*2.**(-exponent)
            pp = arb_mat([[ball(p[i, j])+(ball(guard) if i == j else arb(0)) for j in range(n)] for i in range(n)])
            qq = pp-h
            okp, pivp = spd(pp); okq, pivq = spd(qq)
            if okp and okq:
                value = trace_product(pp, upper)-trace_product(qq, lower)
                return bool(value <= 0), {'t_exact': str(t), 'sign': sign, 'P_ball': dump_matrix(pp), 'Q_ball': dump_matrix(qq),
                    'P_LDL_pivots': pivp, 'Q_LDL_pivots': pivq, 'functional_ball': value.str(60, more=True),
                    'identity': 'P-Q = sign*Hc-t*Ha; P and Q positive definite verified by Arb LDL.'}
        raise RuntimeError('Could not certify a positive semidefinite dual witness')
    certificates = []
    for sign in (1, -1):
        hi = Fraction(16); good, cert = witness(hi, sign)
        while not good:
            hi *= 2; good, cert = witness(hi, sign)
        lo = -hi
        for unused in range(44):
            mid = (lo+hi)/2; good, candidate = witness(mid, sign)
            if good: hi, cert = mid, candidate
            else: lo = mid
        assert witness(hi, sign)[0]
        certificates.append(cert)
    low = -Fraction(certificates[1]['t_exact'])*200
    high = Fraction(certificates[0]['t_exact'])*200
    return low, high, certificates


def endpoint(q, places, upper):
    scaled = ball(q)*10**places
    integer = int((scaled.ceil() if upper else scaled.floor()).fmpz())
    sign = '-' if integer < 0 else ''; integer = abs(integer); scale = 10**places
    return sign+str(integer//scale)+'.'+str(integer % scale).zfill(places)


def integer_dump(grams):
    return [[[str(g[i, j]) for j in range(g.ncols())] for i in range(g.nrows())] for g in grams]


def run(kind, k):
    path = ROOT/f'certified_bounds/verified_{kind}_k{k}.json'
    if path.exists(): print('existing', path.name, flush=True); return
    started = time.perf_counter(); dim = 2 if kind == '2d' else 3
    if dim == 2:
        base, unused = cell(); phases = [PHASES[x] for x in ('p', 'n', 'weak')]
        labels = np.kron(base, np.ones((k, k), int)); assert np.array_equal(labels, np.rot90(labels))
        raw = expose_trial(fem2d.homogenize, 'correction')
        point, u = raw(labels, phases)
        dualpoint, v = raw(labels, inverse_phases(phases))
        qu = quantize(u*len(labels)); qv = quantize(v*len(labels))
        trial = {'labels': labels, 'Q1_primal_nodal_integer': qu, 'Q1_dual_nodal_integer': qv}
        previous = json.loads((ROOT/'certified_bounds/planar_summary.json').read_text())['cases']['baseline']
        symmetry = {'verified_exact': True, 'method': 'integer label equality with rot90'}
    else:
        geo = json.loads((ROOT/'three_d/junction_geometry.json').read_text())
        base = np.array(geo['woodpile']['W2_split_clear']['labels'], dtype=int)
        phases = geo['materials_sigma_S_kappa']; labels = np.kron(base, np.ones((k, k, k), int))
        from three_d.fixed_rasters import exact_group
        symmetry = exact_group(labels)
        assert symmetry['has_C4_z']
        raw = expose_trial(rectangular_voxels.fem, 'u'); point, u = raw(labels, phases)
        rawfv = expose_trial(rectangular_voxels.fv, 'ff'); dualpoint, f = rawfv(labels, phases)
        qu = quantize(u)
        qf, constraints = equilibrate(quantize(f))
        trial = {'labels': labels, 'Q1_primal_nodal_integer': qu, 'RT0_face_integer': qf}
        old = json.loads((ROOT/'three_d/junction_woodpile.json').read_text())['cases']['W2_split_clear']['levels']
        previous = next(x for x in old if x['k'] == k)['continuum_enclosure']
    solve_seconds = time.perf_counter()-started
    print(kind, k, 'trials prepared', solve_seconds, flush=True)
    materials = decimal_materials(phases)
    upper, primal_grams, primal_denominator = q1_energy(labels, qu, materials)
    extra = {}
    if dim == 2:
        dual, dual_grams, dual_denominator = q1_energy(labels, qv, [a.inv() for a in materials])
        rotation = arb_mat(np.kron(np.eye(2, dtype=int), np.array([[0, -1], [1, 0]])).tolist())
        lower = rotation.transpose()*dual.inv()*rotation
        extra.update(dual_energy_ball=dump_matrix(dual), dual_integer_grams=integer_dump(dual_grams), dual_denominator=str(dual_denominator),
            constraints_V={'periodic_nodal_identification': True, 'dual_flux': 'Rotated gradient of periodic conforming Q1 potentials: distributional divergence exactly zero and continuous normal trace; no numerical projection.'})
    else:
        lower, raw_energy, mean, dual_grams, dual_denominator = rt0_energy(labels, qf, materials)
        extra.update(complementary_raw_energy_ball=dump_matrix(raw_energy), flux_mean_ball=dump_matrix(mean),
            flux_mean_determinant_ball=mean.det().str(60, more=True), dual_integer_grams=integer_dump(dual_grams), dual_denominator=str(dual_denominator), constraints_V=constraints)
    upper = p4_symmetry(upper, dim); lower = p4_symmetry(lower, dim)
    ok1, piv1 = spd(lower); ok2, piv2 = spd(upper-lower)
    assert ok1 and ok2, (piv1, piv2)
    low, high, witnesses = certificate_ratio(lower, upper, dim)
    archive = ROOT/f'certified_bounds/trials_{kind}_k{k}.npz'; np.savez_compressed(archive, **trial)
    result = {'classification': 'V: exact dyadic admissible trial fields, exact integer polynomial moments, Arb material/energy/inverse evaluation, and verified semidefinite dual witnesses for the Seebeck ratio.',
        'geometry': 'Fig2b fixed cell(24,15,6,3)' if dim == 2 else 'junction W2_split_clear fixed 24 cubed raster',
        'k': k, 'shape': list(labels.shape), 'phases_sigma_S_kappa_decimal_inputs': [[str(x) for x in p] for p in phases],
        'temperature_K_exact': '300', 'beta_V_K_exact': '1/5000', 'energy_scale_exact': '1/250',
        'dyadic_denominator': str(DEN), 'Arb_precision_bits': ctx.prec, 'python_flint_version': __import__('flint').__version__,
        'symmetry_exact': symmetry, 'trial_archive': archive.relative_to(ROOT).as_posix(), 'trial_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'primal_integer_grams': integer_dump(primal_grams), 'primal_denominator': str(primal_denominator),
        'upper_energy_ball_V': dump_matrix(upper), 'lower_energy_ball_V': dump_matrix(lower),
        'lower_SPD_LDL_pivots_V': piv1, 'gap_SPD_LDL_pivots_V': piv2,
        'Sa_interval_uV_K_V_decimal': [endpoint(low, 8, False), endpoint(high, 8, True)],
        'Sa_interval_uV_K_V_exact_rational': [str(low), str(high)], 'ratio_dual_witnesses_V': witnesses,
        'previous_C_reference': previous, 'primal_solver_P': point, 'dual_solver_P': dualpoint,
        'solver_seconds_P': solve_seconds, 'verification_seconds_P': time.perf_counter()-started-solve_seconds,
        'total_seconds_P': time.perf_counter()-started, **extra}
    save(path.relative_to(ROOT).as_posix(), result)
    print(kind, k, 'V', result['Sa_interval_uV_K_V_decimal'], 'seconds', result['total_seconds_P'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['2d', '3d']); ap.add_argument('--k', type=int, nargs='+', default=[1])
    args = ap.parse_args()
    for level in args.k: run(args.kind, level)
