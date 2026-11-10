"""Layered thermoelectric cells, transport enclosures and finite-sample checks."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import time
import types
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '2'
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    def threadpool_limits(limits):
        return nullcontext()
from common import fem3d as FE

ROOT = RELEASE_ROOT
MAIN = ROOT
PHASES = [(10.45, 10e-6, .9012), (1e5, -180e-6, 1.3), (1e5, 200e-6, 1.2)]


def save(name, value):
    path = ROOT/name
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+chr(10)
    tmp = path.with_name(path.name+'.'+str(time.time_ns())+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    for attempt in range(15):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 14:
                raise
            time.sleep(.15*(attempt+1))


def load_main():
    modules = []
    for name, path in [('homog3d', MAIN/'common/homog3d.py'),
                       ('iterative_fv', MAIN/'common/homog3d_it.py'),
                       ('design_families', MAIN/'planar_cells/families.py')]:
        mod = types.ModuleType(name)
        mod.__file__ = str(path)
        sys.modules[name] = mod
        exec(compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec'), mod.__dict__)
        modules.append(mod)
    return modules


H3, IT, F = load_main()


def source_hashes():
    names = ['common/homog3d.py', 'common/homog3d_it.py',
             'three_d/run3d_helix.py', 'three_d/run3d_checks.py', 'three_d/woodpile.py', 'planar_cells/families.py']
    return {name: hashlib.sha256((MAIN/name).read_bytes()).hexdigest() for name in names}


def cases():
    a = F.F3_pinwheel_tiling(16, .6412, .9)
    shift = lambda t: np.roll(np.roll(t, 8, axis=0), 8, axis=1)
    t2 = np.concatenate((np.repeat(a[:,:,None],4,2), np.repeat(shift(a)[:,:,None],4,2)), axis=2)
    glide = np.concatenate((np.repeat(a[:,:,None],4,2), np.repeat(shift(a[::-1,:])[:,:,None],4,2)), axis=2)
    rng = np.random.default_rng(7)
    field = gaussian_filter(rng.standard_normal((16,16)), 2., mode='wrap')
    layer = np.digitize(field, np.quantile(field,[1/3,2/3]))
    helix = lambda lay: np.concatenate([np.repeat(np.rot90(lay,k)[:,:,None],4,2) for k in range(4)], axis=2)
    h = helix(layer)
    h2 = helix((field > np.median(field)).astype(int))
    return {'A_extruded': (np.repeat(a[:,:,None],4,2), PHASES),
            'T2_staggered': (t2, PHASES), 'helix_41': (h, PHASES),
            'helix_mirror_x': (h[::-1,:,:].copy(), PHASES),
            'two_phase_pn': (h2, PHASES[1:]),
            'two_phase_pCPC': (h2, [PHASES[0], PHASES[2]]),
            'glide': (glide, PHASES)}


def fv_with_complement(labels, phases, tol=2e-11):
    """Unchanged main FV face/assembly, scaled fields and our CG driver.

    Conservative FV face fluxes define RT0 H(div) continuum trial fluxes.
    The raw FV effective matrix is NOT taken as a continuum lower bound.
    """
    start = time.perf_counter()
    coeff = np.array([FE.energy(p) for p in phases])
    mat, faces = IT.assemble(labels, coeff)
    n = labels.size
    rhs = np.zeros((2*n, 6))
    for din, (p, q, mf) in enumerate(faces):
        for fin in range(2):
            j = 3*fin+din
            np.add.at(rhs[:,j], (2*p[:,None]+np.arange(2)).ravel(), mf[:,:,fin].ravel())
            np.add.at(rhs[:,j], (2*q[:,None]+np.arange(2)).ravel(), -mf[:,:,fin].ravel())
    u = np.zeros_like(rhs)
    u[2:], stats = FE.solve_spd(mat[2:,2:], rhs[2:], np.arange(2,2*n)%2, tol)
    u = u.reshape(n,2,6)
    flux = np.empty((n,2,3,6))
    for d, (p,q,mf) in enumerate(faces):
        g = u[q]-u[p]
        g[:,0,d] += 1
        g[:,1,3+d] += 1
        flux[:,:,d,:] = np.einsum('nab,nbj->naj', mf,g)
    c = flux.mean(axis=0).reshape(6,6)
    result = FE.properties((c+c.T)/2)
    result.update({'classification': 'P: main finite-volume uniform-grid point estimate',
                   'shape': list(labels.shape), 'solver': stats,
                   'reciprocity_relative': float(np.linalg.norm(c-c.T)/np.linalg.norm(c))})
    ff = flux.reshape(tuple(labels.shape)+(2,3,6))
    div = sum(ff[:,:,:,:,d,:]-np.roll(ff[:,:,:,:,d,:],1,axis=d) for d in range(3))
    before = np.linalg.norm(div)/max(np.linalg.norm(ff),1e-30)
    # Periodic discrete Hodge projection removes iterative residual. Each face
    # remains single valued, and each cell's two flux divergences vanish.
    lam = np.zeros(labels.shape)
    for d, size in enumerate(labels.shape):
        shp = [1,1,1]
        shp[d] = size
        lam += (2-2*np.cos(2*np.pi*np.arange(size)/size)).reshape(shp)
    safe = lam.copy()
    safe[0,0,0] = 1
    phat = -np.fft.fftn(div,axes=(0,1,2))/safe[:,:,:,None,None]
    phat[0,0,0] = 0
    psi = np.fft.ifftn(phat,axes=(0,1,2)).real
    for d in range(3):
        ff[:,:,:,:,d,:] -= np.roll(psi,-1,axis=d)-psi
    after_div = sum(ff[:,:,:,:,d,:]-np.roll(ff[:,:,:,:,d,:],1,axis=d) for d in range(3))
    after = np.linalg.norm(after_div)/max(np.linalg.norm(ff),1e-30)
    inva = np.linalg.inv(coeff)[labels.ravel()]
    raw = np.zeros((6,6))
    for d in range(3):
        fp = ff[:,:,:,:,d,:].reshape(n,2,6)
        fm = np.roll(ff[:,:,:,:,d,:],1,axis=d).reshape(n,2,6)
        raw += (np.einsum('nai,nab,nbj->ij',fm,inva,fm)
                +np.einsum('nai,nab,nbj->ij',fp,inva,fp))/(3*n)
        cross = np.einsum('nai,nab,nbj->ij',fm,inva,fp)/(6*n)
        raw += cross+cross.T
    mean = ff.mean(axis=(0,1,2)).reshape(6,6)
    inverse_mean = np.linalg.inv(mean)
    resistance_upper = inverse_mean.T@raw@inverse_mean
    lower = np.linalg.inv(resistance_upper)
    result.update({'complementary_resistance_upper': resistance_upper.tolist(),
                   'continuum_energy_lower': ((lower+lower.T)/2).tolist(),
                   'RT0_divergence_relative_before_projection': float(before),
                   'RT0_divergence_relative_after_projection': float(after),
                   'seconds': time.perf_counter()-start})
    return result


def enclosure(lower, upper, p4=True):
    l = (np.asarray(lower)+np.asarray(lower).T)/2
    u = (np.asarray(upper)+np.asarray(upper).T)/2
    # All selected structures have macroscopic fourfold invariance (a screw
    # translation is immaterial to a periodic effective tensor).
    if p4:
        rot = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        r = np.kron(np.eye(2),rot)
        l = sum(np.linalg.matrix_power(r,k)@l@np.linalg.matrix_power(r,k).T for k in range(4))/4
        u = sum(np.linalg.matrix_power(r,k)@u@np.linalg.matrix_power(r,k).T for k in range(4))/4
    raw_minimum = float(np.linalg.eigvalsh(u-l).min())
    if raw_minimum < -1e-8:
        raise RuntimeError(('invalid bound ordering', raw_minimum))
    # Small outward guard for floating-point subtraction/inversion; this is
    # not a replacement for interval arithmetic or directed rounding.
    guard = 5e-12*max(float(np.linalg.norm(u,2)),1.)
    l = l-guard*np.eye(6)
    u = u+guard*np.eye(6)
    vals, vecs = np.linalg.eigh(u-l)
    root = (vecs*np.sqrt(np.maximum(vals,0)))@vecs.T
    def maximum(h):
        return float(np.sum(h*l.T)+np.maximum(np.linalg.eigvalsh(root@h@root),0).sum())
    def ratio(h, den):
        def solve(sign):
            fun = lambda t: maximum(sign*h-t*den)
            left,right = -1.,1.
            while fun(left) < 0:
                left *= 2
            while fun(right) > 0:
                right *= 2
            return brentq(fun,left,right,xtol=1e-13)
        return [-solve(-1)*FE.BETA*1e6, solve(1)*FE.BETA*1e6]
    den = np.zeros((6,6)); den[0,0]=den[1,1]=.5
    denz = np.zeros((6,6)); denz[2,2]=1.
    ha = np.zeros((6,6)); ha[0,4]=ha[4,0]=.25; ha[1,3]=ha[3,1]=-.25
    hx = np.zeros((6,6)); hx[0,3]=hx[3,0]=hx[1,4]=hx[4,1]=.25
    hz = np.zeros((6,6)); hz[2,5]=hz[5,2]=.5
    return {'classification': 'C: continuum variational enclosure, floating-point evaluation (not directed rounding)',
            'method': 'conforming Q1 primal energy and H(div) RT0 complementary energy; p4 symmetrized',
            'lower_energy': l.tolist(), 'upper_energy': u.tolist(),
            'raw_minimum_upper_minus_lower_eigenvalue': raw_minimum,
            'outward_energy_roundoff_guard': guard,
            'minimum_upper_minus_lower_eigenvalue': float(vals.min()),
            'relative_energy_gap': float(np.linalg.norm(u-l)/np.linalg.norm(u)),
            'Sa_z_interval_uV_K': ratio(ha,den), 'Sxx_interval_uV_K': ratio(hx,den),
            'Szz_interval_uV_K': ratio(hz,denz),
            'sigma_xx_interval': [float(np.trace(den@v)*1e5) for v in (l,u)],
            'sigma_zz_interval': [float(v[2,2]*1e5) for v in (l,u)]}


def validate():
    out = {'source_sha256_before': source_hashes(), 'laminates': []}
    for axis in range(3):
        lab = np.zeros((4,4,4),int)
        sl = [slice(None)]*3; sl[axis] = slice(2,None); lab[tuple(sl)] = 1
        phases = [PHASES[0],PHASES[2]]
        coeff = np.array([FE.energy(p) for p in phases])
        parallel = coeff.mean(0)
        serial = np.linalg.inv(np.linalg.inv(coeff).mean(0))
        exact = np.zeros((6,6))
        for d in range(3):
            ix = [d,3+d]
            exact[np.ix_(ix,ix)] = serial if d == axis else parallel
        fem = FE.homogenize(lab,phases)
        fv = fv_with_complement(lab,phases)
        lo = np.array(fv['continuum_energy_lower'])
        record = {'normal_axis': axis, 'fem': fem, 'fv': fv,
                  'exact_scaled_energy': exact.tolist(),
                  'fem_relative_error': float(np.linalg.norm(np.array(fem['scaled_energy'])-exact)/np.linalg.norm(exact)),
                  'fv_relative_error': float(np.linalg.norm(np.array(fv['scaled_energy'])-exact)/np.linalg.norm(exact)),
                  'RT0_lower_relative_error': float(np.linalg.norm(lo-exact)/np.linalg.norm(exact))}
        assert max(record[x] for x in ['fem_relative_error','fv_relative_error','RT0_lower_relative_error']) < 2e-8
        out['laminates'].append(record)
        print('laminate',axis,'errors',record['fem_relative_error'],record['RT0_lower_relative_error'],flush=True)
    from common.fem2d import homogenize as fem2
    out['extrusion_checks'] = []
    for size in (16,64):
        a = F.F3_pinwheel_tiling(size,.6412,.9)
        two = fem2(a,PHASES)
        three = FE.homogenize(np.repeat(a[:,:,None],4,2),PHASES)
        ix = [0,1,3,4]
        err = np.linalg.norm(np.array(three['scaled_energy'])[np.ix_(ix,ix)]-np.array(two['scaled_effective_energy']))/np.linalg.norm(two['scaled_effective_energy'])
        out['extrusion_checks'].append({'base_N': size, 'FEM_2D': two, 'FEM_3D': three, 'xy_energy_relative_error': float(err)})
        assert err < 2e-8
        print('extrusion',size,'Sa',three['Sa_z_uV_K'],'error',err,flush=True)
        save('validation/fem3d_validation.json',out)
    out['original_direct_FV_crosschecks'] = []
    for name in ('A_extruded','T2_staggered','helix_41'):
        lab,phases = cases()[name]
        sig,alpha,s,ons = H3.homogenize3d(lab,[H3.sloc(*p) for p in phases])
        rec = fv_with_complement(lab,phases)
        err = np.linalg.norm(s*1e6-np.array(rec['S_uV_K']))
        out['original_direct_FV_crosschecks'].append({'case': name, 'direct_S_uV_K': (s*1e6).tolist(),
                   'adapter_S_uV_K': rec['S_uV_K'], 'absolute_error_Frobenius_uV_K': float(err)})
        assert err < 1e-6
        print('main direct agreement',name,err,flush=True)
    out['source_sha256_after'] = source_hashes()
    assert out['source_sha256_before'] == out['source_sha256_after']
    save('validation/fem3d_validation.json',out)


def periodic(names, levels):
    path = ROOT/'three_d/layered_periodic.json'
    out = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'source_sha256_before': source_hashes(), 'cases': {}}
    for name in names:
        base,phases = cases()[name]
        rec = out['cases'].setdefault(name,{'base_shape': list(base.shape), 'base_labels': base.tolist(),
                          'base_label_sha256': hashlib.sha256(base.astype(np.int32).tobytes()).hexdigest(),
                          'phases_sigma_S_kappa': phases, 'levels': []})
        for k in levels:
            if any(item['k'] == k for item in rec['levels']):
                continue
            lab = np.kron(base,np.ones((k,k,k),int))
            print('start',name,'k',k,'shape',lab.shape,flush=True)
            fem = FE.homogenize(lab,phases)
            print('FEM',name,k,fem['Sa_z_uV_K'],fem['seconds'],flush=True)
            fv = fv_with_complement(lab,phases)
            bound = enclosure(fv['continuum_energy_lower'],fem['scaled_energy'])
            rec['levels'].append({'k': k, 'fem': fem, 'fv': fv, 'continuum_enclosure': bound})
            out['source_sha256_after'] = source_hashes()
            assert out['source_sha256_after'] == out['source_sha256_before']
            save('three_d/layered_periodic.json',out)
            print('done',name,k,'FV',fv['Sa_z_uV_K'],'C',bound['Sa_z_interval_uV_K'],flush=True)


def extrusion_enriched():
    from certified_bounds.corner_sectors import sector_pair
    a = F.F3_pinwheel_tiling(16,.6412,.9)
    out = {'classification': 'C: exact extrusion reduction of two-dimensional primal/dual corner-enriched enclosure', 'cases': []}
    for n in (16,64):
        if n == 16:
            pair = sector_pair(a,PHASES,m=4,nr=3,radius=.22,ngauss=10)
            enc = pair['enclosure']
            provenance = 'new corner-enriched 2D solve on the exact selected N=16 raster'
        else:
            old = json.loads((ROOT/'certified_bounds/design_enclosures.json').read_text(encoding='utf-8'))
            old_case = old['cases']['A_64']
            new = F.F3_pinwheel_tiling(64,.6412,.9)
            assert np.array_equal(new,np.array(old_case['labels']))
            assert np.array_equal(np.asarray(old_case['phases']),np.asarray(PHASES))
            pair = old_case['sector'][-1]
            enc = pair['enclosure']
            a = new
            provenance = 'reused certified_bounds/design_enclosures.json A_64 final sector enclosure; labels checked byte-for-byte numerically'
        lo = np.zeros((6,6));up = lo.copy()
        xy = [0,1,3,4];zz = [2,5]
        lo[np.ix_(xy,xy)] = np.array(enc['lower_energy'])
        up[np.ix_(xy,xy)] = np.array(enc['upper_energy'])
        az = np.mean(np.array([FE.energy(p) for p in PHASES])[a.ravel()],axis=0)
        lo[np.ix_(zz,zz)] = up[np.ix_(zz,zz)] = az
        bounds = enclosure(lo,up)
        bounds['method'] = 'exact extrusion of 2D corner-enriched primal/dual enclosure; z block is the arithmetic phase energy'
        out['cases'].append({'base_N': n, 'provenance': provenance, 'two_dimensional': pair, 'three_dimensional_enclosure': bounds})
        save('three_d/extrusion_enclosures.json',out)
        print('enriched extrusion',n,bounds['Sa_z_interval_uV_K'],flush=True)


def finite_samples(levels):
    path = ROOT/'devices/finite_extrusion.json'
    out = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'cell_counts': [8,4,2], 'cell_dimensions_mm': [1.,1.,.25], 'sample_dimensions_mm': [8.,4.,.5], 'thermal_axis': 'y', 'DeltaT_K': 1., 'runs': []}
    base,phases = cases()['A_extruded']
    per = json.loads((ROOT/'three_d/layered_periodic.json').read_text(encoding='utf-8'))['cases']['A_extruded']['levels']
    enriched = json.loads((ROOT/'three_d/extrusion_enclosures.json').read_text(encoding='utf-8'))['cases'][0]['three_dimensional_enclosure']['Sa_z_interval_uV_K']
    for k in levels:
        if any(item['k'] == k for item in out['runs']):
            continue
        lab = np.kron(np.tile(base,(8,4,2)),np.ones((k,k,k),int))
        print('finite start',k,lab.shape,flush=True)
        result = FE.finite(lab,phases,electrode_y_width=4*k,electrode_z_width=4*k)
        sa = next(x['fem']['Sa_z_uV_K'] for x in per if x['k'] == k)
        result['ratio_to_same_grid_periodic_FEM'] = result['V_left_minus_right_uV']/(2*sa)
        result['ratio_to_continuum_A16_Sa_interval'] = sorted([result['V_left_minus_right_uV']/(2*s) for s in enriched])
        result['k'] = k
        out['runs'].append(result)
        save('devices/finite_extrusion.json',out)
        print('finite done',k,result['V_left_minus_right_uV'],result['ratio_to_same_grid_periodic_FEM'],flush=True)


def finite_controls():
    path = ROOT/'devices/extrusion_controls.json'
    out = {'classification': 'P: uniform-grid finite-sample controls', 'runs': {}}
    enriched = json.loads((ROOT/'three_d/extrusion_enclosures.json').read_text(encoding='utf-8'))['cases'][0]['three_dimensional_enclosure']
    mid = (np.array(enriched['lower_energy'])+np.array(enriched['upper_energy']))/2
    sa = FE.properties(mid)['Sa_z_uV_K']
    shape = (128,64,8)
    # Homogeneous tensor is a midpoint representative, not itself a certified
    # exact tensor. This tests electrodes and the voltage normalization.
    out['homogeneous_midpoint_tensor'] = FE.properties(mid)
    for name,wy,wz in [('homogeneous_point_probe',None,None), ('homogeneous_short_patch',4,4)]:
        result = FE.finite(np.zeros(shape,int),[mid],wy,wz)
        result['ratio_to_Sa_DeltaT_W_over_H'] = result['V_left_minus_right_uV']/(2*sa)
        out['runs'][name] = result
        print(name,result['V_left_minus_right_uV'],result['ratio_to_Sa_DeltaT_W_over_H'],flush=True)
        save(path.relative_to(ROOT).as_posix(),out)
    base,phases = cases()['A_extruded']
    lab = np.tile(base,(8,4,2))
    for name,labels,wy,wz in [('microstructure_point_probe',lab,None,None),
                             ('microstructure_mirror_short_patch',lab[::-1,:,:].copy(),4,4)]:
        result = FE.finite(labels,phases,wy,wz)
        out['runs'][name] = result
        print(name,result['V_left_minus_right_uV'],flush=True)
        save(path.relative_to(ROOT).as_posix(),out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['validate','periodic','extrusion','finite','finite_controls'])
    parser.add_argument('--cases', nargs='+', default=list(cases()))
    parser.add_argument('--levels', nargs='+', type=int, default=[1,2])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == 'validate': validate()
        if args.stage == 'periodic': periodic(args.cases,args.levels)
        if args.stage == 'extrusion': extrusion_enriched()
        if args.stage == 'finite': finite_samples(args.levels)
        if args.stage == 'finite_controls': finite_controls()
