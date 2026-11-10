"""Checks of the cubic term: weak-reference check, three-dimensional woodpile and mirror counterexample.
Reads common/homog2.py, three_d/junction_geometry.json and weak_contrast/fig2_values.json; writes only in this directory."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import types
from pathlib import Path
import numpy as np
from weak_contrast.c3_fourier_verify import ROOT, cell, geometry, predicted, reference_materials, sqrt_spd, upsample, save


def projection_nd(shape):
    dim = len(shape)
    k = np.meshgrid(*[np.fft.fftfreq(n)*n for n in shape], indexing='ij')
    den = sum(v*v for v in k)
    den[(0,)*dim] = 1
    P = np.empty((*shape, dim, dim))
    for i in range(dim):
        for j in range(dim):
            q = k[i]*k[j]/den
            if i!=j:
                for axis in (i, j):
                    if shape[axis]%2==0:
                        sel = [slice(None)]*dim
                        sel[axis] = shape[axis]//2
                        q[tuple(sel)] = 0
            P[..., i, j] = q
    return P


def geometry_z(labels):
    dim = labels.ndim
    axes = tuple(range(dim))
    P = projection_nd(labels.shape)
    F = [np.fft.ifftn(np.fft.fftn(labels==i)[..., None, None]*P, axes=axes).real
         for i in (0, 1)]
    z = (F[0][..., 0, :]*F[1][..., :, 1]).sum(-1)-(F[0][..., 1, :]*F[1][..., :, 0]).sum(-1)
    return np.array([np.mean(z*(labels==b)) for b in range(3)])


def full_nd(labels, s0, delta, eps):
    dim = labels.ndim
    axes = tuple(range(dim))
    W = sqrt_spd(s0)
    Wi = np.linalg.inv(W)
    D = np.array([Wi@d@Wi for d in delta])[labels]
    P = projection_nd(labels.shape)
    Ebar = np.eye(2*dim).reshape(2, dim, 2*dim)
    E = np.broadcast_to(Ebar, (*labels.shape, 2, dim, 2*dim)).copy()
    for it in range(1000):
        DE = np.einsum('...ab,...bjk->...ajk', D, E)
        h = np.fft.fftn(DE, axes=axes)
        gh = np.einsum('...ij,...ajk->...aik', P, h)
        new = Ebar-eps*np.fft.ifftn(gh, axes=axes).real
        residual = float(np.max(np.abs(new-E)))
        E = new
        if residual<2e-14:
            break
    else:
        raise RuntimeError('No convergence')
    DE = np.einsum('...ab,...bjk->...ajk', D, E)
    Cw = np.eye(2*dim)+eps*DE.mean(axis=axes).reshape(2*dim, 2*dim)
    Wfull = np.kron(W, np.eye(dim))
    C = Wfull@Cw@Wfull
    c = float((C[0, dim+1]-C[1, dim])/2)
    return dict(eps=eps, c=c, c_over_eps3=c/eps**3,
                residual=residual, iterations=it+1)


def main():
    midpoint, delta, s = reference_materials()
    data = json.loads((ROOT/'weak_contrast/numbers.json').read_text())
    sw = s[2]
    W = sqrt_spd(sw)
    Wi = np.linalg.inv(W)
    contrasts = s-sw
    q = contrasts[0]@np.linalg.inv(sw)@contrasts[1]-contrasts[1]@np.linalg.inv(sw)@contrasts[0]
    detscaled = float(np.linalg.det(s[:, [0, 0, 1], [0, 1, 1]]))
    out = dict(phase_determinants=dict(symmetric=detscaled, manuscript_unscaled=300*detscaled,
               q=float(q[0, 1]), det_weak=float(np.linalg.det(sw))),
               contrast_eigenvalues_weak=[np.linalg.eigvalsh(Wi@d@Wi).tolist() for d in contrasts])
    out['geometry_convergence'] = [dict(k=row['k'], G112=2*row['g'][0], G122=2*row['g'][1],
        Gsum=2*row['gsum']) for row in data['geometry']['geometry']]
    out['weak_reference_continuum_prediction'] = predicted(sw, contrasts, np.array(data['extrapolations']['collocation_limit']['g']))
    out['midpoint_reference'] = data['extrapolations']
    source = ROOT/'common/homog2.py'
    H = types.ModuleType('H'); H.__file__ = str(source)
    exec(compile(source.read_text(), str(source), 'exec'), H.__dict__)
    weakruns = []
    for e in [.0003, .0001, -.0001, -.0003]:
        mats = [Wi@(sw+e*d)@Wi for d in contrasts]
        sig, L, _ = H.homogenize(upsample(cell(), 4), mats)
        a = sw[0, 0]*np.trace(sig)/2
        c = W[0, 0]*W[1, 1]*(L[0, 1]-L[1, 0])/2
        weakruns.append(dict(eps=e, c=float(c), Sa_over_eps3_uV_K=float(c/a/e**3*1e6)))
    linear = (3*weakruns[1]['Sa_over_eps3_uV_K']-weakruns[0]['Sa_over_eps3_uV_K'])/2
    prediction = predicted(sw, contrasts, geometry(upsample(cell(), 4)))
    out['weak_reference_fv_k4'] = dict(runs=weakruns, linear_extrapolation=linear,
        prediction=prediction, relative_difference=abs(linear/prediction['Sa3_uV_K']-1))
    src = ROOT/'three_d/junction_geometry.json'
    labels_original = np.array(json.loads(src.read_text())['woodpile']['W2_split_clear']['labels'])
    labels = np.where(labels_original==2, 0, np.where(labels_original==1, 1, 2))
    sref = np.array([[1., .3], [.3, .8]])
    d = np.array([[[.7, -.2], [-.2, .1]], [[-.3, .5], [.5, .9]], np.zeros((2, 2))])
    G = geometry_z(labels)
    runs = [full_nd(labels, sref, d, e) for e in [.02, .01, .005]]
    out['woodpile_3d'] = dict(shape=list(labels.shape), G=G.tolist(),
        prediction=predicted(sref, d, G/2), runs=runs,
        mirror_x=full_nd(labels[::-1], sref, d, .01))
    extruded = np.repeat(cell()[..., None], 24, axis=2)
    out['mirror_3d_counterexample'] = dict(
        description='A z-independent pinwheel has an xy mirror but nonzero G_z.',
        G=geometry_z(extruded).tolist(), mirror_z=geometry_z(extruded[:, :, ::-1]).tolist(),
        mirror_x=geometry_z(extruded[::-1]).tolist())
    figpath = ROOT/'weak_contrast/fig2_values.json'
    out['full_contrast_existing_figure_data'] = dict(source=figpath.relative_to(RELEASE_ROOT).as_posix(), data=json.loads(figpath.read_text()))
    save('weak_contrast/c3_fourier_checks.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ['midpoint_reference','geometry_convergence']}, indent=2))


if __name__=='__main__':
    main()
