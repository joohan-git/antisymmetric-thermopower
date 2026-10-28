"""Brute-force algebra and normalization checks for c3_fourier_verify.py."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import numpy as np
from weak_contrast.c3_fourier_verify import ROOT, cell, geometry, predicted, projectors, reference_materials, sqrt_spd, save


def direct_coefficients(labels, s0, delta):
    N = len(labels)
    k = np.fft.fftfreq(N)*N
    x, y = np.meshgrid(k, k, indexing='ij')
    den = x*x+y*y
    den[0, 0] = 1
    P = np.array([[x*x/den, x*y/den], [x*y/den, y*y/den]])
    r = np.linalg.inv(s0)
    D = delta[labels]
    E = np.broadcast_to(np.eye(4).reshape(2, 2, 4), (N, N, 2, 2, 4)).copy()
    coeff = []
    for order in range(1, 4):
        DE = np.einsum('xyab,xybjk->xyajk', D, E)
        C = DE.mean(axis=(0, 1)).reshape(4, 4)
        coeff.append(float((C[0, 3]-C[1, 2])/2))
        h = np.fft.fft2(DE, axes=(0, 1))
        gh = np.einsum('ab,ijxy,xybjk->xyaik', r, P, h)
        E = -np.fft.ifft2(gh, axes=(0, 1)).real
    return coeff


def brute_geometry(labels):
    N = len(labels)
    hats = np.array([np.fft.fft2(labels==a)/N**2 for a in range(3)])
    k = np.fft.fftfreq(N)*N
    P = {}
    for i in range(N):
        for j in range(N):
            v = np.array([k[i], k[j]])
            P[i, j] = np.outer(v, v)/(v@v) if v@v else np.zeros((2, 2))
    out = np.zeros(3, complex)
    for i in range(N):
        for j in range(N):
            for m in range(N):
                for n in range(N):
                    prod = P[i, j]@P[m, n]
                    axial = (prod[0, 1]-prod[1, 0])/2
                    for b in range(3):
                        out[b] += hats[0, -i % N, -j % N]*hats[b, (i-m) % N, (j-n) % N]*hats[1, m, n]*axial
    return out


def run():
    rng = np.random.default_rng(5871)
    labels = rng.integers(0, 3, size=(7, 7))
    g = geometry(labels)
    brute = brute_geometry(labels)
    out = dict(brute_geometry=g.tolist(), brute_double_sum_real=brute.real.tolist(),
               brute_double_sum_imag=brute.imag.tolist(),
               double_sum_error=float(np.max(np.abs(g-brute))), random_materials=[])
    for _ in range(12):
        R = rng.normal(size=(2, 2))
        s0 = R@R.T+np.eye(2)
        delta = rng.normal(size=(3, 2, 2))
        delta = (delta+delta.transpose(0, 2, 1))/2
        c = direct_coefficients(labels, s0, delta)
        pred = predicted(s0, delta, g)['c3']
        shift = rng.normal(size=(2, 2))
        shift = (shift+shift.T)/2
        shifted = direct_coefficients(labels, s0, delta+shift)[2]
        perm = np.array([2, 0, 1])
        permuted = predicted(s0, delta[np.argsort(perm)], geometry(perm[labels]))['c3']
        out['random_materials'].append(dict(direct=c, formula=pred,
            error=float(abs(c[2]-pred)), shift_error=float(abs(c[2]-shifted)),
            relabel_error=float(abs(pred-permuted))))
    s0, delta, _ = reference_materials()
    Wi = np.linalg.inv(sqrt_spd(s0))
    eig = [np.linalg.eigvalsh(Wi@d@Wi).tolist() for d in delta]
    out['reference_whitened_contrast_eigenvalues'] = eig
    out['reference_contraction_bound'] = float(np.max(np.abs(eig)))
    assert out['double_sum_error']<1e-14
    assert max(r['error'] for r in out['random_materials'])<1e-14
    assert max(r['shift_error'] for r in out['random_materials'])<1e-14
    assert max(r['relabel_error'] for r in out['random_materials'])<1e-14
    save('weak_contrast/algebra_numbers.json', out)
    print(json.dumps(out, indent=2))


if __name__=='__main__':
    run()
