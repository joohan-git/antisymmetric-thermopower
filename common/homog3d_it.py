"""3D periodic FV homogenisation, symmetric scaled formulation, iterative solver (CG + smoothed-aggregation AMG).
Same discretisation as common/homog3d.py (face matrix harmonic mean); one cell pinned; symmetric diagonal scaling."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import numpy as np; import scipy.sparse as sp; import scipy.sparse.linalg as spla; import pyamg
from common.homog3d import sloc, hmean, axial, T0

def assemble(lab, mats):
    M = np.array(mats); nph = len(mats); n = lab.size
    HM = np.array([[hmean(M[i], M[j]) for j in range(nph)] for i in range(nph)])
    idx = np.arange(n).reshape(lab.shape); faces = []
    for d in range(3):
        faces.append((idx.ravel(), np.roll(idx, -1, axis=d).ravel(), HM[lab.ravel(), np.roll(lab, -1, axis=d).ravel()]))
    rows, cols, vals = [], [], []
    for p, q, mf in faces:
        for a in range(2):
            for b in range(2):
                v = mf[:, a, b]; rows += [2*p+a, 2*p+a, 2*q+a, 2*q+a]; cols += [2*p+b, 2*q+b, 2*q+b, 2*p+b]; vals += [v, -v, v, -v]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2*n, 2*n))
    return K, faces

def homogenize3d_it(lab, mats, tol=1e-11, verbose=False):
    n = lab.size; K, faces = assemble(lab, mats)
    keep = np.arange(2, 2*n); Kk = K[keep][:, keep].tocsr()
    dsc = 1.0/np.sqrt(Kk.diagonal()); Dm = sp.diags(dsc); A = (Dm @ Kk @ Dm).tocsr()
    B = np.zeros((A.shape[0], 2)); B[0::2, 0] = 1; B[1::2, 1] = 1        # near-nullspace: constant V and constant T (after pin shift)
    B = B / dsc[:, None]
    rng_state = np.random.get_state(); np.random.seed(0)                   # PyAMG draws from the global RNG; fixed for reproducibility
    ml = pyamg.smoothed_aggregation_solver(A, B=B, max_coarse=2000)
    np.random.set_state(rng_state)
    Mpre = ml.aspreconditioner(cycle='V')
    Meff = np.zeros((2, 3, 2, 3)); its = []
    for fin in range(2):
        for din in range(3):
            g = np.zeros(2); g[fin] = 1.0; pp, qq, mf = faces[din]; mg = mf @ g; rhs = np.zeros(2*n)
            np.add.at(rhs, (2*pp[:, None]+np.arange(2)).ravel(), mg.ravel()); np.add.at(rhs, (2*qq[:, None]+np.arange(2)).ravel(), -mg.ravel())
            b = dsc * rhs[keep]; cnt = [0]
            y, info = spla.cg(A, b, rtol=tol, maxiter=3000, M=Mpre, callback=lambda xk: cnt.__setitem__(0, cnt[0]+1))
            its.append((cnt[0], info))
            if info != 0 or not np.all(np.isfinite(y)):
                raise RuntimeError(f'CG did not converge (info={info}) for field {fin}, direction {din}')
            phi = np.zeros(2*n); phi[keep] = dsc*y; phi = phi.reshape(n, 2)
            for dout, (p2, q2, mf2) in enumerate(faces):
                dPhi = phi[q2] - phi[p2] + (g if dout == din else 0)
                Meff[:, dout, fin, din] = np.einsum('kab,kb->ka', mf2, dPhi).mean(0)
    sig = Meff[0, :, 0, :]; L = Meff[0, :, 1, :]; S = np.linalg.solve(sig, L)
    ons = np.abs(Meff[1, :, 0, :].T - Meff[0, :, 1, :]).max()/np.abs(Meff[0, :, 1, :]).max()
    if verbose: print('CG iterations/info', its)
    return sig, L, S, ons, its
