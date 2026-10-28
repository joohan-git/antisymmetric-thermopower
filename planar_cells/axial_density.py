"""Exact local decomposition of the axial coefficient (2D, symmetric s formulation, FV faces).
Discrete energy identity: C_eff[a,i;b,j] = (1/n) sum_faces dPhi^{(a,i)} . m_f . dPhi^{(b,j)} (symmetric s, conforming cell problems).
Axial density on faces -> cells: d = (C^{VT}_{xy} - C^{VT}_{yx})/2 contributions; S_a = <d>/sigma_eff (sigma_eff isotropic in p4)."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import numpy as np; import scipy.sparse as sp; import scipy.sparse.linalg as spla
T0 = 300.0
def sloc(sig, S, k): return np.array([[sig, sig*S], [sig*S, (k + T0*sig*S*S)/T0]])
def hm(a, b): return 2*b @ np.linalg.solve(a + b, a)
def solve(lab, mats):
    N = lab.shape[0]; n = N*N; M = np.array(mats); nph = len(mats)
    HM = np.array([[hm(M[i], M[j]) for j in range(nph)] for i in range(nph)])
    idx = np.arange(n).reshape(N, N); faces = []
    for d in range(2):
        faces.append((idx.ravel(), np.roll(idx, -1, axis=d).ravel(), HM[lab.ravel(), np.roll(lab, -1, axis=d).ravel()]))
    rows, cols, vals = [], [], []
    for p, q, mf in faces:
        for a in range(2):
            for b in range(2):
                v = mf[:, a, b]; rows += [2*p+a, 2*p+a, 2*q+a, 2*q+a]; cols += [2*p+b, 2*q+b, 2*q+b, 2*p+b]; vals += [v, -v, v, -v]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2*n, 2*n))
    keep = np.arange(2, 2*n); lu = spla.splu(K[keep][:, keep].tocsc())
    dPhi = {}
    for fin in range(2):
        for din in range(2):
            g = np.zeros(2); g[fin] = 1; p, q, mf = faces[din]; mg = mf @ g; rhs = np.zeros(2*n)
            np.add.at(rhs, (2*p[:, None]+np.arange(2)).ravel(), mg.ravel()); np.add.at(rhs, (2*q[:, None]+np.arange(2)).ravel(), -mg.ravel())
            phi = np.zeros(2*n); phi[keep] = lu.solve(rhs[keep]); phi = phi.reshape(n, 2)
            dPhi[(fin, din)] = [phi[q2]-phi[p2]+(g if d == din else 0) for d, (p2, q2, _) in enumerate(faces)]
    def C(a, i, b, j, local=False):
        dens = sum(np.einsum('ka,kab,kb->k', dPhi[(a, i)][d], faces[d][2], dPhi[(b, j)][d]) for d in range(2))
        return dens.reshape(N, N) if local else dens.mean()
    sig = np.array([[C(0, i, 0, j) for j in range(2)] for i in range(2)])
    L = np.array([[C(0, i, 1, j) for j in range(2)] for i in range(2)])
    dens_ax = 0.5*(C(0, 0, 1, 1, True) - C(0, 1, 1, 0, True))
    return sig, L, np.linalg.solve(sig, L), dens_ax
