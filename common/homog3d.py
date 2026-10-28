"""3D periodic FV homogenisation of linear coupled thermoelectric transport, voxel phases (isotropic each).
Symmetric formulation: fields (V, T); local s = [[sig, sig*S], [sig*S, (k + T0 sig S^2)/T0]] (= D^-1 m, D=diag(1,T0)).
Face coefficient = matrix harmonic mean 2 b (a+b)^-1 a. Returns sigma_eff (3x3), L_eff (3x3), S_eff = sigma^-1 L."""

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
def hmean(a, b): return 2*b @ np.linalg.solve(a + b, a)

def homogenize3d(lab, mats):
    Nx, Ny, Nz = lab.shape; n = lab.size
    M = np.array(mats)                                   # nph,2,2
    idx = np.arange(n).reshape(lab.shape)
    # precompute pairwise face matrices
    nph = len(mats); HM = np.zeros((nph, nph, 2, 2))
    for i in range(nph):
        for j in range(nph): HM[i, j] = hmean(M[i], M[j])
    faces = []
    for d in range(3):
        q = np.roll(idx, -1, axis=d).ravel(); p = idx.ravel()
        mf = HM[lab.ravel(), np.roll(lab, -1, axis=d).ravel()]   # n,2,2
        faces.append((p, q, mf))
    rows, cols, vals = [], [], []
    for (p, q, mf) in faces:
        for a in range(2):
            for b in range(2):
                v = mf[:, a, b]
                rows += [2*p+a, 2*p+a, 2*q+a, 2*q+a]; cols += [2*p+b, 2*q+b, 2*q+b, 2*p+b]; vals += [v, -v, v, -v]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2*n, 2*n))
    keep = np.arange(2, 2*n); Kk = K[keep][:, keep].tocsc()
    lu = spla.splu(Kk, permc_spec='COLAMD')
    Meff = np.zeros((2, 3, 2, 3))
    for fin in range(2):
        for din in range(3):
            g = np.zeros(2); g[fin] = 1.0
            pp, qq, mf = faces[din]; mg = mf @ g
            rhs = np.zeros(2*n)
            np.add.at(rhs, (2*pp[:, None] + np.arange(2)).ravel(), mg.ravel())
            np.add.at(rhs, (2*qq[:, None] + np.arange(2)).ravel(), -mg.ravel())
            phi = np.zeros(2*n); phi[keep] = lu.solve(rhs[keep]); phi = phi.reshape(n, 2)
            for dout, (p2, q2, mf2) in enumerate(faces):
                dPhi = phi[q2] - phi[p2] + (g if dout == din else 0)
                Meff[:, dout, fin, din] = np.einsum('kab,kb->ka', mf2, dPhi).mean(0)
    sig = Meff[0, :, 0, :]; L = Meff[0, :, 1, :]
    S = np.linalg.solve(sig, L)
    onsager = np.abs(Meff[1, :, 0, :].T - Meff[0, :, 1, :]).max()/np.abs(Meff[0, :, 1, :]).max()   # s symmetric: C_TV^T = C_VT
    return sig, L, S, onsager

def axial(S):
    """axial vector of antisymmetric part: a_i = 1/2 eps_ijk S_jk  -> (S_yz - S_zy)/2, (S_zx - S_xz)/2, (S_xy - S_yx)/2"""
    return np.array([(S[1,2]-S[2,1])/2, (S[2,0]-S[0,2])/2, (S[0,1]-S[1,0])/2])
