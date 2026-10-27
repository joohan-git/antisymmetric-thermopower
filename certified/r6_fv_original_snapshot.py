"""Clean periodic FV homogenization for 2-field (V,T) linear thermoelectric transport, 2D pixels, isotropic phases.
Flux through face p->q (direction d): F = -m_f (Phi_q - Phi_p) where Phi = phi + G x. Conservation: sum of outgoing F = 0."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
T0 = 300.0
def mloc(s, S, k): return np.array([[s, s*S], [T0*s*S, k + T0*s*S*S]])
def hmean(a, b): return 2 * b @ np.linalg.solve(a + b, a)

def homogenize(ph, mats):
    N = ph.shape[0]; n = N*N
    Mp = np.array(mats)[ph]                      # N,N,2,2
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    p = (I*N + J).ravel()
    faces = []
    for d, (di, dj) in enumerate([(1, 0), (0, 1)]):
        q = (((I+di) % N)*N + (J+dj) % N).ravel()
        A = Mp.reshape(n, 2, 2); B = Mp[(I+di) % N, (J+dj) % N].reshape(n, 2, 2)
        mf = np.array([hmean(a, b) for a, b in zip(A, B)])
        faces.append((p, q, mf))
    # assemble K: outgoing flux from p via face = m (phi_p - phi_q) [+ m*(-g) for G term]
    rows, cols, vals = [], [], []
    for (pp, qq, mf) in faces:
        for a in range(2):
            for b in range(2):
                v = mf[:, a, b]
                rows += [2*pp+a, 2*pp+a, 2*qq+a, 2*qq+a]; cols += [2*pp+b, 2*qq+b, 2*qq+b, 2*pp+b]
                vals += [v, -v, v, -v]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2*n, 2*n))
    keep = np.arange(2, 2*n); lu = spla.splu(K[keep][:, keep].tocsc())
    Meff = np.zeros((2, 2, 2, 2))  # [field_out, dir_out, field_in, dir_in]
    for fin in range(2):
        for din in range(2):
            g = np.zeros(2); g[fin] = 1.0
            rhs = np.zeros(2*n)
            pp, qq, mf = faces[din]; mg = mf @ g          # outgoing from p includes -m g ; move to rhs: +m g at p, -m g at q
            np.add.at(rhs, (2*pp[:, None] + np.arange(2)).ravel(), mg.ravel())
            np.add.at(rhs, (2*qq[:, None] + np.arange(2)).ravel(), -mg.ravel())
            phi = np.zeros(2*n); phi[keep] = lu.solve(rhs[keep]); phi = phi.reshape(n, 2)
            for dout, (pp2, qq2, mf2) in enumerate(faces):
                dPhi = phi[qq2] - phi[pp2] + (g if dout == din else 0)
                F = -np.einsum('kab,kb->ka', mf2, dPhi).mean(0)
                Meff[:, dout, fin, din] = -F
    sig = Meff[0, :, 0, :]; L = Meff[0, :, 1, :]
    return sig, L, np.linalg.solve(sig, L)
