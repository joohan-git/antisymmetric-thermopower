"""Checks of the cubic axial term beyond the generic algebra test in c3.py.
(1) Real materials of the reference pinwheel, scaled toward the weak phase, s_i(eps) = s3 + eps (s_i - s3):
    finite-volume S_a(eps)/eps^3 (homog2.py, an independent discretization, k-fold refined pixels) versus the
    prediction S_a^(3) = det[s1, s2, s3] / (det s3 * s3_VV) * (theta_1 G_112 + theta_2 G_122).
(2) Three dimensions: woodpile W2 (certified/r13_geometries.json, W2_split_clear voxels), spectral Lippmann-Schwinger at small eps versus the
    same formula with G built from the z component of (k x q)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'solvers', 'fv2d'))
import homog2 as H2
import c3 as M

T0 = M.T0


def vec(s): return [s[0, 0], s[0, 1], s[1, 1]]


def prediction(s1, s2, s3, G112, G122):
    det = np.linalg.det(np.array([vec(s1), vec(s2), vec(s3)]))
    th = [0.5 * np.trace(np.linalg.solve(s3, s)) - 1 for s in (s1, s2)]
    c3 = det / np.linalg.det(s3) * (th[0] * G112 + th[1] * G122)
    return c3, c3 / s3[0, 0], det, th


out = {}
# ---------------- (1) real materials, 2D pinwheel ----------------
mat = {'p': (1e5, 200e-6, 1.0), 'n': (1e5, -200e-6, 1.0), 'w': (1e3, 0.0, 0.2)}
S = {k: M.smat(*v) for k, v in mat.items()}
base = M.cell(24, 15, 6, 3)
res = {}
for k in (1, 2, 4):
    lab = np.kron(base, np.ones((k, k), int)); KX, KY, K2 = M.freqs(lab.shape[0])
    G = M.G3([(lab == 0).astype(float), (lab == 1).astype(float)], KX, KY, K2)
    c3, Sa3, det, th = prediction(S['p'], S['n'], S['w'], G[0, 0, 1], G[0, 1, 1])
    fv = []
    for eps in (0.04, 0.02, 0.01):
        mats = []
        for key in ('p', 'n', 'w'):                      # unscaled matrices for homog2: interpolate the symmetric s, then unscale
            s = S['w'] + eps * (S[key] - S['w'])
            sig, sS, tt = s[0, 0], s[0, 1], s[1, 1]
            mats.append(np.array([[sig, sS], [T0 * sS, T0 * tt]]))
        sig, L, Seff = H2.homogenize(lab, mats)
        fv.append((eps, 0.5 * (Seff[0, 1] - Seff[1, 0]) / eps ** 3))
    res[f'k{k}'] = {'G112': G[0, 0, 1], 'G122': G[0, 1, 1], 'pred_Sa3_V_per_K': Sa3, 'pred_Sa3_uV_per_K': Sa3 * 1e6,
                    'fv_Sa_over_eps3_uV_per_K': [(e, v * 1e6) for e, v in fv], 'det_s123': det, 'theta': th}
    print('k', k, 'pred', Sa3 * 1e6, 'FV', [(e, round(v * 1e6, 4)) for e, v in fv], flush=True)
out['pinwheel_real_materials'] = res

# ---------------- (2) three dimensions: W2 ----------------
def freqs3(N):
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij'); K2 = KX ** 2 + KY ** 2 + KZ ** 2; K2[0, 0, 0] = 1.0
    return KX, KY, KZ, K2


def proj3(f, KX, KY, KZ, K2):
    F = np.fft.fftn(f, axes=(0, 1, 2)); kf = KX * F[..., 0] + KY * F[..., 1] + KZ * F[..., 2]
    o = np.stack([KX * kf, KY * kf, KZ * kf], axis=3) / K2[..., None]; o[0, 0, 0] = 0
    return np.real(np.fft.ifftn(o, axes=(0, 1, 2)))


def G3_3d(chis, F):
    n = len(chis); G = np.zeros((n, n, n))
    for c in range(n):
        U = []
        for j in range(2):
            f = np.zeros(chis[c].shape + (3,)); f[..., j] = chis[c]; U.append(proj3(f, *F))
        for b in range(n):
            V = [proj3(chis[b][..., None] * U[j], *F) for j in range(2)]
            for a in range(n):
                G[a, b, c] = np.mean(chis[a] * V[1][..., 0]) - np.mean(chis[a] * V[0][..., 1])
    return G


def keff3_c(lab, mats, s0, tol=1e-12, maxit=500):
    N = lab.shape[0]; KX, KY, KZ, K2 = freqs3(N); P = np.linalg.inv(s0); Ms = np.array(mats)[lab]; dS = Ms - s0
    cs = {}
    for fe, sp in ((1, 1), (1, 0)):                      # T gradient along y and along x
        E = np.zeros((2, 3)); E[fe, sp] = 1.0; e = np.broadcast_to(E, (N, N, N, 2, 3)).copy()
        for it in range(maxit):
            tau = np.einsum('xyzab,xyzbj->xyzaj', dS, e); Ft = np.fft.fftn(tau, axes=(0, 1, 2))
            kt = KX[..., None] * Ft[..., 0] + KY[..., None] * Ft[..., 1] + KZ[..., None] * Ft[..., 2]
            Pk = np.einsum('ab,xyzb->xyza', P, kt)
            Gh = np.stack([KX[..., None] * Pk, KY[..., None] * Pk, KZ[..., None] * Pk], axis=4) / K2[..., None, None]
            Gh[0, 0, 0] = 0
            en = E - np.real(np.fft.ifftn(Gh, axes=(0, 1, 2))); d = np.max(np.abs(en - e)); e = en
            if d < tol: break
        J = np.einsum('xyzab,xyzbj->xyzaj', Ms, e).mean((0, 1, 2)); cs[(fe, sp)] = J
    return 0.5 * (cs[(1, 1)][0, 0] - cs[(1, 0)][0, 1])      # c = (C_{Vx,Ty} - C_{Vy,Tx}) / 2


g13 = json.load(open(os.path.join(HERE, '..', 'certified', 'r13_geometries.json')))
lab3 = np.array(g13['woodpile']['W2_split_clear']['labels'])        # 0 weak, 1 n, 2 p
F3 = freqs3(lab3.shape[0])
G = G3_3d([(lab3 == 2).astype(float), (lab3 == 1).astype(float)], F3)  # phases 1 = p, 2 = n; reference = weak
s0 = np.array([[1.0, 0.3], [0.3, 0.8]]); D1 = np.array([[0.7, -0.2], [-0.2, 0.1]]); D2 = np.array([[-0.3, 0.5], [0.5, 0.9]])
pred = M.series_c3(s0, [D1, D2], G)
lab3m = np.where(lab3 == 2, 0, np.where(lab3 == 1, 1, 2))          # relabel to 0 p, 1 n, 2 weak
full = []
for eps in (0.02, 0.01):
    c = keff3_c(lab3m, [s0 + eps * D1, s0 + eps * D2, s0], s0); full.append((eps, c / eps ** 3))
    print('3D eps', eps, c / eps ** 3, 'pred', pred, flush=True)
cmir = keff3_c(lab3m[::-1, :, :], [s0 + 0.01 * D1, s0 + 0.01 * D2, s0], s0) / 1e-6
out['W2_3d'] = {'G112': G[0, 0, 1], 'G122': G[0, 1, 1], 'pred_c3': pred, 'full_c_over_eps3': full, 'mirror_c_over_eps3_at_0.01': cmir}
print(out['W2_3d'], flush=True)
json.dump(out, open(os.path.join(HERE, 'c3_checks.json'), 'w'), indent=1, default=float)
