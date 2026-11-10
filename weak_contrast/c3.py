"""Leading weak-contrast axial term of a periodic 2D composite of isotropic thermoelectric phases.

Local conductivity K(x) = s(x) (x) I_2 with symmetric 2 x 2 transport matrices s(x) = s0 + eps * sum_a chi_a(x) D_a.
Brown series with reference K0 = s0 (x) I_2 and Green operator Gamma(k) = s0^-1 (x) k k^T / |k|^2:
  K_eff = K0 + <dK> - <dK G dK> + <dK G dK G dK> - ...
The first two corrections are spatially symmetric; the third carries the antisymmetric part
  c3 = 1/2 sum_abc [D_a P D_b P D_c]_VT G_abc,   P = s0^-1,
  G_abc = sum_{k,q != 0} chi_a(-k) chi_b(k-q) chi_c(q) (k.q)(k x q) / (|k|^2 |q|^2)   (geometry only),
and with the reference equal to phase 3 (only chi_1, chi_2 enter) the material factor collapses to
  c3 = 2 sqrt(det s0) (d1 x d2) [d1_0 G_112 + d2_0 G_122],
where s0^-1/2 D_a s0^-1/2 = d_a0 I + d_az sigma_z + d_ax sigma_x and d1 x d2 = d1z d2x - d1x d2z.
All checks below are on the same spectral discretization (FFT grid), so the series term must match the full
Lippmann-Schwinger solution at small eps to O(eps^4)."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json; import os
import numpy as np

HERE = str(RELEASE_ROOT)

T0 = 300.0


def cell(N, alpha, period, nph):                     # reference pinwheel geometry (same as planar_cells/pinwheel.py)
    h = N // 2; x, y = np.meshgrid(np.arange(h) + 0.5, np.arange(h) + 0.5, indexing='ij')
    a = np.deg2rad(alpha); u = x * np.cos(a) + y * np.sin(a)
    Pq = (np.floor(u / (period / nph)) % nph).astype(int) + 1
    Cc = np.zeros((N, N), int); Cc[:h, :h] = Pq
    return Cc + np.rot90(Cc, 1) + np.rot90(Cc, 2) + np.rot90(Cc, 3) - 1


def smat(sig, S, kap):                                # Onsager-symmetric scaled transport matrix
    return np.array([[sig, sig * S], [sig * S, (kap + T0 * sig * S * S) / T0]])


def freqs(N):
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing='ij'); K2 = KX ** 2 + KY ** 2; K2[0, 0] = 1.0
    return KX, KY, K2


def proj(f, KX, KY, K2):
    """Pi f = grad div Laplacian^-1 f for a vector field f[..., 2] (component-wise on any leading axes): (k k^T / k^2) f_hat."""
    F = np.fft.fft2(f, axes=(0, 1))
    kf = (KX[..., None] * F[:, :, 0] + KY[..., None] * F[:, :, 1]) if F.ndim == 4 else (KX * F[:, :, 0] + KY * F[:, :, 1])
    if F.ndim == 4:
        out = np.stack([KX[..., None] * kf, KY[..., None] * kf], axis=2) / K2[..., None, None]
        out[0, 0] = 0
    else:
        out = np.stack([KX * kf, KY * kf], axis=2) / K2[..., None]
        out[0, 0] = 0
    return np.real(np.fft.ifft2(out, axes=(0, 1)))


def G3(chis, KX, KY, K2):
    """G_abc = T_xy - T_yx with T_ij = < chi_a [Pi (chi_b [Pi (chi_c e_j)])]_i >."""
    n = len(chis); G = np.zeros((n, n, n))
    for c in range(n):
        T = []
        for j in range(2):
            f = np.zeros(chis[c].shape + (2,)); f[..., j] = chis[c]
            u = proj(f, KX, KY, K2); T.append(u)
        for b in range(n):
            v = [proj(chis[b][..., None] * T[j], KX, KY, K2) for j in range(2)]
            for a in range(n):
                Txy = np.mean(chis[a] * v[1][..., 0]); Tyx = np.mean(chis[a] * v[0][..., 1])
                G[a, b, c] = Txy - Tyx
    return G


def keff_LS(lab, mats, s0, tol=1e-13, maxit=2000):
    """Full effective 4 x 4 matrix by the basic Lippmann-Schwinger scheme with reference s0 (same FFT grid)."""
    N = lab.shape[0]; KX, KY, K2 = freqs(N); P = np.linalg.inv(s0)
    dS = np.array(mats)[lab] - s0                      # N,N,2,2 fluctuation of s
    C = np.zeros((4, 4))
    for col in range(4):
        fe, sp = divmod(col, 2)                        # field (V,T), spatial (x,y)
        E = np.zeros((2, 2)); E[fe, sp] = 1.0          # e[field, spatial]
        e = np.broadcast_to(E, (N, N, 2, 2)).copy()
        for it in range(maxit):
            tau = np.einsum('xyab,xybj->xyaj', dS, e)  # polarization
            F = np.fft.fft2(tau, axes=(0, 1))
            kt = KX[..., None] * F[:, :, :, 0] + KY[..., None] * F[:, :, :, 1]          # (k . tau_a)
            Pk = np.einsum('ab,xyb->xya', P, kt)
            Ghat = np.stack([KX[..., None] * Pk, KY[..., None] * Pk], axis=3) / K2[..., None, None]
            Ghat[0, 0] = 0
            en = E - np.real(np.fft.ifft2(Ghat, axes=(0, 1)))
            d = np.max(np.abs(en - e)); e = en
            if d < tol: break
        J = np.einsum('xyab,xybj->xyaj', np.array(mats)[lab], e).mean((0, 1))
        C[:, col] = J.reshape(4)
    return C


def axial_c(C): return 0.5 * (C[0, 3] - C[1, 2])


def closed_form(s0, D1, D2, G):
    w, V = np.linalg.eigh(s0); R = V @ np.diag(w ** -0.5) @ V.T
    d = []
    for D in (D1, D2):
        M = R @ D @ R; d.append((0.5 * (M[0, 0] + M[1, 1]), 0.5 * (M[0, 0] - M[1, 1]), M[0, 1]))   # (d0, dz, dx)
    cross = d[0][1] * d[1][2] - d[0][2] * d[1][1]
    return 2 * np.sqrt(np.linalg.det(s0)) * cross * (d[0][0] * G[0, 0, 1] + d[1][0] * G[0, 1, 1])


def series_c3(s0, Ds, G):
    P = np.linalg.inv(s0); n = len(Ds); tot = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                M = Ds[a] @ P @ Ds[b] @ P @ Ds[c]; tot += 0.5 * M[0, 1] * G[a, b, c]
    return tot


if __name__ == '__main__':
    out = {}
    up = [1, 2, 4, 8]
    base = cell(24, 15, 6, 3)                          # labels: 0 p, 1 n, 2 weak
    geoms = {'pinwheel': base, 'mirror': base[::-1, :], 'p4mm': cell(24, 45, 6, 3)}
    # geometry factor and its convergence under k-fold upsampling of the pixels
    for name, g in geoms.items():
        out[name] = {}
        for k in up:
            lab = np.kron(g, np.ones((k, k), int)); KX, KY, K2 = freqs(lab.shape[0])
            chis = [(lab == 0).astype(float), (lab == 1).astype(float)]           # reference = phase 2 (weak)
            G = G3(chis, KX, KY, K2)
            out[name][f'k{k}'] = {'G112': G[0, 0, 1], 'G122': G[0, 1, 1], 'G_antisym_check': float(np.max(np.abs(G + np.transpose(G, (2, 1, 0)))))}
            print(name, k, out[name][f'k{k}'], flush=True)
    # algebra check on the same grid: full LS solution versus the cubic term (generic weak contrast)
    rng = np.random.default_rng(1)
    s0 = np.array([[1.0, 0.3], [0.3, 0.8]])
    D1 = np.array([[0.7, -0.2], [-0.2, 0.1]]); D2 = np.array([[-0.3, 0.5], [0.5, 0.9]])
    lab = base; KX, KY, K2 = freqs(24); chis = [(lab == 0).astype(float), (lab == 1).astype(float)]
    G = G3(chis, KX, KY, K2)
    pred_series = series_c3(s0, [D1, D2], G); pred_closed = closed_form(s0, D1, D2, G)
    fits = []
    for eps in (0.02, 0.01, 0.005):
        mats = [s0 + eps * D1, s0 + eps * D2, s0]
        c = axial_c(keff_LS(lab, mats, s0))
        fits.append({'eps': eps, 'c_full': c, 'c_over_eps3': c / eps ** 3})
        print('eps', eps, c / eps ** 3, 'series', pred_series, 'closed', pred_closed, flush=True)
    out['algebra_check'] = {'s0': s0.tolist(), 'D1': D1.tolist(), 'D2': D2.tolist(), 'pred_series': pred_series,
                            'pred_closed_form': pred_closed, 'full': fits}
    # vanishing cases at this order
    c2 = axial_c(keff_LS(cell(24, 15, 6, 2), [s0 + 0.01 * D1, s0 + 0.01 * D2], s0))
    cm = axial_c(keff_LS(cell(24, 45, 6, 3), [s0 + 0.01 * D1, s0 + 0.01 * D2, s0], s0))
    cmir = axial_c(keff_LS(base[::-1, :], [s0 + 0.01 * D1, s0 + 0.01 * D2, s0], s0))
    # special relation: traceless contrasts in the s0-normalized frame (d1_0 = d2_0 = 0) remove the cubic term
    w, V = np.linalg.eigh(s0); Rh = V @ np.diag(w ** 0.5) @ V.T
    T1 = Rh @ np.array([[0.4, 0.3], [0.3, -0.4]]) @ Rh; T2 = Rh @ np.array([[-0.2, 0.6], [0.6, 0.2]]) @ Rh
    ct = [axial_c(keff_LS(base, [s0 + e * T1, s0 + e * T2, s0], s0)) / e ** 3 for e in (0.02, 0.01)]
    out['vanishing'] = {'two_phase_eps0.01': c2, 'p4mm_eps0.01': cm, 'mirror_eps0.01_over_eps3': cmir / 1e-6,
                        'traceless_c_over_eps3_at_0.02_0.01': ct}
    print(out['vanishing'], flush=True)
    json.dump(out, open(os.path.join(HERE, 'weak_contrast/c3_results.json'), 'w'), indent=1, default=float)
