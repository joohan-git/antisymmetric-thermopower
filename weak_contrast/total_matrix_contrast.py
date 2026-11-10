"""Weak total-matrix contrast: fixed Fig. 1b raster, FV and Taylor coefficients."""

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
import ast
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from common.transport2d import ROOT, FV, PHASES, material_energy, save, BETA


def cell():
    """Reference pinwheel cell(24, 15, 6, 3) of planar_cells/pinwheel.py (read only)."""
    path = ROOT/'planar_cells/pinwheel.py'
    namespace = {'np': np, '__file__': str(path), '__name__': 'pinwheel_geometry'}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    return namespace['cell'](24, 15, 6, 3), path


def from_energy(a):
    return [float(1e5*a[0, 0]), float(BETA*a[0, 1]/a[0, 0]), float(1.2*(a[1, 1]-a[0, 1]**2/a[0, 0]))]


def fv_energy(labels, matrices):
    return FV.homogenize(labels, matrices)[3].reshape(4, 4)


def taylor_fv(labels, delta, order=5):
    """Taylor expansion of the actual harmonic-face FV equations, no finite differences."""
    n = len(labels); count = n*n
    ii, jj = np.indices(labels.shape)
    p = (ii*n+jj).ravel()
    values = delta[labels].reshape(count, 2, 2)
    edges = []
    for di, dj in ((1, 0), (0, 1)):
        q = (((ii+di) % n)*n+(jj+dj) % n).ravel()
        ap = [np.broadcast_to(np.eye(2), values.shape).copy()]
        aq = [ap[0].copy()]
        for k in range(order):
            ap.append(-values @ ap[-1]); aq.append(-values[q] @ aq[-1])
        h = [(a+b)/2 for a, b in zip(ap, aq)]
        f = [ap[0].copy()]
        for k in range(1, order+1):
            f.append(-sum(h[j] @ f[k-j] for j in range(1, k+1)))
        edges.append((p, q, f))
    matrices, rhs = [], []
    for k in range(order+1):
        rows, cols, entries = [], [], []
        b = np.zeros((2*count, 4))
        for direction, (p, q, series) in enumerate(edges):
            f = series[k]
            for a in range(2):
                for c in range(2):
                    rows.extend([2*p+a, 2*p+a, 2*q+a, 2*q+a])
                    cols.extend([2*p+c, 2*q+c, 2*q+c, 2*p+c])
                    entries.extend([f[:, a, c], -f[:, a, c], f[:, a, c], -f[:, a, c]])
                    np.add.at(b[:, 2*c+direction], 2*p+a, f[:, a, c])
                    np.add.at(b[:, 2*c+direction], 2*q+a, -f[:, a, c])
        matrices.append(coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))), shape=(2*count, 2*count)).tocsc())
        rhs.append(b)
    lu = splu(matrices[0][2:, 2:])
    phi = []
    for k in range(order+1):
        forcing = rhs[k] - sum((matrices[j] @ phi[k-j] for j in range(1, k+1)), start=np.zeros_like(rhs[k]))
        potential = np.zeros_like(forcing)
        potential[2:] = lu.solve(forcing[2:])
        phi.append(potential)
    effective = []
    for k in range(order+1):
        c = np.zeros((4, 4))
        for direction, (p, q, f) in enumerate(edges):
            flux = np.zeros((count, 2, 4))
            flux[:, :, direction] += f[k][:, :, 0]
            flux[:, :, 2+direction] += f[k][:, :, 1]
            for j in range(k+1):
                pp = phi[k-j].reshape(count, 2, 4)
                flux += f[j] @ (pp[q]-pp[p])
            c[[direction, 2+direction]] = flux.mean(axis=0)
        effective.append(c)
    return effective


def main():
    labels, source = cell()
    phases = [PHASES[x] for x in ('p', 'n', 'weak')]
    matrices = np.array([material_energy(x) for x in phases])
    fractions = np.bincount(labels.ravel(), minlength=3)/labels.size
    base = np.einsum('p,pij->ij', fractions, matrices)
    eig, vec = np.linalg.eigh(base)
    half = (vec*np.sqrt(eig)) @ vec.T
    invhalf = (vec/np.sqrt(eig)) @ vec.T
    delta = np.array([invhalf @ (a-base) @ invhalf for a in matrices])
    transform = np.kron(half, np.eye(2))
    cseries = [transform @ c @ transform for c in taylor_fv(labels, delta)]
    sseries = []
    for k, c in enumerate(cseries):
        b = c[:2, 2:] - sum((cseries[j][:2, :2] @ sseries[k-j] for j in range(1, k+1)), start=np.zeros((2, 2)))
        sseries.append(np.linalg.solve(cseries[0][:2, :2], b))
    axial = lambda s: float((s[0, 1]-s[1, 0])*BETA*5e5)
    out = {'classification': 'P: fixed 24x24 harmonic-face FV estimates and algebraic Taylor coefficients; no continuum amplitude certificate',
        'geometry_source': source.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'labels': labels.tolist(),
        'endpoint_phases_sigma_S_kappa': phases, 'fractions': fractions.tolist(), 's0_scaled': base.tolist(),
        'd_i_scaled': (matrices-base).tolist(), 'D_i_whitened': delta.tolist(), 'C_Taylor_scaled': [c.tolist() for c in cseries],
        'Sa_Taylor_uV_K': [axial(s) for s in sseries], 'rows': [], 'controls': {}}
    out['analytic_derivation'] = {
        'scope': 'Continuum periodic transport with locally isotropic symmetric positive definite two-field matrices; fixed geometry; small total matrix contrast. This is not a Seebeck-only scaling.',
        'normalization': 's_i = s0 + epsilon d_i. Choose s0 as the volume mean of the endpoint matrices, so mean(d)=0. Set H=s0^(-1/2), D(x)=H d(x) H. The local normalized coefficient is I+epsilon D(x).',
        'projection': 'Gamma is the orthogonal projection onto mean-zero spatial gradients, acting identically on both fields. For nonzero Fourier wave vector k, Gamma_ij(k)=k_i k_j/|k|^2; Gamma(0)=0.',
        'corrector': 'g = E - epsilon Gamma D g; hence g = E - epsilon Gamma D E + epsilon^2 Gamma D Gamma D E + ... . The Neumann series converges when |epsilon| times the essential maximum norm of D is less than one.',
        'effective_series': 'Cstar = I + epsilon mean(D) - epsilon^2 mean(D Gamma D) + epsilon^3 mean(D Gamma D Gamma D) + O(epsilon^4). Restore physical field scaling by congruence with s0^(1/2).',
        'orders_1_2': 'The first-order term is spatially isotropic. The second-order kernel Gamma_ij is symmetric in i,j, so every second-order spatial block is symmetric. The zeroth and first-order electric and thermoelectric blocks are spatial scalars; multiplying by the electrical inverse therefore produces no axial Seebeck part through second order.',
        'cubic_structure': 'Let F_ij(x,y,z)=sum_l Gamma_il(x-y) Gamma_lj(y-z). The spatially antisymmetric part of the cubic block is one half of the cell-normalized triple integral of [D(x)D(y)D(z)-D(z)D(y)D(x)] F_ij(x,y,z). This is obtained by exchanging x and z in the transposed spatial component. Interpret the kernel expression through the bounded projection operator or the Fourier-truncation limit, not as an unregularized product of singular kernels.',
        'phase_reduction': 'If all local s_i lie in the span of two fixed matrices, their mean s0 lies in that span. Whitening sends that plane to span(I,D0), so all normalized contrasts commute and all reversed products agree. In p4, the electric block is a scalar spatial tensor, so the two-mode reduction enforces Sa=0 exactly. Without p4, two symmetric effective spatial blocks need not commute; their product A^(-1)B can have an axial part, as the rod/block counterexample shows.',
        'qualification': 'Noncommuting contrasts remove the two-mode obstruction but do not guarantee a nonzero cubic coefficient for every geometry or material relation. Additional symmetry or cancellations can remove the cubic term. The proper general claim is first allowed order, generically cubic, not an unconditional nonzero third-order term.',
        'suggested_English': 'For weak contrast in the full coupled transport matrices, the first symmetry-allowed axial contribution is cubic. Its coefficient can vanish for special material relations or geometries; under p4 symmetry the two-phase reduction removes the axial response at every order.'}
    for epsilon in np.logspace(-3, -1, 9):
        mixed = base + epsilon*(matrices-base)
        c = fv_energy(labels, mixed)
        s = np.linalg.solve(c[:2, :2], c[:2, 2:])
        sa = axial(s)
        out['rows'].append({'epsilon': float(epsilon), 'Sa_uV_K_P': sa, 'Sa_over_eps3_P': sa/epsilon**3,
            'Sa_Taylor_order5_P': sum(out['Sa_Taylor_uV_K'][j]*epsilon**j for j in range(6)),
            'phases_sigma_S_kappa': [from_energy(x) for x in mixed], 'Onsager_relative_P': float(np.linalg.norm(c-c.T)/np.linalg.norm(c))})
    x = np.log([r['epsilon'] for r in out['rows']]); y = np.log(np.abs([r['Sa_uV_K_P'] for r in out['rows']]))
    out['loglog_slopes_P'] = {'all': float(np.polyfit(x, y, 1)[0]), 'eps_le_001': float(np.polyfit(x[:5], y[:5], 1)[0]),
        'adjacent': (np.diff(y)/np.diff(x)).tolist()}
    for name, mats, lab in [('mirror', matrices, labels[::-1, :].copy()),
                            ('two_matrix_span', np.array([matrices[0], matrices[1], .35*matrices[0]+.65*matrices[1]]), labels)]:
        s0 = np.einsum('p,pij->ij', fractions, mats)
        c = fv_energy(lab, s0 + .1*(mats-s0))
        out['controls'][name] = {'epsilon': .1, 'Sa_uV_K_P': axial(np.linalg.solve(c[:2, :2], c[:2, 2:]))}
    special = np.array([[[1., 0.], [0., -1.]], [[0., 1.], [1., 0.]], [[-.4, .6], [.6, .4]]])
    special -= np.einsum('p,pij->ij', fractions, special)
    special_c = taylor_fv(labels, special)
    special_s = []
    for k, c in enumerate(special_c):
        special_s.append(np.linalg.solve(special_c[0][:2, :2], c[:2, 2:] - sum((special_c[j][:2, :2] @ special_s[k-j] for j in range(1, k+1)), start=np.zeros((2, 2)))))
    fourth = {'D_i': special.tolist(), 's0': np.eye(2).tolist(), 'Sa_Taylor_uV_K_P': [axial(s) for s in special_s], 'rows': [],
        'analytic_C': 'Every real symmetric traceless two-by-two matrix is a linear combination of the two real Pauli matrices. Every product of three such matrices is symmetric; consequently D(x)D(y)D(z)-D(z)D(y)D(x)=0 and the cubic axial coefficient vanishes exactly, even though the three local matrices are linearly independent.'}
    for eps in (.001, .00316227766016838, .01, .0316227766016838, .1):
        c = fv_energy(labels, np.eye(2)+eps*special)
        value = axial(np.linalg.solve(c[:2, :2], c[:2, 2:]))
        fourth['rows'].append({'epsilon': eps, 'Sa_uV_K_P': value, 'Sa_over_eps4_P': value/eps**4})
    fourth['slope_eps_ge_001_P'] = float(np.polyfit(np.log([r['epsilon'] for r in fourth['rows'][-3:]]), np.log(np.abs([r['Sa_uV_K_P'] for r in fourth['rows'][-3:]])), 1)[0])
    out['controls']['traceless_cubic_cancellation'] = fourth
    save('weak_contrast/total_matrix_contrast.json', out)
    print('Taylor coefficients:', out['Sa_Taylor_uV_K'], flush=True)
    print('slopes:', out['loglog_slopes_P'], flush=True)
    for row in out['rows']: print(row['epsilon'], row['Sa_uV_K_P'], row['Sa_over_eps3_P'], flush=True)


if __name__ == '__main__':
    main()
