"""Extreme-contrast 3D Q1/FV/RT0 with local gradient-factor energies.

Uses junction geometry/assembly conventions. No difference of large global energies,
no squared-eigenvalue extraction, and no inverse-of-inverse RT0 evaluation.
Results are floating-point variational C enclosures, not interval verification.
"""

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
import itertools
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import coo_matrix
from common import rectangular_voxels as OLD

FE = OLD.FE


def factors(phases):
    out = []; values = []
    for sigma, seebeck, kappa in phases:
        # The thermodynamic square root is available without an eigensolve.
        root = np.array([[FE.BETA*np.sqrt(sigma/FE.ESCALE), seebeck*np.sqrt(sigma/FE.ESCALE)],
                         [0., np.sqrt(kappa/(FE.T0*FE.ESCALE))]])
        unused, singular, right = np.linalg.svd(root, full_matrices=False)
        out.append(singular[:, None]*right)
        values.append(singular.tolist())
    return np.array(out), values


def gram_sum(values):
    # Positive diagonal sums. Windows longdouble can have float64 precision;
    # stabilization comes from local gradient factors, not extra mantissa bits.
    a = values.reshape(-1, 6).astype(np.longdouble)
    return np.asarray(a.T@a, dtype=np.longdouble)


def q1_energy(labels, nodal, roots, h):
    u = nodal.reshape(tuple(labels.shape)+(2, 6)); count = labels.size
    energy = np.zeros((6, 6), dtype=np.longdouble)
    corners = list(itertools.product((0, 1), repeat=2))
    for d in range(3):
        others = [k for k in range(3) if k != d]; edges = []
        for corner in corners:
            shifts = [0, 0, 0]
            for ax, value in zip(others, corner): shifts[ax] = -value
            left = np.roll(u, tuple(shifts), axis=(0, 1, 2))
            shifts[d] = -1
            right = np.roll(u, tuple(shifts), axis=(0, 1, 2))
            # Differences remove each element's constant component before integration.
            grad = (right-left).reshape(count, 2, 6).copy()/h[d]
            grad[:, 0, d] += 1.; grad[:, 1, 3+d] += 1.
            edges.append(grad)
        for mode in corners:
            val = sum(((-1)**sum(a*b for a, b in zip(mode, corner))*g for corner, g in zip(corners, edges)), start=np.zeros_like(edges[0]))
            weight = 3**(2-sum(mode))/144
            for phase, root in enumerate(roots):
                transformed = root@val[labels.ravel() == phase]
                energy += weight*gram_sum(transformed)/count
    return np.asarray(energy, dtype=float)


def fem(labels, phases, h=(1., 1., 1.), tol=2e-11):
    started = time.perf_counter(); h = np.asarray(h, float)
    roots, singular = factors(phases)
    mat, dofs, coeff, ig, vol, sizes = FE.assemble(labels, phases, h=h)
    force = -np.einsum('id,padbe->piabe', ig, coeff.reshape(-1, 2, 3, 2, 3)).reshape(-1, 16, 6)
    rhs = np.zeros((mat.shape[0], 6))
    np.add.at(rhs, dofs.ravel(), force[labels.ravel()].reshape(-1, 6))
    u = np.zeros_like(rhs)
    u[2:], stats = FE.solve_spd(mat[2:, 2:], rhs[2:], np.arange(2, mat.shape[0]) % 2, tol)
    energy = q1_energy(labels, u, roots, h)
    legacy = np.mean(coeff[labels.ravel()], axis=0)-(rhs.T@u+u.T@rhs-u.T@(mat@u))/(labels.size*vol)
    out = FE.properties(energy)
    out.update(classification='P: uniform Q1; local gradient trial energy supplies continuum upper bound',
               shape=list(labels.shape), voxel_spacing=h.tolist(), solver=stats,
               phase_gradient_factor_singular_values=singular,
               local_energy_method='SVD of analytic two-field gradient factors; element constants removed by edge differences; Legendre polynomial moments and sums of squares',
               legacy_subtractive_energy_max_difference_P=float(np.max(np.abs(legacy-energy))),
               seconds=time.perf_counter()-started)
    return out


def project_flux(flux, h):
    shape = flux.shape[:3]
    def divergence(f):
        return sum((f[..., d, :]-np.roll(f[..., d, :], 1, axis=d))/h[d] for d in range(3))
    before = float(np.linalg.norm(divergence(flux))*min(h)/max(np.linalg.norm(flux), 1e-30))
    eigenvalues = np.zeros(shape)
    for d, size in enumerate(shape):
        sh = [1, 1, 1]; sh[d] = size
        eigenvalues += ((2-2*np.cos(2*np.pi*np.arange(size)/size))/h[d]**2).reshape(sh)
    eigenvalues[0, 0, 0] = 1.
    for unused in range(2):
        phat = -np.fft.fftn(divergence(flux), axes=(0, 1, 2))/eigenvalues[..., None, None]
        phat[0, 0, 0] = 0
        psi = np.fft.ifftn(phat, axes=(0, 1, 2)).real
        for d in range(3): flux[..., d, :] -= (np.roll(psi, -1, axis=d)-psi)/h[d]
    after = float(np.linalg.norm(divergence(flux))*min(h)/max(np.linalg.norm(flux), 1e-30))
    return flux, before, after


def rt0_lower(labels, flux, roots):
    count = labels.size
    inverse_roots = np.array([np.linalg.inv(r).T for r in roots])
    energy = np.zeros((6, 6), dtype=np.longdouble)
    for d in range(3):
        plus = flux[..., d, :].reshape(count, 2, 6)
        minus = np.roll(flux[..., d, :], 1, axis=d).reshape(count, 2, 6)
        for values, weight in ((plus+minus, .25), (plus-minus, 1/12)):
            for phase, root in enumerate(inverse_roots):
                energy += weight*gram_sum(root@values[labels.ravel() == phase])/count
    energy = np.asarray(energy, float)
    mean = np.asarray(np.mean(flux.astype(np.longdouble), axis=(0, 1, 2)), float).reshape(6, 6)
    chol = cholesky(energy, lower=True)
    projected = solve_triangular(chol, mean.T, lower=True)
    lower = projected.T@projected
    return lower, energy, mean


def fv(labels, phases, h=(1., 1., 1.), tol=2e-11):
    started = time.perf_counter(); h = np.asarray(h, float); vol = float(np.prod(h)); n = labels.size
    roots, singular = factors(phases)
    a = np.array([FE.energy(p) for p in phases]); inva = np.linalg.inv(a)
    hm = np.array([[np.linalg.inv((x+y)/2) for y in inva] for x in inva])
    idx = np.arange(n, dtype=np.int32).reshape(labels.shape)
    rows = []; cols = []; entries = []; faces = []; rhs = np.zeros((2*n, 6))
    for d in range(3):
        p = idx.ravel(); q = np.roll(idx, -1, axis=d).ravel()
        face = hm[labels.ravel(), np.roll(labels, -1, axis=d).ravel()]
        conductance = face*vol/h[d]**2; faces.append((p, q, face))
        for x in range(2):
            for y in range(2):
                value = conductance[:, x, y]
                rows.extend([2*p+x, 2*p+x, 2*q+x, 2*q+x])
                cols.extend([2*p+y, 2*q+y, 2*q+y, 2*p+y])
                entries.extend([value, -value, value, -value])
        for field in range(2):
            j = 3*field+d; forcing = conductance[:, :, field]*h[d]
            np.add.at(rhs[:, j], (2*p[:, None]+np.arange(2)).ravel(), forcing.ravel())
            np.add.at(rhs[:, j], (2*q[:, None]+np.arange(2)).ravel(), -forcing.ravel())
    mat = coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))), shape=(2*n, 2*n)).tocsr()
    del rows, cols, entries
    u = np.zeros_like(rhs)
    u[2:], stats = FE.solve_spd(mat[2:, 2:], rhs[2:], np.arange(2, 2*n) % 2, tol)
    u = u.reshape(n, 2, 6); flux = np.empty((n, 2, 3, 6)); discrete_energy = np.zeros((6, 6), dtype=np.longdouble)
    for d, (p, q, face) in enumerate(faces):
        g = (u[q]-u[p])/h[d]; g[:, 0, d] += 1.; g[:, 1, 3+d] += 1.
        flux[:, :, d, :] = face@g
        transform = np.linalg.cholesky(face).transpose(0, 2, 1)@g
        discrete_energy += gram_sum(transform)/n
    old = flux.mean(axis=0).reshape(6, 6)
    out = FE.properties(np.asarray(discrete_energy, float))
    flux, before, after = project_flux(flux.reshape(tuple(labels.shape)+(2, 3, 6)), h)
    lower, complementary, mean = rt0_lower(labels, flux, roots)
    out.update(classification='P: rectangular FV discrete trial energy; projected RT0 complementary field separately supplies continuum lower bound',
               shape=list(labels.shape), voxel_spacing=h.tolist(), solver=stats,
               phase_gradient_factor_singular_values=singular,
               flux_energy_max_difference_P=float(np.max(np.abs(old-discrete_energy))),
               continuum_energy_lower=lower.tolist(), complementary_raw_energy=complementary.tolist(), flux_mean=mean.tolist(),
               RT0_divergence_relative_before_projection=before, RT0_divergence_relative_after_projection=after,
               seconds=time.perf_counter()-started)
    return out


def pair(labels, phases, h):
    first = fem(labels, phases, h)
    print('  FEM', first['Sa_z_uV_K'], first['sigma_diag'], flush=True)
    second = fv(labels, phases, h)
    enc = OLD.R.R.enclosure(second['continuum_energy_lower'], first['scaled_energy'])
    # Fourfold symmetry makes the symmetric electric tensor exactly diagonal.
    a = enc['sigma_xx_interval']; z = enc['sigma_zz_interval']
    enc['sigma_tensor_component_intervals_S_m'] = [[a, [0., 0.], [0., 0.]], [[0., 0.], a, [0., 0.]], [[0., 0.], [0., 0.], z]]
    enc['sigma_tensor_form_C'] = 'diag(sigma_perp, sigma_perp, sigma_z), off-diagonals exactly zero by the verified colour-preserving fourfold symmetry'
    assert min(a+z) > 0
    assert enc['raw_minimum_upper_minus_lower_eigenvalue'] > 0
    assert second['RT0_divergence_relative_after_projection'] < 1e-13
    print('  FV', second['Sa_z_uV_K'], 'C', enc['Sa_z_interval_uV_K'], flush=True)
    return {'fem_P': first, 'fv_P': second, 'enclosure_C': enc}
