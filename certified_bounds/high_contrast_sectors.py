"""planar conforming sectors with SVD DtN and cancellation-free trial energies.

Extreme electrical contrast makes squared-eigenvalue extraction and global
direct-minus-corrector energy subtraction inaccurate. Geometry and trial space
are unchanged. This module uses the gradient factor before squaring and sums
local energies after removal of the exact constant nullspaces.
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, svd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from certified_bounds.corner_sectors import outer_template, angular_template
from common.transport2d import compact, inverse_phases, enclosure_from_pair
from common.fem2d import material_energy, BETA, ENERGY_SCALE


def dtn(coeff, quadrants, m):
    angles, mm, kk = angular_template(m); n = len(angles)
    mass = np.zeros((2*n, 2*n)); factor = np.zeros_like(mass)
    for i in range(n):
        nodes = [i, (i+1) % n]
        dofs = (2*np.array(nodes)[:, None] + np.arange(2)).ravel()
        a = coeff[quadrants[i//(2*m)]]
        mass[np.ix_(dofs, dofs)] += np.kron(mm[i], a)
        f = np.sqrt(kk[i, 0, 0])*cholesky(a, lower=True).T
        factor[2*i:2*i+2, 2*nodes[0]:2*nodes[0]+2] += f
        factor[2*i:2*i+2, 2*nodes[1]:2*nodes[1]+2] -= f
    scale = np.sqrt(mass.diagonal())
    chol = cholesky(mass/scale[:, None]/scale[None, :], lower=True)
    transformed = solve_triangular(chol, (factor/scale[None, :]).T, lower=True).T
    unused, lam, vt = svd(transformed, full_matrices=False, lapack_driver='gesvd')
    lam[-2:] = 0.
    weighted = scale[:, None]*(chol @ vt.T)
    block = (weighted*lam) @ weighted.T
    constants = np.tile(np.eye(2), (n, 1))/np.sqrt(n)
    project = np.eye(2*n)-constants @ constants.T
    block = project @ block @ project
    return (block+block.T)/2, float(lam[-3])


def sector_homogenize(base, phases, m=4, nr=3, radius=.22, ngauss=10):
    started = time.perf_counter(); n = len(base); count = n*n
    coords_local, elem_local, shape = outer_template(m, nr, radius, ngauss)
    shifts = np.array([(i, j) for i in range(n) for j in range(n)])
    allcoords = (coords_local[None, :, :]+shifts[:, None, :]).reshape(-1, 2) % n
    rounded = np.round(allcoords, 11); rounded[rounded == n] = 0
    vertices, index = np.unique(rounded, axis=0, return_inverse=True); nv = len(vertices)
    localmap = index.reshape(count, len(coords_local))
    nodes = localmap[:, elem_local].reshape(-1, 4)
    dofs = (2*nodes[:, :, None]+np.arange(2)).reshape(-1, 8); ne = len(nodes)
    coeff = np.array([material_energy(p) for p in phases])
    values = np.einsum('eij,pab->peiajb', shape, coeff[base.ravel()]).reshape(ne, 8, 8)
    row = [np.broadcast_to(dofs[:, :, None], values.shape).ravel()]
    col = [np.broadcast_to(dofs[:, None, :], values.shape).ravel()]
    entries = [values.ravel()]
    g = np.zeros((len(elem_local), 4, 2, 4)); xy = coords_local[elem_local]/n
    g[:, :, 0, :2] = xy; g[:, :, 1, 2:] = xy
    force = -np.einsum('eij,ejk->eik', values, np.tile(g.reshape(-1, 8, 4), (count, 1, 1)))
    rhs = np.zeros((2*nv, 4)); np.add.at(rhs, dofs.ravel(), force.reshape(-1, 4))
    lookup = {tuple(x): i for i, x in enumerate(vertices)}
    angles = angular_template(m)[0]
    circle = radius*np.column_stack((np.cos(angles), np.sin(angles)))
    gc = np.zeros((len(angles), 2, 4)); gc[:, 0, :2] = circle/n; gc[:, 1, 2:] = circle/n
    gc = gc.reshape(-1, 4)
    cache = {}; patches = []; exponents = []
    for i in range(n):
        for j in range(n):
            pattern = (int(base[i, j]), int(base[i-1, j]), int(base[i-1, j-1]), int(base[i, j-1]))
            if pattern not in cache: cache[pattern] = dtn(coeff, pattern, m)
            block, lam = cache[pattern]; exponents.append(lam)
            cp = np.round((circle+np.array([i, j])) % n, 11); cp[cp == n] = 0
            bn = np.array([lookup[tuple(p)] for p in cp]); bd = (2*bn[:, None]+np.arange(2)).ravel()
            row.append(np.broadcast_to(bd[:, None], block.shape).ravel())
            col.append(np.broadcast_to(bd[None, :], block.shape).ravel())
            entries.append(block.ravel())
            np.add.at(rhs, bd, -block @ gc)
            patches.append((bd, pattern))
    stiffness = coo_matrix((np.concatenate(entries), (np.concatenate(row), np.concatenate(col))), shape=(2*nv, 2*nv)).tocsc()
    del row, col, entries, values, force, allcoords, rounded, lookup
    mat = stiffness[2:, 2:]; diagonal = 1/np.sqrt(mat.diagonal())
    scaled = (diags(diagonal) @ mat @ diags(diagonal)).tocsc()
    lu = splu(scaled, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0., options={'SymmetricMode': True})
    correction = np.zeros_like(rhs)
    correction[2:] = diagonal[:, None]*lu.solve(diagonal[:, None]*rhs[2:])
    for unused in range(3):
        correction[2:] += diagonal[:, None]*lu.solve(diagonal[:, None]*(rhs[2:]-mat @ correction[2:]))
    residual = float(np.linalg.norm(stiffness @ correction-rhs)/max(np.linalg.norm(rhs), 1e-30))
    scaled_residual = float(np.linalg.norm(diagonal[:, None]*(mat @ correction[2:]-rhs[2:]))/np.linalg.norm(diagonal[:, None]*rhs[2:]))
    del lu, mat, stiffness, scaled
    eff = np.zeros((4, 4)); nne = len(elem_local)
    for start in range(0, count, 32):
        end = min(start+32, count)
        u = correction[dofs[start*nne:end*nne]].reshape(end-start, nne, 4, 2, 4) + g[None, :, :, :, :]
        u -= u[:, :, :1, :, :]
        chol = np.linalg.cholesky(coeff[base.ravel()[start:end]])
        weighted = np.einsum('pba,peibm->peiam', chol, u)
        eff += np.einsum('peiam,eij,pejan->mn', weighted, shape, weighted)
    for bd, pattern in patches:
        u = (correction[bd]+gc).reshape(-1, 2, 4)
        u -= u[:1]
        u = u.reshape(-1, 4)
        block = cache[pattern][0]
        eff += u.T @ block @ u
    eff = (eff+eff.T)/2
    factors = np.array([BETA, BETA, 1., 1.])
    physical = eff*ENERGY_SCALE/factors[:, None]/factors[None, :]
    sig = physical[:2, :2]; alpha = physical[:2, 2:]; seebeck = np.linalg.solve(sig, alpha)
    return compact({'m': m, 'radial_elements': nr, 'radius_per_pixel': radius, 'outer_gauss_order': ngauss,
        'vertices': nv, 'outer_elements': ne, 'smallest_angular_exponent': min(exponents),
        'sigma': sig.tolist(), 'alpha': alpha.tolist(), 'S_uV_K': (1e6*seebeck).tolist(),
        'Santi_main_uV_K': float(5e5*(seebeck[0, 1]-seebeck[1, 0])), 'scaled_effective_energy': eff.tolist(),
        'residual_relative': residual, 'scaled_residual_relative': scaled_residual, 'seconds': time.perf_counter()-started,
        'method': 'planar H1 trial space, gradient-factor SVD DtN, sum of local trial energies, with exact constant components removed before integration'})


def sector_pair(base, phases, m=4, nr=3, radius=.22, ngauss=10):
    assert np.array_equal(base, np.rot90(base))
    a = sector_homogenize(base, phases, m, nr, radius, ngauss)
    b = sector_homogenize(base, inverse_phases(phases), m, nr, radius, ngauss)
    enc = enclosure_from_pair(a, b)
    assert min(enc['gap_eigenvalues']) > -1e-8, enc['gap_eigenvalues']
    assert enc['sigma_interval'][1] >= enc['sigma_interval'][0]
    return {'fem': a, 'dual': b, 'enclosure': enc}
