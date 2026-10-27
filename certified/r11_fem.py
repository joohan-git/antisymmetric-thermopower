"""Independent 3D conforming Q1 thermoelectric FEM, periodic and finite.

Cube nodes, tensor-product Gauss integration, and coupled nodal potentials.
No main FV solver routines are used here. Energy variables are (V/beta,T).
Flux/gradient block order: electrical x,y,z; thermal x,y,z.
"""
import itertools
import time
import numpy as np
import pyamg
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

T0 = 300.
BETA = 200e-6
ESCALE = 1e5 * BETA**2
BITS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int32)


def energy(triple):
    s, a, k = triple
    return np.array([[s*BETA**2, s*BETA*a],
                     [s*BETA*a, k/T0+s*a*a]])/ESCALE


def tensors(phases):
    a = np.asarray(phases)
    if a.ndim == 3 and a.shape[1:] == (6, 6):
        return a
    return np.array([np.kron(energy(p), np.eye(3)) for p in phases])


def geometry(h):
    h = np.broadcast_to(np.asarray(h, float), (3,))
    signs = 2.*BITS-1
    geom = np.zeros((8, 8, 3, 3))
    vol = np.prod(h)
    for xi in itertools.product((-1/np.sqrt(3), 1/np.sqrt(3)), repeat=3):
        grad = np.zeros((8, 3))
        for d in range(3):
            others = [e for e in range(3) if e != d]
            grad[:, d] = signs[:, d]/(4*h[d])*np.prod(1+signs[:, others]*np.array(xi)[others], axis=1)
        geom += np.einsum('id,je->ijde', grad, grad)*vol/8
    return geom, vol*signs/(4*h), vol


def mesh_nodes(shape, periodic):
    sizes = np.array(shape)+(0 if periodic else 1)
    ijk = np.indices(shape).reshape(3, -1).T
    nodes = np.empty((len(ijk), 8), np.int32)
    for b, shift in enumerate(BITS):
        xyz = (ijk+shift) % sizes if periodic else ijk+shift
        nodes[:, b] = np.ravel_multi_index(xyz.T, sizes)
    return nodes, tuple(sizes)


def assemble(labels, phases, periodic=True, h=1., group=None):
    labels = np.asarray(labels, np.int32)
    coeff = tensors(phases)
    geom, ig, vol = geometry(h)
    ke = np.einsum('ijde,padbe->piajb', geom, coeff.reshape(-1, 2, 3, 2, 3)).reshape(-1, 16, 16)
    nodes, sizes = mesh_nodes(labels.shape, periodic)
    dofs = (2*nodes[:, :, None]+np.arange(2, dtype=np.int32)).reshape(-1, 16)
    nd = 2*int(np.prod(sizes))
    if group is not None:
        dofs = group[dofs].astype(np.int32)
        nd = int(group.max())+1
    vals = ke[labels.ravel()]
    rows = np.broadcast_to(dofs[:, :, None], vals.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], vals.shape).ravel()
    mat = coo_matrix((vals.ravel(), (rows, cols)), shape=(nd, nd)).tocsr()
    return mat, dofs, coeff, ig, vol, sizes


def solve_spd(mat, rhs, field_id=None, tol=2e-11):
    """Diagonal congruence scaling plus two-field smoothed aggregation CG."""
    rhs = np.asarray(rhs)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    scale = 1/np.sqrt(mat.diagonal())
    a = (diags(scale) @ mat @ diags(scale)).tocsr()
    if field_id is None:
        field_id = np.arange(len(scale)) % 2
    modes = np.column_stack((field_id == 0, field_id == 1)).astype(float)/scale[:, None]
    np.random.seed(11)
    ml = pyamg.smoothed_aggregation_solver(a, B=modes, max_coarse=250,
                                         symmetry='symmetric')
    pre = ml.aspreconditioner()
    sol = np.zeros_like(rhs)
    stats = []
    for j in range(rhs.shape[1]):
        b = scale*rhs[:, j]
        if np.linalg.norm(b) < 1e-25:
            stats.append({'iterations': 0, 'info': 0, 'relative_residual': 0.})
            continue
        count = [0]
        def callback(x):
            count[0] += 1
        x, info = cg(a, b, M=pre, rtol=tol, atol=1e-25, maxiter=2500, callback=callback)
        sol[:, j] = scale*x
        residual = np.linalg.norm(mat@sol[:, j]-rhs[:, j])/np.linalg.norm(rhs[:, j])
        stats.append({'iterations': count[0], 'info': int(info), 'relative_residual': float(residual)})
        if info != 0 or residual > 2e-7:
            raise RuntimeError(('CG failure', stats[-1]))
    return sol, stats


def properties(c):
    factors = np.array([BETA]*3+[1.]*3)
    physical = np.asarray(c)*ESCALE/factors[:, None]/factors[None, :]
    sig = physical[:3, :3]
    alpha = physical[:3, 3:]
    s = np.linalg.solve(sig, alpha)
    kap = T0*(physical[3:, 3:]-physical[3:, :3]@s)
    return {'scaled_energy': np.asarray(c).tolist(), 'sigma': sig.tolist(),
            'sigma_diag': np.diag(sig).tolist(), 'S_uV_K': (s*1e6).tolist(),
            'S_diag_uV_K': (np.diag(s)*1e6).tolist(),
            'Sa_vector_uV_K': (5e5*np.array([s[1,2]-s[2,1], s[2,0]-s[0,2], s[0,1]-s[1,0]])).tolist(),
            'Sa_z_uV_K': float(5e5*(s[0,1]-s[1,0])),
            'kappa_open_circuit': kap.tolist()}


def homogenize(labels, phases, tol=2e-11):
    start = time.perf_counter()
    mat, dofs, coeff, ig, vol, sizes = assemble(labels, phases)
    force = -np.einsum('id,padbe->piabe', ig, coeff.reshape(-1, 2, 3, 2, 3)).reshape(-1, 16, 6)
    rhs = np.zeros((mat.shape[0], 6))
    np.add.at(rhs, dofs.ravel(), force[labels.ravel()].reshape(-1, 6))
    u = np.zeros_like(rhs)
    u[2:], stats = solve_spd(mat[2:, 2:], rhs[2:], np.arange(2, mat.shape[0]) % 2, tol)
    volume = labels.size*vol
    c0 = np.mean(coeff[labels.ravel()], axis=0)
    flux = c0-rhs.T@u/volume
    # Actual admissible trial energy, valid even with a nonzero iterative residual.
    upper = c0-(rhs.T@u+u.T@rhs-u.T@(mat@u))/volume
    out = properties((upper+upper.T)/2)
    out.update({'classification': 'P: uniform-grid point estimate; energy matrix is a primal trial upper bound',
                'shape': list(labels.shape), 'dofs': mat.shape[0], 'solver': stats,
                'flux_energy_relative': float(np.linalg.norm(flux-upper)/np.linalg.norm(upper)),
                'reciprocity_relative': float(np.linalg.norm(flux-flux.T)/np.linalg.norm(flux)),
                'seconds': time.perf_counter()-start})
    return out


def finite(labels, phases, electrode_y_width=4, electrode_z_width=4, tol=3e-11):
    """Hot y=max, cold y=0; floating equipotential patches on x=0 and x=max.

    All other boundaries are thermally insulating; no external electric current.
    Electrodes have zero net current, not prescribed point potentials.
    Widths are in voxel units. None creates passive point probes, without metal.
    """
    start = time.perf_counter()
    nx, ny, nz = labels.shape
    sizes = np.array(labels.shape)+1
    coords = np.indices(sizes).reshape(3, -1).T
    nd = 2*len(coords)
    group = np.arange(nd)
    patches = []
    for x in (0, nx):
        if electrode_y_width is None:
            select = (coords[:,0] == x) & (coords[:,1] == ny//2) & (coords[:,2] == nz//2)
        else:
            select = ((coords[:,0] == x) & (abs(coords[:,1]-ny/2) <= electrode_y_width/2+1e-10)
                      & (abs(coords[:,2]-nz/2) <= electrode_z_width/2+1e-10))
        dofs = 2*np.flatnonzero(select)
        patches.append(dofs)
        if electrode_y_width is not None:
            group[dofs] = dofs[0]
    _, group = np.unique(group, return_inverse=True)
    mat, _, _, _, _, _ = assemble(labels, phases, periodic=False, group=group)
    field_id = np.empty(mat.shape[0], int)
    field_id[group] = np.arange(nd) % 2
    cold = group[2*np.flatnonzero(coords[:,1] == 0)+1]
    hot = group[2*np.flatnonzero(coords[:,1] == ny)+1]
    fixed = np.unique(np.r_[cold, hot, group[patches[0][0]]])
    u = np.zeros(mat.shape[0])
    u[hot] = 1.
    free = np.ones(len(u), bool)
    free[fixed] = False
    rhs = -(mat@u)[free]
    sol, stats = solve_spd(mat[free][:,free], rhs, field_id[free], tol)
    u[free] = sol[:,0]
    fields = u[group].reshape(-1, 2)
    voltage = -float(np.mean(u[group[patches[1]]]))*BETA*1e6
    reaction = mat@u
    mid = []
    for fraction in (.25, .5, .75):
        y = int(round(ny*fraction))
        li = np.ravel_multi_index((0,y,nz//2), sizes)
        ri = np.ravel_multi_index((nx,y,nz//2), sizes)
        mid.append({'y_over_H': y/ny,
                    'V_left_minus_right_uV': float((fields[li,0]-fields[ri,0])*BETA*1e6),
                    'T_left_minus_right_K': float(fields[li,1]-fields[ri,1])})
    return {'classification': 'P: uniform-grid finite-sample estimate', 'shape': list(labels.shape),
            'dofs_after_electrode_constraint': len(u), 'electrode_y_width_voxels': electrode_y_width,
            'electrode_z_width_voxels': electrode_z_width, 'V_left_minus_right_uV': voltage,
            'electrode_T_left_minus_right_K': float(np.mean(fields[patches[0]//2,1])-np.mean(fields[patches[1]//2,1])),
            'probe_positions': mid, 'solver': stats,
            'net_electrode_current_scaled': [float(reaction[np.unique(group[p])].sum()) for p in patches],
            'seconds': time.perf_counter()-start}
