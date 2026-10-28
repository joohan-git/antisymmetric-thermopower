"""Finite annulus Q1 FEM: resolved microstructure versus homogenized prediction.

Insulated electrical boundaries, fixed inner/outer temperatures, no electrodes.
The mesh is a staircase approximation of an annulus. Q1 integration and
constitutive normalization are shared only with our own periodic FEM.
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from common.fem2d import material_energy, pinwheel, PHASES, homogenize, BETA, ENERGY_SCALE


def annulus(nside, material_matrices, labels, box=.02, ri=.004, ro=.009, thickness=.001):
    started = time.perf_counter()
    h = box/nside
    ix, iy = np.indices((nside, nside))
    xx = (ix+.5)*h-box/2
    yy = (iy+.5)*h-box/2
    radius = np.hypot(xx, yy)
    active = (radius > ri) & (radius < ro)
    ix, iy = ix[active], iy[active]
    x, y = xx[active], yy[active]
    nodes = np.stack((ix*(nside+1)+iy, (ix+1)*(nside+1)+iy,
                      (ix+1)*(nside+1)+iy+1, ix*(nside+1)+iy+1), axis=1)
    used, inverse = np.unique(nodes, return_inverse=True)
    nodes = inverse.reshape(-1, 4)
    dofs = (2*nodes[:, :, None]+np.arange(2)).reshape(-1, 8)
    constitutive = material_matrices[labels[active]].reshape(-1, 2, 2, 2, 2)
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    shape = np.zeros((4, 4, 2, 2))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            deriv = np.column_stack((signs[:, 0]*(1+signs[:, 1]*eta),
                                     signs[:, 1]*(1+signs[:, 0]*xi)))/4
            shape += np.einsum('id,je->ijde', deriv, deriv)
    element = np.einsum('ijde,xadbe->xiajb', shape, constitutive).reshape(-1, 8, 8)
    count = len(nodes)
    row = np.broadcast_to(dofs[:, :, None], (count, 8, 8)).ravel()
    col = np.broadcast_to(dofs[:, None, :], (count, 8, 8)).ravel()
    matrix = coo_matrix((element.ravel(), (row, col)), shape=(2*len(used), 2*len(used))).tocsc()
    # Boundary nodes from exposed element faces. All exposed faces are
    # electrically insulated; only temperature is prescribed there.
    inner = set()
    outer = set()
    for dx, dy, face_nodes in ((-1, 0, (0, 3)), (1, 0, (1, 2)),
                                (0, -1, (0, 1)), (0, 1, (3, 2))):
        nx, ny = ix+dx, iy+dy
        neighbour_radius = np.hypot((nx+.5)*h-box/2, (ny+.5)*h-box/2)
        inside_hole = neighbour_radius <= ri
        outside = neighbour_radius >= ro
        inner.update(nodes[inside_hole][:, face_nodes].ravel().tolist())
        outer.update(nodes[outside][:, face_nodes].ravel().tolist())
    assert not (inner & outer)
    imposed = np.zeros(2*len(used))
    imposed[2*np.array(sorted(inner))+1] = 1.0
    fixed = np.concatenate(([0], 2*np.array(sorted(inner | outer))+1))
    free = np.setdiff1d(np.arange(len(imposed)), fixed)
    rhs = -matrix[free][:, fixed]@imposed[fixed]
    imposed[free] = splu(matrix[free][:, free].tocsc()).solve(rhs)
    residual = np.linalg.norm(matrix[free]@imposed)/max(np.linalg.norm(rhs), 1e-30)
    gradient = np.einsum('id,eia->ead', signs/(2*h), imposed[dofs].reshape(-1, 4, 2))
    flux = -np.einsum('eadbf,ebf->ead', constitutive, gradient)
    current = flux[:, 0, :]*ENERGY_SCALE/BETA
    cross = x*current[:, 1]-y*current[:, 0]
    r = np.hypot(x, y)
    # Angularly averaged current crossing a radial cut; unlike magnetic moment,
    # this excludes any contractible local closed-current loops in the continuum.
    winding_current = thickness*np.sum(cross/r**2)*h**2/(2*np.pi)
    moment = thickness*.5*np.sum(cross)*h**2
    b_center = 1e-7*thickness*np.sum(cross/r**3)*h**2
    return {'N': nside, 'elements': count, 'deltaT_K': 1., 'box_m': box,
            'ri_m': ri, 'ro_m': ro, 'thickness_m': thickness,
            'winding_current_A': float(winding_current), 'moment_A_m2': float(moment),
            'B_center_thin_sheet_T': float(b_center), 'residual': float(residual),
            'seconds': time.perf_counter()-started}


def predicted(periodic, ri=.004, ro=.009, thickness=.001):
    sig = periodic['sigma'][0][0]
    a = periodic['a_cross_z_uV_K']*1e-6
    current = thickness*sig*a
    return {'sigma_S_m': sig, 'a_V_K': a, 'deltaT_K': 1.,
            'winding_current_A': current,
            'moment_A_m2': np.pi*current*(ro**2-ri**2)/(2*np.log(ro/ri)),
            'B_center_thin_sheet_T': 2*np.pi*1e-7*current/np.log(ro/ri)*(1/ri-1/ro)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subcell', type=int, default=24)
    parser.add_argument('--tiles', type=int, nargs='+', default=[4, 8, 12])
    args = parser.parse_args()
    phases = [PHASES[p] for p in ('p', 'n', 'weak')]
    base = pinwheel(args.subcell, 15, 3)
    effective = homogenize(base, phases)
    mats = np.array([np.kron(material_energy(p), np.eye(2)) for p in phases])
    result = {'method': 'Finite annulus independent Q1 FEM; deltaT=1 K linear response',
              'periodic_cell': effective, 'homogeneous_prediction': predicted(effective), 'cases': {}}
    destination = (RELEASE_ROOT / ('devices/annulus_results.json'))
    def save(name, value):
        result['cases'][name] = value
        destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(name, json.dumps(value), flush=True)
    for tiles in args.tiles:
        labels = np.tile(base, (tiles, tiles))
        nside = labels.shape[0]
        save(f'microstructure_{tiles}_tiles', annulus(nside, mats, labels))
    save('homogeneous_validation', annulus(nside, np.array([effective['scaled_effective_energy']]), np.zeros_like(labels)))
    save(f'mirror_{tiles}_tiles', annulus(nside, mats, labels[::-1, :].copy()))


if __name__ == '__main__':
    main()
