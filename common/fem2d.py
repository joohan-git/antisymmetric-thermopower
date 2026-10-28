"""Independent periodic bilinear Q1 FEM for linear thermoelectric homogenization.

Independent of the finite-volume solvers. Each pixel is a square finite
element, not a finite-volume face. Symmetric energy variables are (V/beta, T).
Array coordinates are [x, y]; flux blocks have [field_out, x/y, field_in, x/y].
The Seebeck convention is j = sigma * (E - S_eff grad T).
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

T0 = 300.0
BETA = 200e-6
ENERGY_SCALE = 1e5 * BETA**2
PHASES = {
    'p': (1e5, 200e-6, 1.0),
    'n': (1e5, -200e-6, 1.0),
    'metal': (1e7, 0.0, 50.0),
    'weak': (1e3, 0.0, 0.2),
    'contrast': (1e3, -100e-6, 10.0),
}


def material_energy(triple):
    sigma, seebeck, kappa = triple
    return np.array([[sigma*BETA**2, sigma*BETA*seebeck],
                     [sigma*BETA*seebeck, kappa/T0 + sigma*seebeck**2]]) / ENERGY_SCALE


def q1_geometry():
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    stiffness = np.zeros((4, 4))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            deriv = np.column_stack((signs[:, 0]*(1 + signs[:, 1]*eta),
                                     signs[:, 1]*(1 + signs[:, 0]*xi))) / 4
            stiffness += deriv @ deriv.T
    # Integral of physical gradient over a square of side h, divided by h.
    return stiffness, signs/2


def homogenize(labels, phases):
    started = time.perf_counter()
    nside = labels.shape[0]
    assert labels.shape == (nside, nside)
    h = 1/nside
    count = nside*nside
    ix, iy = np.indices(labels.shape)
    nodes = np.stack((ix*nside+iy, ((ix+1)%nside)*nside+iy,
                      ((ix+1)%nside)*nside+(iy+1)%nside,
                      ix*nside+(iy+1)%nside), axis=-1).reshape(count, 4)
    dofs = (2*nodes[:, :, None] + np.arange(2)).reshape(count, 8)
    constitutive = np.array([material_energy(x) for x in phases])[labels.ravel()]
    assert np.min(np.linalg.eigvalsh(constitutive)) > 0
    element_shape, integral_shape = q1_geometry()
    element_matrix = np.einsum('ij,eab->eiajb', element_shape, constitutive).reshape(count, 8, 8)
    row = np.broadcast_to(dofs[:, :, None], (count, 8, 8)).ravel()
    col = np.broadcast_to(dofs[:, None, :], (count, 8, 8)).ravel()
    stiffness = coo_matrix((element_matrix.ravel(), (row, col)), shape=(2*count, 2*count)).tocsc()
    # Four imposed macroscopic gradients, ordered (Vx, Vy, Tx, Ty).
    forcing = -h*np.einsum('id,eab->eiabd', integral_shape, constitutive).reshape(count, 8, 4)
    rhs = np.zeros((2*count, 4))
    np.add.at(rhs, dofs.ravel(), forcing.reshape(-1, 4))
    correction = np.zeros_like(rhs)
    correction[2:] = splu(stiffness[2:, 2:]).solve(rhs[2:])
    residual = np.linalg.norm(stiffness@correction-rhs)/max(np.linalg.norm(rhs), 1e-30)
    gradients = np.einsum('id,eiaq->eadq', integral_shape/h,
                          correction[dofs].reshape(count, 4, 2, 4))
    gradients += np.eye(4).reshape(1, 2, 2, 4)
    # Direct mean flux: do not symmetrize or impose Onsager after solving.
    effective_scaled = np.einsum('eab,ebdq->adq', constitutive, gradients).reshape(4, 4)/count
    factors = np.array([BETA, BETA, 1., 1.])
    effective = effective_scaled*ENERGY_SCALE/factors[:, None]/factors[None, :]
    sigma = effective[:2, :2]
    alpha = effective[:2, 2:]
    seebeck = np.linalg.solve(sigma, alpha)
    kappa_oc = T0*(effective[2:, 2:] - effective[2:, :2]@np.linalg.solve(sigma, alpha))
    # Schur-energy formula supplies a second extraction, independent of flux averaging.
    energy_check = np.kron(constitutive.mean(axis=0), np.eye(2)) - rhs.T@correction
    return {
        'N': nside, 'fractions': np.bincount(labels.ravel(), minlength=len(phases)).tolist(),
        'sigma': sigma.tolist(), 'alpha': alpha.tolist(), 'S_uV_K': (1e6*seebeck).tolist(),
        'kappa_open_circuit': kappa_oc.tolist(),
        'Santi_main_uV_K': float(0.5e6*(seebeck[0, 1]-seebeck[1, 0])),
        'a_cross_z_uV_K': float(0.5e6*(seebeck[1, 0]-seebeck[0, 1])),
        'alpha_antisymmetric': float(0.5*(alpha[0, 1]-alpha[1, 0])),
        'scaled_effective_energy': effective_scaled.tolist(),
        'onsager_relative': float(np.linalg.norm(effective_scaled-effective_scaled.T)/np.linalg.norm(effective_scaled)),
        'flux_energy_difference': float(np.linalg.norm(effective_scaled-energy_check)/np.linalg.norm(effective_scaled)),
        'residual_relative': float(residual),
        'seconds': time.perf_counter()-started,
    }


def pinwheel(nside, angle_deg, phase_count):
    """Continuous quadrant-fold construction of the specified pinwheel geometry.

    Base square: 0<x,y<1/2; stripes normal to (cos angle,sin angle),
    repeat period 1/4. Other quadrants are obtained by fourfold rotation.
    This specifies the geometry, not the construction algorithm of common.
    """
    x, y = np.meshgrid((np.arange(nside)+0.5)/nside,
                       (np.arange(nside)+0.5)/nside, indexing='ij')
    region = (x >= 0.5).astype(int) + 2*(y >= 0.5).astype(int)
    xb = np.choose(region, (x, y, 1-y, 1-x))
    yb = np.choose(region, (y, 1-x, x, 1-y))
    theta = np.deg2rad(angle_deg)
    labels = (np.floor(4*phase_count*(xb*np.cos(theta)+yb*np.sin(theta))).astype(int) % phase_count)
    assert np.array_equal(labels, np.rot90(labels))
    return labels


def asymmetric_geometry(nside):
    x, y = np.meshgrid((np.arange(nside)+0.5)/nside,
                       (np.arange(nside)+0.5)/nside, indexing='ij')
    inside = np.zeros_like(x, dtype=bool)
    for cx, cy, a, b, deg in ((0.28, 0.29, 0.24, 0.055, 22),
                             (0.67, 0.66, 0.23, 0.10, -47),
                             (0.28, 0.76, 0.13, 0.045, 65)):
        angle = np.deg2rad(deg)
        u = (x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)
        v = -(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)
        inside |= (u/a)**2+(v/b)**2 < 1
    return inside.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[24, 48, 96])
    parser.add_argument('--output', default='validation/fem2d_results.json')
    args = parser.parse_args()
    results = {'method': 'Independent periodic Q1 FEM, 2x2 Gauss integration, symmetric scaled energy variables',
               'T0_K': T0, 'phase_inputs_sigma_S_kappa': PHASES, 'cases': {}}
    destination = (RELEASE_ROOT / (args.output))
    def run(name, labels, phases):
        result = homogenize(labels, [PHASES[x] for x in phases])
        results['cases'][name] = result
        print(name, 'N', result['N'], 'S', np.array(result['S_uV_K']),
              'main Sa', result['Santi_main_uV_K'], 'Onsager', result['onsager_relative'],
              'seconds', round(result['seconds'], 3), flush=True)
        destination.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
        return result
    layers = np.zeros((16, 16), dtype=int)
    layers[8:] = 1
    lam = run('laminate', layers, ['p', 'metal'])
    phase_matrices = [material_energy(PHASES[x]) for x in ('p', 'metal')]
    normal = np.linalg.inv(sum(np.linalg.inv(x) for x in phase_matrices)/2)
    tangent = sum(phase_matrices)/2
    exact = np.zeros((4, 4))
    exact[np.ix_([0, 2], [0, 2])] = normal
    exact[np.ix_([1, 3], [1, 3])] = tangent
    assert np.max(np.abs(np.array(lam['scaled_effective_energy'])-exact)) < 1e-9
    results['laminate_exact_S_uV_K'] = [196.078431372549, 1.98019801980198]
    for nside in args.sizes:
        run(f'two_p4_{nside}', pinwheel(nside, 15, 2), ['p', 'metal'])
        run(f'three_p4_{nside}', pinwheel(nside, 15, 3), ['p', 'n', 'weak'])
        run(f'two_low_symmetry_{nside}', asymmetric_geometry(nside), ['p', 'contrast'])
    nside = args.sizes[-1]
    run(f'three_mirror_{nside}', pinwheel(nside, 15, 3)[::-1, :].copy(), ['p', 'n', 'weak'])
    run(f'three_mirror_symmetric_{nside}', pinwheel(nside, 45, 3), ['p', 'n', 'weak'])
    destination.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
