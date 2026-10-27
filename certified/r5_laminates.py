"""Exact scale-separated laminate counterexample; no spatial discretization."""
import json
from pathlib import Path
import numpy as np
from r5_fem import material_energy, BETA, ENERGY_SCALE


def laminate(a, b, fraction_a, angle_deg):
    theta = np.deg2rad(angle_deg)
    normal = np.array([np.cos(theta), np.sin(theta)])
    normal_map = np.kron(np.eye(2), normal[:, None])
    jump = (a-b) @ normal_map
    denominator = normal_map.T @ ((1-fraction_a)*a + fraction_a*b) @ normal_map
    return (fraction_a*a + (1-fraction_a)*b
            - fraction_a*(1-fraction_a)*jump @ np.linalg.solve(denominator, jump.T))


def main():
    phase_a = (1e5, 200e-6, 1.)
    phase_b = (1e3, -200e-6, .1)
    a = np.kron(material_energy(phase_a), np.eye(2))
    b = np.kron(material_energy(phase_b), np.eye(2))
    first = laminate(a, b, .3, 0.)
    second = laminate(a, b, .7, 45.)
    result = laminate(first, second, .5, 70.)
    factors = np.array([BETA, BETA, 1., 1.])
    physical = result*ENERGY_SCALE/factors[:, None]/factors[None, :]
    sigma = physical[:2, :2]
    alpha = physical[:2, 2:]
    seebeck = np.linalg.solve(sigma, alpha)
    out = {
        'method': 'Exact hierarchical laminate: normal flux and tangential gradient continuity',
        'phase_A_sigma_S_kappa': phase_a, 'phase_B_sigma_S_kappa': phase_b,
        'construction': 'L1=(A:.3,B:.7), normal 0 deg; L2=(A:.7,B:.3), normal 45 deg; L=(L1:.5,L2:.5), normal 70 deg. Fine/coarse scales separated.',
        'sigma': sigma.tolist(), 'alpha': alpha.tolist(),
        'S_uV_K': (seebeck*1e6).tolist(),
        'Santi_main_uV_K': float((seebeck[0, 1]-seebeck[1, 0])*1e6/2),
        'alpha_symmetry_error': float(np.linalg.norm(alpha-alpha.T)),
        'energy_min_eigenvalue': float(np.linalg.eigvalsh(result).min()),
        'sigma_alpha_commutator_norm': float(np.linalg.norm(sigma@alpha-alpha@sigma)),
    }
    print(json.dumps(out, indent=2))
    Path(__file__).with_name('r5_laminate_counterexample.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
