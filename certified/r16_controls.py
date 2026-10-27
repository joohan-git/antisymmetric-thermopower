"""Bulk reference refinement and a homogeneous control for probe offsets."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import numpy as np
from r6_core import ROOT, save
from r8_finite import solve_domain, sample
from r15_sector import sector_pair


def main():
    source = json.loads((ROOT/'r16_measure.json').read_text())
    a = source['geometry']['A_gap4pct']
    path = ROOT/'r16_controls.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    if 'homogeneous_P' not in out:
        c = np.array(a['periodic_same_grid_P']['scaled_effective_energy'])
        rot = np.kron(np.eye(2), np.array([[0., -1.], [1., 0.]]))
        c = sum(np.linalg.matrix_power(rot, i) @ ((c+c.T)/2) @ np.linalg.matrix_power(rot, i).T for i in range(4))/4
        values = {}; length = .008; mid = length/2; delta = .001/8
        for axis in ('x', 'y'):
            cc = c if axis == 'y' else c[np.ix_([1, 0, 3, 2], [1, 0, 3, 2])]
            stats, state = solve_domain(np.zeros((16, 16), int), [cc], (length, length))
            values[axis] = {}
            for name, points in [('midpoint', [(0, mid), (length, mid)]), ('tilted', [(0, mid-delta), (length, mid+delta)])]:
                u = sample(state, points)
                values[axis][name] = {'Sapp_uV_K_P': float((u[0, 0]-u[1, 0])*200), 'probe_deltaT_K_P': float(u[1, 1]-u[0, 1])}
        s = np.linalg.solve(c[:2, :2], c[:2, 2:])*200
        values['bulk_S_uV_K_P'] = s.tolist()
        values['matched_tilt_half_difference_P'] = (values['y']['tilted']['Sapp_uV_K_P']-values['x']['tilted']['Sapp_uV_K_P'])/2
        values['matched_tilt_half_sum_P'] = (values['y']['tilted']['Sapp_uV_K_P']+values['x']['tilted']['Sapp_uV_K_P'])/2
        values['analytic_common_offset_P'] = s[0, 0]*(2*delta)/length
        assert abs(values['matched_tilt_half_difference_P']-(s[0, 1]-s[1, 0])/2) < 1e-8
        assert abs(values['matched_tilt_half_sum_P']-values['analytic_common_offset_P']) < 1e-8
        out['homogeneous_P'] = values; save(path.name, out)
    if 'gap_bulk_m4_C' not in out:
        result = sector_pair(np.array(a['base_labels']), a['phases_sigma_S_kappa'], 4, 4)
        out['gap_bulk_m4_C'] = result; save(path.name, out)
        print('gap bulk C', result['enclosure']['Sa_interval_uV_K'], flush=True)


if __name__ == '__main__': main()
