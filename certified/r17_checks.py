"""Analytic high-contrast laminates and periodic topology controls."""
import json
import numpy as np
import r17_core as C
from r17_connectivity import ROOT, NAMES, topology
from r6_core import save


def main():
    out = {'laminates': [], 'topology_controls': {}, 'symmetry': {}}
    a = np.zeros((4, 4, 4), np.int32); a[:, 1, 1] = 1
    b = np.zeros_like(a); b[0, 1, 1] = b[3, 1, 1] = 1
    c = np.zeros_like(a); c[0, 0, 0] = c[1, 1, 1] = 1
    for name, labels, expected, components in [('x_rod', a, [True, False, False], 1),
                                               ('boundary_crossing_without_cycle', b, [False]*3, 1),
                                               ('vertex_contact_is_disconnected', c, [False]*3, 2)]:
        result = topology(labels)
        assert result['percolates_xyz'] == expected and result['component_count'] == components
        out['topology_controls'][name] = result
    phases = [(1e-6, 0., .3), (1e5, 200e-6, 1.2)]; h = [.7, .7, 1.3]
    coefficients = np.array([C.FE.energy(p) for p in phases])
    for axis in range(3):
        labels = np.zeros((4, 4, 4), np.int32); sl = [slice(None)]*3; sl[axis] = slice(2, None); labels[tuple(sl)] = 1
        exact = np.zeros((6, 6))
        for d in range(3):
            ids = [d, 3+d]
            exact[np.ix_(ids, ids)] = np.linalg.inv(np.linalg.inv(coefficients).mean(axis=0)) if axis == d else coefficients.mean(axis=0)
        # The serial electric conductivity is only 2e-6 S/m. Tighten the
        # proposal solve so a small solenoidal trial-flux error does not widen
        # the complementary bound despite a small global residual.
        first = C.fem(labels, phases, h, tol=1e-14); second = C.fv(labels, phases, h, tol=1e-14)
        diagonal = np.sqrt(np.diag(exact)); errors = {}
        for name, matrix in [('FEM', first['scaled_energy']), ('FV', second['scaled_energy']), ('RT0', second['continuum_energy_lower'])]:
            error = float(np.max(np.abs((np.array(matrix)-exact)/diagonal[:, None]/diagonal[None, :])))
            assert error < 1e-7, (axis, name, error)
            errors[name] = error
        out['laminates'].append({'axis': axis, 'phases': phases, 'h': h, 'exact_scaled_energy': exact.tolist(),
                                'relative_energy_metric_errors_P': errors, 'fem_P': first, 'fv_P': second})
        print('laminate', axis, errors, flush=True)
    geometry = json.loads((ROOT/'r13_geometries.json').read_text())
    for category, name in NAMES:
        labels = np.array(geometry[category][name]['labels'], dtype=np.int32)
        # Reuse r12 all-translation search, accepting only exact label equality.
        group = C.OLD.R.exact_group(labels)
        assert group['has_C4_z']
        out['symmetry'][name] = group
    meta = geometry['woodpile']['W2_split_clear']; phases = geometry['materials_sigma_S_kappa']
    labels = np.array(meta['labels'], np.int32)
    result = C.pair(labels, phases, meta['voxel_spacing_base'])
    old = json.loads((ROOT/'r13_woodpile.json').read_text())['cases']['W2_split_clear']['levels'][0]
    errors = {method: float(np.max(np.abs(np.array(result[method+'_P']['scaled_energy'])-old[method]['scaled_energy']))) for method in ('fem', 'fv')}
    assert max(errors.values()) < 1e-9
    out['original_weak_W2_reproduction'] = {'max_energy_difference_P': errors, 'result': result}
    out['all_passed'] = True
    save('r17_checks.json', out)


if __name__ == '__main__': main()
