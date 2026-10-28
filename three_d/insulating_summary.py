"""Compact index (three_d/insulating_summary.json) of the complete insulating transport tensors."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path

ROOT = RELEASE_ROOT


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def interval(values, places=2, scale=1):
    quantum = Decimal(1).scaleb(-places)
    return '['+', '.join(format((Decimal(str(x))/Decimal(str(scale))).quantize(quantum, rounding=mode), 'f')
                         for x, mode in zip(values, (ROUND_FLOOR, ROUND_CEILING)))+']'


def main():
    data = read('three_d/insulating_transport.json'); topology = read('three_d/periodic_connectivity.json'); checks = read('validation/insulating_checks.json')
    assert checks['all_passed']
    rows = []; compact = {}; levels = []
    for name, display in [('W2_split_clear', 'W2'), ('W1_split', 'W1'), ('cyc', 'cyc'), ('unm', 'unm'), ('bbr', 'bbr')]:
        source = data['cases'][name]; graph = topology['cases'][name]['pn_union']
        assert graph['component_count'] == 1 and graph['percolates_xyz'] == [True]*3
        assert graph['components'][0]['winding_rank'] == 3
        compact[name] = {'percolates_xyz_exact': graph['percolates_xyz'], 'periodic_face_component_count_exact': 1,
                         'winding_rank_exact': 3, 'original_k2': {}, 'insulator_finest': {}}
        old = next(x for x in source['junction_transport_reference']['levels'] if x['k'] == 2)
        entries = [('weak', old['fem'], old['fv'], old['continuum_enclosure'])]
        compact[name]['original_k2'] = {'fem_Sa_uV_K_P': old['fem']['Sa_z_uV_K'], 'fv_Sa_uV_K_P': old['fv']['Sa_z_uV_K'],
                                        'Sa_interval_uV_K_C': old['continuum_enclosure']['Sa_z_interval_uV_K']}
        for kappa in ('0.3', '1.0'):
            candidates = source['insulator'][kappa]['levels']; levels.extend(candidates)
            current = next(x for x in candidates if x['k'] == 2)
            entries.append(('I'+kappa, current['fem_P'], current['fv_P'], current['enclosure_C']))
            finest = max(candidates, key=lambda x: x['k']); enc = finest['enclosure_C']
            assert enc['Sa_z_interval_uV_K'][0]*enc['Sa_z_interval_uV_K'][1] > 0
            compact[name]['insulator_finest'][kappa] = {'k': finest['k'],
                'Sa_interval_uV_K_C': enc['Sa_z_interval_uV_K'],
                'sigma_tensor_component_intervals_S_m_C': enc['sigma_tensor_component_intervals_S_m'],
                'fem_sigma_tensor_S_m_P': finest['fem_P']['sigma'], 'fv_sigma_tensor_S_m_P': finest['fv_P']['sigma'],
                'fem_Sa_uV_K_P': finest['fem_P']['Sa_z_uV_K'], 'fv_Sa_uV_K_P': finest['fv_P']['Sa_z_uV_K']}
    summary = {'classification': 'C floating-point variational continuum intervals; P uniform-grid point estimates; exact integer connectivity separately identified',
        'source_geometry_sha256': data['source_sha256'], 'cases': compact, 'all_five_face_percolate_xyz': True,
        'all_ten_material_cases_have_nonzero_Sa_C_at_finest_level': True,
        'validation_passed': checks['all_passed'], 'total_transport_seconds_P': sum(x['seconds_P'] for x in levels),
        'completed_material_geometry_levels': len(levels)}
    (ROOT/'three_d/insulating_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+chr(10), encoding='utf-8')
    print('calculations', len(levels))


if __name__ == '__main__': main()
