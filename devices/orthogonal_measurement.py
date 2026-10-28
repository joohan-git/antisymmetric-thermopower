"""Orthogonal thermopower measurements: fixed rasters and passive point probes."""

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
import argparse
import gc
import json
import numpy as np
from common.transport2d import ROOT, save, solve, PHASES
from weak_contrast.total_matrix_contrast import cell
from certified_bounds.design_enclosures import readonly_inputs
from devices.contact import junction_gap
from common.finite_sample import solve_domain, sample
from common.fem2d import material_energy
from certified_bounds.high_contrast_sectors import sector_pair


def geometries():
    fam, mats = readonly_inputs()
    a = junction_gap(fam.F3_pinwheel_tiling(50, r=.6412, split=.9), .04)
    p, unused = cell()
    return {'A_gap4pct': (a, [(10.45, 10e-6, .9012), mats['nBiTeSe'], mats['pBiSbTe']]),
            'Fig2b_pinwheel': (p, [PHASES[k] for k in ('p', 'n', 'weak')])}


def direction(labels, phases, cells, axis):
    # Coordinate permutation only: in this solve u is transverse, v is thermal.
    lab = labels if axis == 'y' else labels.T.copy()
    length = cells*.001
    mats = [np.kron(material_energy(p), np.eye(2)) for p in phases]
    stats, state = solve_domain(lab, mats, (length, length))
    mid = length/2; pitch = .001
    pairs = {'midpoint': (mid, mid), 'tilt_plus': (mid-pitch/8, mid+pitch/8),
             'tilt_minus': (mid+pitch/8, mid-pitch/8), 'common_shift': (mid+pitch/4, mid+pitch/4)}
    samples = {}
    for name, (minus, plus) in pairs.items():
        values = sample(state, [(0, minus), (length, plus)])
        samples[name] = {'probe_coordinates_uv_m': [[0, minus], [length, plus]],
            'V_minus_minus_V_plus_uV_P': float((values[0, 0]-values[1, 0])*200),
            'T_plus_minus_T_minus_K_P': float(values[1, 1]-values[0, 1])}
    result = {'axis': axis, 'stats_P': stats, 'probes_P': samples}
    del state
    gc.collect()
    return result


def main(sizes, shifted):
    path = ROOT/'devices/orthogonal_measurement.json'
    out = json.loads(path.read_text()) if path.exists() else {
        'classification': 'P: uniform Cartesian Q1, open electrical boundaries and passive voltage probes; C bulk references evaluated in floating point.',
        'sign_convention': 'E=-grad(V). Hot at positive drive axis, DeltaT=1 K. S_yx_app=(V_bottom-V_top)/DeltaT for x drive; S_xy_app=(V_left-V_right)/DeltaT for y drive, for a square. Thus Ea=Sa grad(T) cross zhat with Sa=(Sxy-Syx)/2.',
        'geometry': {}, 'cases': {}}
    refined_path = ROOT/'devices/probe_controls.json'
    if refined_path.exists() and 'A_gap4pct' in out['geometry']:
        controls = json.loads(refined_path.read_text())
        if 'gap_bulk_m4_C' in controls:
            ref = controls['gap_bulk_m4_C']
            out['geometry']['A_gap4pct'].update(bulk_Sa_C_uV_K=ref['enclosure']['Sa_interval_uV_K'], bulk_reference=ref)
            for key, row in out['cases'].items():
                if key.startswith('A_gap4pct_'):
                    lo, hi = ref['enclosure']['Sa_interval_uV_K']
                    for value in row['measurements'].values():
                        sa = value['half_difference_Sa_uV_K_P']
                        value['ratio_to_bulk_C_interval_P'] = sorted([sa/lo, sa/hi])
            save(path.relative_to(ROOT).as_posix(), out)
    for name, (base, phases) in geometries().items():
        if name not in out['geometry']:
            per = solve(base, phases)
            if name == 'A_gap4pct':
                ref = sector_pair(base, phases, 2, 2)
                interval = ref['enclosure']['Sa_interval_uV_K']
            else:
                ref = json.loads((ROOT/'certified_bounds/planar_summary.json').read_text())['cases']['baseline']
                interval = ref['Sa_interval_uV_K']
            out['geometry'][name] = {'base_labels': base.tolist(), 'phases_sigma_S_kappa': phases,
                'periodic_same_grid_P': per, 'bulk_Sa_C_uV_K': interval, 'bulk_reference': ref}
            save(path.relative_to(ROOT).as_posix(), out)
        for n in sizes:
            shifts = [(0, 0)] + ([(len(base)//4, 0)] if shifted and n == 8 else [])
            for shift in shifts:
                key = f'{name}_N{n}_shift{shift[0]}_{shift[1]}'
                if key in out['cases']: continue
                raster = np.tile(np.roll(base, shift, axis=(0, 1)), (n, n))
                byaxis = {axis: direction(raster, phases, n, axis) for axis in ('x', 'y')}
                values = {}
                specs = {'midpoint': ('midpoint', 'midpoint'), 'matched_tilt': ('tilt_plus', 'tilt_plus'),
                    'one_axis_tilt': ('midpoint', 'tilt_plus'), 'opposed_tilt': ('tilt_minus', 'tilt_plus'),
                    'common_shift': ('common_shift', 'common_shift')}
                sxx = out['geometry'][name]['periodic_same_grid_P']['Sxx']
                for probe, (px, py) in specs.items():
                    mx = byaxis['x']['probes_P'][px]; my = byaxis['y']['probes_P'][py]
                    syx = mx['V_minus_minus_V_plus_uV_P']; sxy = my['V_minus_minus_V_plus_uV_P']
                    tyx = mx['T_plus_minus_T_minus_K_P']; txy = my['T_plus_minus_T_minus_K_P']
                    syxc = syx-sxx*tyx; sxyc = sxy-sxx*txy
                    sa = (sxy-syx)/2
                    lo, hi = out['geometry'][name]['bulk_Sa_C_uV_K']
                    values[probe] = {'Sxy_uV_K_P': sxy, 'Syx_uV_K_P': syx, 'half_difference_Sa_uV_K_P': sa,
                        'half_sum_symmetric_uV_K_P': (sxy+syx)/2,
                        'half_difference_temperature_corrected_uV_K_P': (sxyc-syxc)/2,
                        'half_sum_temperature_corrected_uV_K_P': (sxyc+syxc)/2,
                        'ratio_to_bulk_C_interval_P': sorted([sa/lo, sa/hi]),
                        'ratio_to_same_grid_bulk_P': sa/out['geometry'][name]['periodic_same_grid_P']['Sa']}
                row = {'classification': 'P', 'cells_per_side': n, 'base_pixels_per_cell': len(base),
                    'termination_shift_pixels': list(shift), 'termination_shift_fraction_cell': [x/len(base) for x in shift],
                    'directions': byaxis, 'measurements': values}
                out['cases'][key] = row; save(path.relative_to(ROOT).as_posix(), out)
                print(key, 'Sa', values['midpoint']['half_difference_Sa_uV_K_P'],
                      'sym', values['midpoint']['half_sum_symmetric_uV_K_P'],
                      'tilt', values['matched_tilt']['half_difference_Sa_uV_K_P'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sizes', type=int, nargs='+', default=[4, 8, 16]); ap.add_argument('--shifted', action='store_true')
    args = ap.parse_args(); main(args.sizes, args.shifted)
