"""Assemble the compact verified result index (certified_bounds/verified_summary.json) from completed data."""

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


def interval(values, places=6):
    quantum = Decimal(1).scaleb(-places)
    return '['+', '.join(format(Decimal(str(x)).quantize(quantum, rounding=mode), 'f')
                         for x, mode in zip(values, (ROUND_FLOOR, ROUND_CEILING)))+']'


def main():
    measurements = read('devices/orthogonal_measurement.json')
    controls = read('devices/probe_controls.json')
    check = read('certified_bounds/certificate_checks.json')
    assert check['all_passed'] and len(check['records']) == 7
    certificates = [read(f'certified_bounds/verified_{kind}_k{k}.json')
                    for kind, levels in [('2d', [1, 2, 4, 8, 16]), ('3d', [1, 2])] for k in levels]
    gap_c = controls['gap_bulk_m4_C']['enclosure']['Sa_interval_uV_K']
    pin_c = measurements['geometry']['Fig2b_pinwheel']['bulk_Sa_C_uV_K']
    assert measurements['geometry']['A_gap4pct']['bulk_Sa_C_uV_K'] == gap_c
    compact = {
        'classification_legend': {'C': 'floating-point variational continuum enclosure', 'P': 'uniform-grid point estimate or computational timing', 'V': 'ball-arithmetic verified continuum enclosure'},
        'bulk_Sa_uV_K_C': {'A_gap4pct': gap_c, 'Fig2b_pinwheel': pin_c},
        'finite_cases_P': {k: {field: row[field] for field in ('cells_per_side', 'base_pixels_per_cell', 'termination_shift_fraction_cell', 'measurements')} for k, row in measurements['cases'].items()},
        'verified_cases': [{field: row[field] for field in ('geometry', 'k', 'shape', 'Sa_interval_uV_K_V_decimal', 'solver_seconds_P', 'verification_seconds_P', 'total_seconds_P', 'trial_archive', 'trial_sha256')} for row in certificates],
        'check_all_passed': check['all_passed'],
        'scope_limitations': ['Finite samples use passive point probes and unrefined within-cell rasters.', 'C enclosures are not relabelled as V.', 'V assumes exact decimal materials, the recorded fixed raster, ideal interfaces, and the linear model at 300 K.', 'Nonpolynomial sector quadrature is not verified here.'],
    }
    (ROOT/'certified_bounds/verified_summary.json').write_text(json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False)+chr(10), encoding='utf-8')
    print('Saved certified_bounds/verified_summary.json')


if __name__ == '__main__': main()
