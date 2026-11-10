"""Summarize material symmetry, high-contrast bounds and the cubic onset."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import math
from pathlib import Path
import numpy as np
from common.transport2d import ROOT, save


def interval(values, digits=3):
    scale = 10**digits
    return f'[{math.floor(values[0]*scale)/scale:.{digits}f}, {math.ceil(values[1]*scale)/scale:.{digits}f}]'


def axis(a):
    vals, vecs = np.linalg.eigh((a+a.T)/2)
    return float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])) % 180)


def main():
    read = lambda name: json.loads((ROOT/name).read_text(encoding='utf-8'))
    group = read('three_d/material_symmetry.json'); p1 = read('certified_bounds/rod_block_enclosures.json')
    weak = read('weak_contrast/total_matrix_contrast.json'); ins = read('certified_bounds/insulating_planar.json'); checks = read('validation/high_contrast_checks.json')
    finest = max(p1['sector'], key=lambda r: r['m'])
    assert finest['m'] == 4
    p1c = finest['two_phase_C']['Sa_interval_uV_K']
    aa = np.array(finest['fem_P']['sigma']); bb = np.array(finest['fem_P']['alpha'])
    axes = {'sigma_axis_deg_P': axis(aa), 'L_axis_deg_P': axis(bb),
        'relative_L_antisymmetry_P': float(np.linalg.norm(bb-bb.T)/np.linalg.norm(bb))}
    metrics = {'original': ins['original_sector'][-1]['metrics']}
    for name, case in ins['cases'].items():
        row = max(case['sector'], key=lambda x: x['fem']['m'])
        assert row['fem']['m'] == 4
        metrics[name] = row['metrics_C']
    proposed = {
        'symmetry': 'All 17 straight-rod rasters have material point group 422. Among the 82 segmented rasters, 46 have group 4, 35 have the horizontal-mirror group m, and one has group 422.',
        'p1': f'The original two-phase rod-and-block cell has misaligned electrical and thermoelectric principal axes, with a certified axial Seebeck interval of {interval(p1c, 4)} microvolts per kelvin for its fixed 64 by 64 raster.',
        'weak': 'For weak contrast in the full coupled transport matrices, the first symmetry-allowed axial contribution is cubic. The cubic coefficient is generically nonzero but can vanish for special material relations or geometries.',
        'insulator': f'For the fixed design-A raster, replacing the weak phase by an electrical insulator with conductivity 1e-6 S/m and thermal conductivity 0.3 or 1.0 W/(m K) gives certified axial Seebeck intervals of {interval(metrics["0.3"]["Sa_uV_K"])} and {interval(metrics["1.0"]["Sa_uV_K"])} microvolts per kelvin, respectively.'}
    summary = {'classification': 'C: continuum variational enclosure or exact algebraic/material-symmetry statement. P: numerical point estimate. Floating-point and quadrature errors are not interval-certified.',
        'symmetry': group['summary'], 'p1_Sa_C': p1c, 'p1_axes_P': axes,
        'weak_slope_P': weak['loglog_slopes_P'], 'weak_Taylor_coefficients_P': weak['Sa_Taylor_uV_K'],
        'quartic_counterexample': weak['controls']['traceless_cubic_cancellation'], 'design_A_metrics_C': metrics,
        'physical_interpretation': proposed, 'source_inputs_preserved': True,
        'archived_description_word_count': 859}
    save('certified_bounds/contrast_summary.json', summary)
    print('Saved certified_bounds/contrast_summary.json; p1 C', p1c, 'axes', axes, flush=True)


if __name__ == '__main__':
    main()
