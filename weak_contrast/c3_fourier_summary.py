"""Consolidate all raw results into one JSON and compute extrapolations."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import numpy as np
from weak_contrast.c3_fourier_verify import ROOT, save


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    out = {name: read(file) for name, file in [
        ('geometry', 'weak_contrast/geometry_numbers.json'), ('exact_fourier', 'weak_contrast/exact_fourier_numbers.json'),
        ('fft', 'weak_contrast/fft_numbers.json'), ('finite_volume', 'weak_contrast/fv_numbers.json'),
        ('algebra', 'weak_contrast/algebra_numbers.json'), ('units', 'weak_contrast/units_check.json')]}
    out['extrapolations'] = {}
    for name in ['fft', 'finite_volume']:
        rows = out[name]['runs']
        estimates = []
        for k in sorted({row['k'] for row in rows}):
            by_eps = {row['eps']: row for row in rows if row['k']==k}
            small = .00625 if -.00625 in by_eps else .0125
            large = 2*small
            estimates.append(dict(k=k, h=large))
            for field, label in [('c_over_eps3', 'c3'), ('Sa_over_eps3_uV_K', 'Sa3_uV_K')]:
                even_large = (by_eps[large][field]+by_eps[-large][field])/2
                even_small = (by_eps[small][field]+by_eps[-small][field])/2
                estimates[-1][label] = (4*even_small-even_large)/3
        out['extrapolations'][name] = estimates
    rows = out['geometry']['geometry']
    # Empirical second-order mesh extrapolation, not a certified bound.
    out['extrapolations']['collocation_limit'] = {
        key: ((4*np.array(rows[-1][key])-np.array(rows[-2][key]))/3).tolist()
        for key in ['g', 'c3', 'Sa3_uV_K']}
    rows = out['exact_fourier']
    # Exact-cutoff sums converge at leading order 1/M; use three levels to
    # remove both 1/M and 1/M^2, independently of the collocation estimate.
    out['extrapolations']['exact_fourier_limit'] = {
        key: ((8*np.array(rows[-1][key])-6*np.array(rows[-2][key])+np.array(rows[-3][key]))/3).tolist()
        for key in ['g', 'c3', 'Sa3_uV_K']}
    for name in ['fft', 'finite_volume']:
        r = out['extrapolations'][name]
        out['extrapolations'][name+'_mesh_limit'] = {
            key: (4*r[-1][key]-r[-2][key])/3 for key in ['c3', 'Sa3_uV_K']}
    ref_runs = [r for r in out['fft']['runs'] if r['k']==8 and r['eps']>0]
    for maxeps in [.2, .05, .0125]:
        rr = [r for r in ref_runs if r['eps']<=maxeps]
        fit = np.polyfit(np.log([r['eps'] for r in rr]), np.log(np.abs([r['c'] for r in rr])), 1)
        out['extrapolations'][f'exponent_eps_le_{maxeps}'] = float(fit[0])
    rr = {r['eps']: r for r in out['fft']['runs'] if r['k']==8}
    eps = np.array([.025, .0125, .00625])
    codd = np.array([(rr[e]['c']-rr[-e]['c'])/2 for e in eps])
    out['extrapolations']['odd_part_exponent'] = float(np.polyfit(np.log(eps), np.log(np.abs(codd)), 1)[0])
    trace = out['fft']['equal_trace_relation']['runs']
    out['extrapolations']['equal_trace_c_over_eps4'] = [r['c']/r['eps']**4 for r in trace]
    out['extrapolations']['max_algebra_error'] = max(r['error'] for r in out['algebra']['random_materials'])
    save('weak_contrast/numbers.json', out)
    print(json.dumps(out['extrapolations'], indent=2))


if __name__ == '__main__':
    main()
