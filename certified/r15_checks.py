"""Numerical audit for the new high-contrast sector evaluation and general p1 bounds."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import numpy as np
from r6_core import save, inverse_phases, enclosure_from_pair
from r15_sector import sector_homogenize
from r15_continuum import generic_enclosure


def main():
    checker = np.array([[0, 1], [1, 0]])
    out = {'checkerboard': [], 'p1_enclosure_sampling': {}}
    from r6_sector import sector_homogenize as old_sector
    lab = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 1]])
    phases = [(10.45, 10e-6, .9012), (1e5, -180e-6, 1.3), (1e5, 200e-6, 1.2)]
    old = old_sector(lab, phases, 2, 2)
    new = sector_homogenize(lab, phases, 2, 2)
    discrepancy = float(np.max(np.abs(np.array(old['scaled_effective_energy'])-new['scaled_effective_energy'])))
    assert discrepancy < 1e-9
    out['moderate_contrast_three_phase_crosscheck'] = {'max_scaled_energy_difference_P': discrepancy, 'old': old, 'new': new}
    for contrast in (1e4, 1e11):
        phases = [(1e5/contrast, 0., 1.), (1e5, 0., 1.)]
        exact = 1e5/np.sqrt(contrast)
        for m in (1, 2, 4):
            primal = sector_homogenize(checker, phases, m, m)
            dual = sector_homogenize(checker, inverse_phases(phases), m, m)
            row = {'fem': primal, 'dual': dual, 'enclosure': enclosure_from_pair(primal, dual)}
            lo, hi = row['enclosure']['sigma_interval']
            row.update(contrast=contrast, exact_sigma_C=float(exact), contains_exact_C=bool(lo <= exact <= hi))
            assert lo <= exact <= hi, (contrast, m, lo, exact, hi)
            out['checkerboard'].append(row)
            print('checker', contrast, m, lo, exact, hi, flush=True)
    data = json.load(open('r15_p1.json'))
    r = data['sector'][-1]
    enc = r['enclosure_C']; lo = np.array(enc['lower_energy']); hi = np.array(enc['upper_energy'])
    w, v = np.linalg.eigh(hi-lo); root = (v*np.sqrt(np.maximum(w, 0))) @ v.T
    rng = np.random.default_rng(1515); vals = []
    for unused in range(1000):
        q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        x = (q*rng.uniform(0, 1, 4)) @ q.T
        c = lo + root @ x @ root
        s = np.linalg.solve(c[:2, :2], c[:2, 2:])*200
        a = (s[0, 1]-s[1, 0])/2
        assert enc['Sa_interval_uV_K'][0] <= a <= enc['Sa_interval_uV_K'][1]
        vals.append(a)
    out['p1_enclosure_sampling'] = {'seed': 1515, 'samples': 1000, 'm': r['m'], 'sampled_minmax_P': [min(vals), max(vals)],
        'enclosure_C': enc['Sa_interval_uV_K'], 'scope': 'Implementation check only; analytic norm remainder supplies the enclosure, not sampling.'}
    if 'two_phase_C' in r:
        accepted = []
        for unused in range(1000):
            q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
            x = (q*rng.uniform(.3, .7, 4)) @ q.T
            c = lo + root @ x @ root
            b = (c[:2, 2:]+c[:2, 2:].T)/2
            c[:2, 2:] = b; c[2:, :2] = b.T
            if min(np.linalg.eigvalsh(c-lo)) < 0 or min(np.linalg.eigvalsh(hi-c)) < 0: continue
            s = np.linalg.solve(c[:2, :2], b)*200
            a = (s[0, 1]-s[1, 0])/2
            assert r['two_phase_C']['Sa_interval_uV_K'][0] <= a <= r['two_phase_C']['Sa_interval_uV_K'][1]
            accepted.append(a)
        assert len(accepted) > 100
        out['p1_two_phase_sampling'] = {'accepted_samples': len(accepted), 'sampled_minmax_P': [float(min(accepted)), float(max(accepted))], 'enclosure_C': r['two_phase_C']['Sa_interval_uV_K']}
    save('r15_checks.json', out)


if __name__ == '__main__':
    main()
