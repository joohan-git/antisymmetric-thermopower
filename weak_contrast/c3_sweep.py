"""Axial thermopower of the reference pinwheel along the midpoint-reference path s_i(eps) = s0 + eps (s_i - s0),
s0 = (s_p + s_n)/2, which reaches the real materials at eps = 1 (whitened contrast norm 0.99 < 1, so the series
converges). Finite volumes on the 4-fold refined pixel grid; compared with the cubic term -0.160870 eps^3 uV/K
(Fourier extrapolation of the closed form for this path). Output c3_sweep.json."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import numpy as np
import os
from common import homog2 as H2
from weak_contrast import c3 as M

T0 = M.T0
S = {k: M.smat(*v) for k, v in {'p': (1e5, 200e-6, 1.0), 'n': (1e5, -200e-6, 1.0), 'w': (1e3, 0.0, 0.2)}.items()}
s0 = 0.5 * (S['p'] + S['n'])
C3 = -0.160870                                   # uV/K per eps^3, midpoint path (weak_contrast/numbers.json: Fourier and finite-volume limits)
lab = np.kron(M.cell(24, 15, 6, 3), np.ones((4, 4), int))
rows = []
for eps in np.concatenate([np.geomspace(0.003, 0.3, 9), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]):
    mats = []
    for key in ('p', 'n', 'w'):
        s = s0 + eps * (S[key] - s0)
        mats.append(np.array([[s[0, 0], s[0, 1]], [T0 * s[0, 1], T0 * s[1, 1]]]))
    sig, L, Se = H2.homogenize(lab, mats)
    Sa = 0.5 * (Se[0, 1] - Se[1, 0]) * 1e6
    rows.append({'eps': float(eps), 'Sa_uV_K': float(Sa), 'cubic_uV_K': C3 * eps ** 3})
    print(f'{eps:.4f}  Sa {Sa: .6e}  cubic {C3 * eps ** 3: .6e}  ratio {Sa / (C3 * eps ** 3):.4f}', flush=True)
json.dump({'path': 'midpoint reference s0 = (s_p + s_n)/2; s_i(eps) = s0 + eps (s_i - s0)', 'grid': '24 x 24 pinwheel, 4-fold refined, finite volumes',
           'cubic_coefficient_uV_K': C3, 'rows': rows}, open('weak_contrast/c3_sweep.json', 'w'), indent=1)
