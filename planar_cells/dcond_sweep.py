"""Material condition D != 0 (Supplementary Information).

A third phase whose scaled matrix is a linear combination of the p- and n-type matrices (D = 0) gives S_a = 0 in the
p4 pinwheel cell cell(48, 15, 12, 3); detuning its thermal conductivity by a factor f makes D and S_a non-zero.
Output: planar_cells/dcond_sweep.json with rows (f, D, S_a in uV/K).
"""

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
from common import homog2 as H
from planar_cells.pinwheel import cell

T0 = 300.0
G = cell(48, 15, 12, 3)
sa_, Sa_, ka_ = 1e5, 200e-6, 1.0
sb_, Sb_, kb_ = 1e5, -200e-6, 1.0
Sc = 0.35*Sa_ + 0.65*Sb_
q = 0.35*(ka_ + T0*sa_*Sa_**2) + 0.65*(kb_ + T0*sb_*Sb_**2)   # kappa + T0 sigma S^2 is linear in the mixture
kc0 = q - T0*1e5*Sc**2


def D(kc):
    rows = [[1, s_, k_/(T0*1e5) + s_**2] for s_, k_ in [(Sa_, ka_), (Sb_, kb_), (Sc, kc)]]
    return np.linalg.det(np.array(rows))


res = []
for f in [1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]:
    kc = kc0*f
    sig, L, S = H.homogenize(G, [H.mloc(sa_, Sa_, ka_), H.mloc(sb_, Sb_, kb_), H.mloc(1e5, Sc, kc)])
    res.append((f, D(kc), 0.5*(S[0, 1] - S[1, 0])*1e6))
    print(f, kc, D(kc), res[-1][2])
json.dump(dict(kc0=kc0, rows=res), open('planar_cells/dcond_sweep.json', 'w'), indent=1)
