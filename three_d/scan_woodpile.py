
from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import time; import numpy as np
from common import homog3d as H3; from common import homog3d_it as HI; from three_d import woodpile as W
MAT = {'p': H3.sloc(1e5, 200e-6, 1.2), 'n': H3.sloc(1e5, -180e-6, 1.3), 'weak': H3.sloc(10.45, 10e-6, 0.9012), 'cpc': H3.sloc(1e3, 10e-6, 0.3)}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24; nz = N//4
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
assign = [('p', 'n', 'weak'), ('p', 'n', 'cpc'), ('p', 'weak', 'n'), ('n', 'weak', 'p')]   # rod1, rod2, matrix
res = []; t0 = time.time()
for it in range(int(sys.argv[3]) if len(sys.argv) > 3 else 60):
    c1, c2 = rng.uniform(0, 1, 2); w1, w2 = rng.uniform(0.15, 0.45, 2); a = assign[rng.integers(len(assign))]
    if abs(((c1 - c2 + 0.5) % 1) - 0.5) < (w1 + w2)/2 + 1.0/N: continue          # rods must not overlap
    T = W.woodpile(N, nz, c1, c2, w1, w2, 2, 1, 0)
    mats = [MAT[a[2]], MAT[a[1]], MAT[a[0]]]
    sig, L, S, ons, its = HI.homogenize3d_it(T, mats)
    ax = H3.axial(S)*1e6
    res.append(dict(c1=c1, c2=c2, w1=w1, w2=w2, rods=a[:2], matrix=a[2], az=ax[2], ax_inplane=float(np.hypot(ax[0], ax[1])), sxx=sig[0, 0], szz=sig[2, 2], Sxx=S[0, 0]*1e6, Szz=S[2, 2]*1e6, ons=ons))
res.sort(key=lambda r: -abs(r['az']))
print('evals', len(res), 'sec', round(time.time()-t0))
for r in res[:8]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(res, open(f'three_d/scan_woodpile_N{N}.json', 'w'), indent=0, default=float)
