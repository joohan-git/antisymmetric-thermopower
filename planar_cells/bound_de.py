"""Idealised upper-bound search: free phase properties within bounds + geometry params, per family.
Objective: |S_a| / max_i |S_i|  (S_i bounded by 200 uV/K). Electrical percolation enforced via sigma_eff floor."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import time; import numpy as np
from scipy.optimize import differential_evolution
from common import homog2 as H; from planar_cells import families as F
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
fam = sys.argv[1]
def mats_from(x):
    out = []
    for i in range(3):
        ls, S, lk = x[3*i:3*i+3]
        out.append(H.mloc(10**ls, S*1e-6, 10**lk))
    Smax = max(abs(x[1]), abs(x[4]), abs(x[7]))
    return out, Smax
mb = [(2, 6), (-200, 200), (np.log10(0.2), np.log10(5))]*3
if fam == 'F1':
    gb = [(3, 42), (0.1, 0.6), (0.1, 0.6)]
    geo = lambda g: F.F1_stripes(N, g[0], g[1], min(g[2], 0.9-g[1]))
elif fam == 'F2':
    gb = [(2, 40), (0.4, 0.9)]
    def geo(g):
        th, s = g; s = min(s, 0.95/(np.cos(np.deg2rad(th))+np.sin(np.deg2rad(th)))); return F.F2_rotsquares(N, th, s)
elif fam == 'F3':
    gb = [(0.1, 0.6), (0.1, 0.9)]
    geo = lambda g: F.F3_pinwheel_tiling(N, g[0], g[1])
best = {'f': 0}
def obj(x):
    mats, Smax = mats_from(x[:9])
    if Smax < 1: return 0.0
    try:
        sig, L, S = H.homogenize(geo(x[9:]), mats)
    except Exception:
        return 0.0
    if sig[0, 0] < 1.0: return 0.0
    r = abs(0.5*(S[0,1]-S[1,0]))*1e6 / Smax
    if r > best['f']: best.update(f=r, x=list(map(float, x)))
    return -r
t0 = time.time()
res = differential_evolution(obj, mb + gb, maxiter=int(sys.argv[3]) if len(sys.argv) > 3 else 25, popsize=10, seed=1, polish=False, tol=1e-6)
print(fam, 'N', N, 'best ratio', round(best['f'], 4), 'x', [round(v, 3) for v in best['x']], 'sec', round(time.time()-t0), flush=True)
json.dump(dict(fam=fam, N=N, ratio=best['f'], x=best['x']), open(f'planar_cells/bound_{fam}_N{N}.json', 'w'), indent=1)
