
from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import numpy as np
from scipy.optimize import minimize
from common import homog3d as H3; from common import homog3d_it as HI; from three_d import woodpile as W
p, n, weak = H3.sloc(1e5, 200e-6, 1.2), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(10.45, 10e-6, 0.9012); mats = [weak, n, p]
V = json.load(open('three_d/verify_segwood.json')); base = [v for v in V if v['N'] == 48][1]
o1, o2, stack = tuple(base['o1']), tuple(base['o2']), base['stack']; x0 = np.array(base['params'])
hist = []
def f(x, N=24):
    c1, c2, w1, w2, a, b = x
    if not (0.12 <= w1 <= 0.48 and 0.12 <= w2 <= 0.48 and 0.05 <= a < b - 0.08 and b <= 0.95): return 0.0
    if abs(((c1 - c2 + 0.5) % 1) - 0.5) < (w1 + w2)/2 + 1.0/N: return 0.0
    T = W.seg_woodpile(N, N//4, c1 % 1, c2 % 1, w1, w2, a, b, o1, o2, 0, stack)
    az = H3.axial(HI.homogenize3d_it(T, mats)[2])[2]*1e6; hist.append((list(map(float, x)), az)); print(round(az, 2), np.round(x, 3), flush=True)
    return -abs(az)
r = minimize(f, x0, method='Nelder-Mead', options=dict(maxiter=60, xatol=5e-3, fatol=0.05, initial_simplex=None))
xb = r.x; out = {'x': list(map(float, xb)), 'o1': o1, 'o2': o2, 'stack': stack}
for N in [24, 48]:
    T = W.seg_woodpile(N, N//4, xb[0] % 1, xb[1] % 1, *xb[2:], o1, o2, 0, stack); s, L, S, _, _ = HI.homogenize3d_it(T, mats)
    out[f'az_N{N}'] = H3.axial(S)[2]*1e6; out[f'sxx_N{N}'] = s[0, 0]; print('final N', N, out[f'az_N{N}'], flush=True)
json.dump(dict(best=out, hist=hist), open('three_d/opt_segwood.json', 'w'), indent=1, default=float)
