import sys, json, itertools, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '../fv2d')
import homog2 as H, families as F
from mats import m, MAT
lib = ['pBiSbTe', 'nBiTeSe', 'Cu', 'Ni', 'PEDOT', 'CPC']
N = 64; res = []; t0 = time.time()
for trip in itertools.permutations(lib, 3):
    mats = [m(t) for t in trip]; Smax = max(abs(MAT[t][1]) for t in trip)
    for r in [0.2, 0.35, 0.5, 0.65]:
        for sp in [0.25, 0.5, 0.75]:
            sig, L, S = H.homogenize(F.F3_pinwheel_tiling(N, r, sp), mats)
            Sa = 0.5*(S[0,1]-S[1,0])
            res.append(dict(trip='/'.join(trip), r=r, split=sp, Sa=Sa*1e6, ratio=abs(Sa)/Smax, sxx=sig[0,0], PFa=Sa**2*sig[0,0]*1e6))
res.sort(key=lambda d: -abs(d['Sa']))
print('evals', len(res), 'sec', round(time.time()-t0))
for d in res[:12]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
json.dump(res, open('f3_real.json', 'w'), indent=0)
