
from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import numpy as np
from common import homog2 as H; from planar_cells import families as F
from common.mats import m, MAT
trip = ['pBiSbTe', 'nBiTeSe', 'CPC']
mats = [m(t) for t in trip]
out = {}
for N in [64, 128]:
    for th in [-20, -10, 0, 5, 10, 15, 20, 25]:
        T = F.F2_rotsquares(N, th, .8)
        sig, L, S = H.homogenize(T, mats)
        Sa = 0.5*(S[0,1]-S[1,0])*1e6
        out[f'{N}|{th}'] = dict(Sa=Sa, Sxx=S[0,0]*1e6, sxx=sig[0,0])
        print(N, th, 'Sa %.3f  Sxx %.3f  sigma %.1f' % (Sa, S[0,0]*1e6, sig[0,0]), flush=True)
json.dump(out, open('planar_cells/f2_sweep.json', 'w'), indent=1)
