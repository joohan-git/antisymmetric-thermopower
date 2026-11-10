
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
from common.mats import m
out = {}
p, n = m('pBiSbTe'), m('nBiTeSe')
print('-- convergence, s=0.8 th=12 and s=0.7 th=16, CPC matrix')
for s, th in [(0.8, 12), (0.7, 16), (0.8, 8)]:
    for N in [96, 160, 256]:
        sig, L, S = H.homogenize(F.F2_rotsquares(N, th, s), [p, n, m('CPC')])
        Sa = 0.5*(S[0,1]-S[1,0])*1e6; out[f'conv|{s}|{th}|{N}'] = Sa
        print(s, th, N, 'Sa %.3f sxx %.1f' % (Sa, sig[0,0]), flush=True)
print('-- matrix conductivity sweep (S=10uV/K, kappa=0.3), s=0.7 th=16, N=160')
for sm in [1e1, 1e2, 1e3, 1e4, 3e4, 1e5]:
    sig, L, S = H.homogenize(F.F2_rotsquares(160, 16, 0.7), [p, n, H.mloc(sm, 10e-6, 0.3)])
    Sa = 0.5*(S[0,1]-S[1,0]); PFa = Sa**2*sig[0,0]
    out[f'mat|{sm}'] = dict(Sa=Sa*1e6, sxx=sig[0,0], PF_a_uW_mK2=PFa*1e6)
    print('sigma_m %.0e  Sa %.3f uV/K  sigma_eff %.1f  S_a^2 sigma = %.4f uW/mK^2' % (sm, Sa*1e6, sig[0,0], PFa*1e6), flush=True)
json.dump(out, open('planar_cells/f2_conv_matrix.json', 'w'), indent=1)
