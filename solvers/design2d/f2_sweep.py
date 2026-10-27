import sys, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '../fv2d')
import homog2 as H, families as F
from mats import m, MAT
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
json.dump(out, open('f2_sweep.json', 'w'), indent=1)
