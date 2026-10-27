import sys, json, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.'); sys.path.insert(0, '../fv2d')
import homog2 as H, families as F
from mats import m, MAT
out = {}
# (a) realistic triples: optimise r, split, and CPC (log sigma in [1,4], kappa in [0.2,1.0]) at N=64, verify N=160
for trip in [('CPC', 'nBiTeSe', 'pBiSbTe'), ('CPC', 'nBiTeSe', 'PEDOT')]:
    Smax = max(abs(MAT[t][1]) for t in trip)
    def build(z):
        r, sp, ls, k = z
        mats = [H.mloc(10**ls, 10e-6, k) if t == 'CPC' else m(t) for t in trip]
        return mats, np.clip(r, .1, .8), np.clip(sp, .1, .9)
    def f(z, N=64):
        if not (1 <= z[2] <= 4 and 0.2 <= z[3] <= 1.0): return 0
        mats, r, sp = build(z); sig, L, S = H.homogenize(F.F3_pinwheel_tiling(N, r, sp), mats)
        return -abs(0.5*(S[0,1]-S[1,0]))*1e6
    best = None
    for z0 in [(0.5, 0.5, 3, 0.3), (0.6, 0.6, 2, 0.5), (0.4, 0.4, 3.5, 0.3)]:
        rr = minimize(f, z0, method='Nelder-Mead', options=dict(maxiter=120, xatol=1e-3, fatol=1e-3))
        if best is None or rr.fun < best.fun: best = rr
    mats, r, sp = build(best.x)
    for N in [64, 160]:
        sig, L, S = H.homogenize(F.F3_pinwheel_tiling(N, r, sp), mats); Sa = 0.5*(S[0,1]-S[1,0])
        out[f"{'/'.join(trip)}|{N}"] = dict(z=list(map(float, best.x)), Sa=Sa*1e6, ratio=abs(Sa)/Smax, sxx=sig[0,0], PFa=Sa**2*sig[0,0]*1e6)
        print('/'.join(trip), N, 'r %.3f split %.3f CPC sigma %.0f kappa %.2f | Sa %.2f ratio %.3f sigma %.0f PFa %.1f uW/mK2' %
              (r, sp, 10**best.x[2], best.x[3], Sa*1e6, abs(Sa)/Smax, sig[0,0], Sa**2*sig[0,0]*1e6), flush=True)
# (b) zT<=1 idealised F3 verification
d = json.load(open('boundzt1_F3_N40.json')); x = d['x']
mats = [H.mloc(10**x[3*i], x[3*i+1]*1e-6, 10**x[3*i+2]) for i in range(3)]; Smax = max(abs(x[1]), abs(x[4]), abs(x[7]))
for N in [80, 160]:
    sig, L, S = H.homogenize(F.F3_pinwheel_tiling(N, x[9], x[10]), mats); Sa = 0.5*(S[0,1]-S[1,0])*1e6
    out[f'zt1|{N}'] = dict(Sa=Sa, ratio=abs(Sa)/Smax, sxx=sig[0,0]); print('zT<=1 ideal F3', N, 'Sa %.2f ratio %.3f sigma %.0f' % (Sa, abs(Sa)/Smax, sig[0,0]), flush=True)
json.dump(out, open('f3_refine.json', 'w'), indent=1)
