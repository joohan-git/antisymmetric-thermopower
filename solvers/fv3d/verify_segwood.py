import sys, json, time, glob, numpy as np
sys.path.insert(0, '.')
import homog3d as H3, homog3d_it as HI, woodpile as W
p, n, weak = H3.sloc(1e5, 200e-6, 1.2), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(10.45, 10e-6, 0.9012); mats = [weak, n, p]
R = []
for f in glob.glob('scan_segwood_N16_s*.json'): R += json.load(open(f))
R.sort(key=lambda r: -r['amax']); out = []
for r in R[:2]:
    prm = [r[k] for k in ('c1', 'c2', 'w1', 'w2', 'a', 'b')]
    for N, k in [(16, 1), (16, 2), (32, 1), (48, 1)]:
        t = time.time(); T = W.seg_woodpile(N, N//4, *prm, tuple(r['o1']), tuple(r['o2']), 0, r['stack'])
        if k > 1: T = np.kron(T, np.ones((k, k, k), int))
        sig, L, S, ons, its = HI.homogenize3d_it(T, mats); az = H3.axial(S)[2]*1e6
        res = dict(N=N, k=k, shape=list(T.shape), az=az, sxx=sig[0, 0], szz=sig[2, 2], ons=ons, sec=time.time()-t)
        if N == 32 and k == 1:
            sm, Lm, Sm, _, _ = HI.homogenize3d_it(T[::-1, :, :], mats); res['az_mirror'] = H3.axial(Sm)[2]*1e6
        out.append(dict(params=prm, o1=r['o1'], o2=r['o2'], stack=r['stack'], **res)); print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in res.items()}, flush=True)
json.dump(out, open('verify_segwood.json', 'w'), indent=1, default=float)
