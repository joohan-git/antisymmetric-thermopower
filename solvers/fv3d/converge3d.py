import sys, json, time, numpy as np
sys.path.insert(0, '.')
import homog3d as H3, homog3d_it as HI, woodpile as W, rcsr_color as R
p, n, weak = H3.sloc(1e5, 200e-6, 1.2), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(10.45, 10e-6, 0.9012); mats = [weak, n, p]
out = []
B = json.load(open('opt_segwood.json'))['best']; x = B['x']
for N in [32, 48, 64, 80]:
    t = time.time(); T = W.seg_woodpile(N, N//4, x[0] % 1, x[1] % 1, *x[2:], tuple(B['o1']), tuple(B['o2']), 0, B['stack'])
    s, L, S, ons, its = HI.homogenize3d_it(T, mats); az = H3.axial(S)[2]*1e6
    rec = dict(kind='woodpile_opt', N=N, az=az, sxx=s[0, 0], szz=s[2, 2], sec=time.time()-t)
    if N == 64: rec['az_mirror'] = H3.axial(HI.homogenize3d_it(T[::-1], mats)[2])[2]*1e6
    out.append(rec); print(rec, flush=True)
nets = {x['name']: x for x in R.read_cgd()}
for name, rad in [('cyc', 0.08), ('unm', 0.12), ('bbr', 0.08)]:
    net = nets[name]; pts, cns = R.expand(net); segs = R.edges(net, pts, cns)
    for N in [48, 64]:
        t = time.time(); lab = R.voxelize(net, segs, N, rad); s, L, S, ons, its = HI.homogenize3d_it(lab, mats); az = H3.axial(S)[2]*1e6
        rec = dict(kind=name, rad=rad, N=N, shape=list(lab.shape), az=az, sxx=s[0, 0], sec=time.time()-t); out.append(rec); print(rec, flush=True)
json.dump(out, open('converge3d.json', 'w'), indent=1, default=float)
