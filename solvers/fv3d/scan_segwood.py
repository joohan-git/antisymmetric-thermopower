import sys, json, time, numpy as np
sys.path.insert(0, '.')
import homog3d as H3, homog3d_it as HI, woodpile as W, coloring as C
p, n, weak = H3.sloc(1e5, 200e-6, 1.2), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(10.45, 10e-6, 0.9012)
mats = [weak, n, p]
N = int(sys.argv[1]); nz = N//4; rng = np.random.default_rng(int(sys.argv[2])); res = []; t0 = time.time()
for it in range(int(sys.argv[3])):
    c1, c2 = rng.uniform(0, 1, 2); w1, w2 = rng.uniform(0.2, 0.45, 2); a, b = np.sort(rng.uniform(0.1, 0.9, 2))
    if abs(((c1 - c2 + 0.5) % 1) - 0.5) < (w1 + w2)/2 + 1.0/N or b - a < 0.1: continue
    o1 = tuple(rng.permutation([0, 1, 2])); o2 = tuple(rng.permutation([0, 1, 2]))
    stack = 'screw' if rng.random() < 0.5 else 'alt'
    T = W.seg_woodpile(N, nz, c1, c2, w1, w2, a, b, o1, o2, 0, stack)
    G = C.colour_group(T); allowed = C.axial_allowed(G)[0]
    sig, L, S, ons, its = HI.homogenize3d_it(T, mats); ax = H3.axial(S)*1e6
    res.append(dict(c1=c1, c2=c2, w1=w1, w2=w2, a=a, b=b, o1=list(map(int, o1)), o2=list(map(int, o2)), stack=stack, group_order=len(G),
                    allowed=allowed, ax=ax.tolist(), amax=float(np.abs(ax).max()), sdiag=np.diag(sig).tolist()))
    print(it, stack, 'order', len(G), 'allowed', allowed, 'axial', np.round(ax, 3), round(time.time()-t0), flush=True)
json.dump(res, open(f'scan_segwood_N{N}_s{sys.argv[2]}.json', 'w'), indent=0, default=float)
