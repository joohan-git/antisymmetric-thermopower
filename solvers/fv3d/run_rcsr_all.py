"""Finite-volume axial coefficients of the selected RCSR-derived structures (names from rcsr_filter.json).
Voxel maps: 16 voxels along a, strut radius 0.08a (rcsr_color.voxelize); colour group from every voxel translation
(rcsr_colour_pg.colour_group_full). Output rcsr_color_N16_all.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcsr_color as R, coloring as C, homog3d as H3, homog3d_it as HI
from rcsr_colour_pg import colour_group_full, symbol

if __name__ == '__main__':
    p, n, weak = H3.sloc(1e5, 200e-6, 1.2), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(10.45, 10e-6, 0.9012); mats = [weak, n, p]
    names = json.load(open(os.path.join(HERE, 'rcsr_filter.json')))['steps'][-1]['names']
    nets = {x['name']: x for x in R.read_cgd()}; out = []; t0 = time.time()
    for name in names:
        net = nets[name]; pts, cns = R.expand(net); segs = R.edges(net, pts, cns); lab = R.voxelize(net, segs, 16, 0.08)
        frac = np.bincount(lab.ravel(), minlength=3)/lab.size
        G = colour_group_full(lab); allowed = [bool(a) for a in C.axial_allowed(G)[0]]
        sig, L, S, ons, its = HI.homogenize3d_it(lab, mats); ax = H3.axial(S)*1e6
        out.append(dict(name=name, group=net['group'], nodes=len(pts), struts=len(segs), shape=list(lab.shape), frac=frac.tolist(),
                        colour_point_group=symbol(G), colour_group_order=len(G), allowed=allowed, ax=ax.tolist(), sxx=float(sig[0, 0]), szz=float(sig[2, 2])))
        print(name, net['group'], symbol(G), 'allowed', allowed, 'axial', np.round(ax, 3), 'frac', np.round(frac, 2), round(time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, 'rcsr_color_N16_all.json'), 'w'), indent=1, default=float)
