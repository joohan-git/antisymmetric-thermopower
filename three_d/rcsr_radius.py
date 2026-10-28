
from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import numpy as np
from three_d import rcsr_color as R; from three_d import coloring as C; from common import homog3d as H3; from common import homog3d_it as HI
p, n, weak = H3.sloc(1e5, 200e-6, 1.2), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(10.45, 10e-6, 0.9012); mats = [weak, n, p]
nets = {x['name']: x for x in R.read_cgd()}; out = []
for name in ['cyc', 'unm', 'bbr']:
    net = nets[name]; pts, cns = R.expand(net); segs = R.edges(net, pts, cns)
    for rad in [0.06, 0.08, 0.10, 0.12, 0.15]:
        for N in [16, 32]:
            lab = R.voxelize(net, segs, N, rad); frac = np.bincount(lab.ravel(), minlength=3)/lab.size
            sig, L, S, ons, its = HI.homogenize3d_it(lab, mats); az = H3.axial(S)[2]*1e6
            out.append(dict(name=name, rad=rad, N=N, az=az, frac=frac.tolist(), sxx=sig[0, 0], szz=sig[2, 2])); print(name, rad, N, round(az, 2), np.round(frac, 2), round(sig[0, 0]), flush=True)
json.dump(out, open('three_d/rcsr_radius.json', 'w'), indent=1, default=float)
