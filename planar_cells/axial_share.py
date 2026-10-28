"""Local decomposition of the axial coefficient of design A (Fig. 3c).

Design A (pinwheel tiling family F3, r = 0.6412, split = 0.9; weak / n / p phases) on a 128 x 128 raster. The exact
face-wise decomposition of the cross block of the energy matrix (planar_cells/axial_density.py) gives the axial density; the
script reports the share of the axial integral carried by cells next to phase boundaries and by each phase interior.
Output: axial_share.json.
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import os; import sys; import json
import numpy as np
from planar_cells import families as F
N = 128
T = F.F3_pinwheel_tiling(N, 0.6412, 0.9)
from planar_cells import axial_density as AD
sig2, L2, S2, cross = AD.solve(T, [AD.sloc(10.45, 10e-6, 0.9012), AD.sloc(1e5, -180e-6, 1.3), AD.sloc(1e5, 200e-6, 1.2)])
sig_eff = sig2[0, 0]; Sa_exact = 0.5*(S2[0, 1]-S2[1, 0])*1e6
# share of the axial integral carried by cells adjacent to a phase boundary vs phase interiors
bnd = np.zeros_like(T, bool)
for d in range(2):
    for sh in (1, -1): bnd |= (np.roll(T, sh, axis=d) != T)
share = {'boundary_cells_fraction_of_area': float(bnd.mean()), 'boundary_share_of_axial': float(cross[bnd].sum()/cross.sum())}
for ph, nm in enumerate(['third', 'n', 'p']): share[f'interior_{nm}_share'] = float(cross[(~bnd) & (T == ph)].sum()/cross.sum())
json.dump(dict(N=N, sigma_eff=float(sig_eff), Sa_exact_uVK=float(Sa_exact), **share), open('planar_cells/axial_share.json', 'w'), indent=1)
print('sigma_eff', sig_eff, 'Sa', Sa_exact, share)
