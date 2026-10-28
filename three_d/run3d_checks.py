
from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json; import numpy as np
from scipy.ndimage import gaussian_filter
from common import homog3d as H3
CPCm = H3.sloc(10.45, 10e-6, 0.9012); nm = H3.sloc(1e5, -180e-6, 1.3); pm = H3.sloc(1e5, 200e-6, 1.2)
N = 16; rng = np.random.default_rng(7)
f = gaussian_filter(rng.standard_normal((N, N)), 2.0, mode='wrap')
layer3 = np.digitize(f, np.quantile(f, [1/3, 2/3])); layer2 = (f > np.median(f)).astype(int)
def helix(layer, q=4):
    h = np.zeros((N, N, 4*q), int)
    for k in range(4): h[:, :, k*q:(k+1)*q] = np.rot90(layer, k)[:, :, None]
    return h
out = {}
for name, T, mats in [('2phase_helix_p_n', helix(layer2), [nm, pm]), ('2phase_helix_p_CPC', helix(layer2), [CPCm, pm])]:
    sig, L, S, ons = H3.homogenize3d(T, mats); a = H3.axial(S)*1e6; out[name] = a.tolist()
    print(name, 'axial', np.round(a, 6), flush=True)
T = helix(layer3)
for k in [1, 2]:
    Tk = np.kron(T, np.ones((k, k, k), int))
    sig, L, S, ons = H3.homogenize3d(Tk, [CPCm, nm, pm]); a = H3.axial(S)*1e6; out[f'3phase_refine_k{k}'] = a.tolist()
    print('3phase helix refine k', k, Tk.shape, 'axial z %.4f' % a[2], flush=True)
json.dump(out, open('three_d/run3d_checks.json', 'w'), indent=1)
