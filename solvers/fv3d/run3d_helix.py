import sys, json, numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, '.')
import homog3d as H3
CPCm = H3.sloc(10.45, 10e-6, 0.9012); nm = H3.sloc(1e5, -180e-6, 1.3); pm = H3.sloc(1e5, 200e-6, 1.2)
mats = [CPCm, nm, pm]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
rng = np.random.default_rng(7)
f = gaussian_filter(rng.standard_normal((N, N)), 2.0, mode='wrap')
qs = np.quantile(f, [1/3, 2/3]); layer = np.digitize(f, qs)        # 3 phases, no symmetry (p1)
assert not np.array_equal(np.rot90(layer), layer)
q = max(N//4, 2)
def helix(k_sign):
    h = np.zeros((N, N, 4*q), int)
    for k in range(4): h[:, :, k*q:(k+1)*q] = np.rot90(layer, k_sign*k)[:, :, None]
    return h
out = {}
H41 = helix(+1)
cases = {'helix_4_1': H41, 'helix_mirror': H41[::-1, :, :], 'single_layer_extruded_p1': np.repeat(layer[:, :, None], 4, 2)}
for name, T in cases.items():
    sig, L, S, ons = H3.homogenize3d(T, mats); a = H3.axial(S)*1e6
    out[name] = dict(axial=a.tolist(), Sdiag=(np.diag(S)*1e6).tolist(), offdiag_sym=((S+S.T)/2*1e6).tolist(), ons=ons)
    print(name, 'axial', np.round(a, 4), 'Sdiag', np.round(np.diag(S)*1e6, 2), 'sym xy %.3f' % ((S[0,1]+S[1,0])/2*1e6), 'ons %.1e' % ons, flush=True)
json.dump(out, open(f'run3d_helix_N{N}.json', 'w'), indent=1)
