"""Three-material woodpile (log-pile) cells, the native geometry of extrusion printing.
Layer 0: two rods along x at y-centres c1, c2 (fractions of the cell), widths w1, w2, materials m1, m2, in matrix m0.
Layer k = rot90^k(layer 0) about the cell axis -> 4_1 screw axis along z (point group 4 when the rods break the mirrors)."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import numpy as np
def layer(N, c1, c2, w1, w2, m1, m2, m0):
    y = (np.arange(N) + 0.5)/N
    L = np.full((N, N), m0, int)
    for c, w, m in [(c1, w1, m1), (c2, w2, m2)]:
        d = np.abs(((y - c + 0.5) % 1.0) - 0.5)
        L[:, d < w/2] = m
    return L
def woodpile(N, nz, c1, c2, w1, w2, m1=2, m2=1, m0=0, sense=+1):
    L0 = layer(N, c1, c2, w1, w2, m1, m2, m0)
    T = np.zeros((N, N, 4*nz), int)
    for k in range(4): T[:, :, k*nz:(k+1)*nz] = np.rot90(L0, sense*k)[:, :, None]
    return T

def seg_layer(N, c1, c2, w1, w2, a, b, order1=(2, 1, 0), order2=(1, 2, 0), m0=0):
    """rods along x; each rod is segmented along x into three materials with boundaries at x=a and x=b (0<a<b<1)."""
    x = (np.arange(N) + 0.5)/N; y = (np.arange(N) + 0.5)/N
    seg = np.where(x < a, 0, np.where(x < b, 1, 2))
    L = np.full((N, N), m0, int)
    for c, w, order in [(c1, w1, order1), (c2, w2, order2)]:
        d = np.abs(((y - c + 0.5) % 1.0) - 0.5) < w/2
        for i in range(N):
            L[i, d] = order[seg[i]]
    return L
def seg_woodpile(N, nz, c1, c2, w1, w2, a, b, order1=(2, 1, 0), order2=(1, 2, 0), m0=0, stack='screw'):
    L0 = seg_layer(N, c1, c2, w1, w2, a, b, order1, order2, m0)
    T = np.zeros((N, N, 4*nz), int)
    for k in range(4):
        Lk = np.rot90(L0, k) if stack == 'screw' else (L0 if k % 2 == 0 else np.rot90(L0, 1))
        T[:, :, k*nz:(k+1)*nz] = Lk[:, :, None]
    return T
