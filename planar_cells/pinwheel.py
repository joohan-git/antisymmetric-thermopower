"""Stripe pinwheel cell of Fig. 1 and Fig. 2 (p4 symmetry).

cell(N, alpha, period, nph) returns an N x N integer phase map: stripes of nph phases at angle alpha (degrees) and
the given period (pixels) fill one quadrant, and 90-degree rotations about the cell centre complete the cell.
The reference cell of the manuscript is cell(24, 15, 6, 3); its mirror image is cell(24, 15, 6, 3)[::-1, :].
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import numpy as np


def cell(N, alpha, period, nph):
    h = N//2; x, y = np.meshgrid(np.arange(h)+0.5, np.arange(h)+0.5, indexing='ij')
    a = np.deg2rad(alpha); u = x*np.cos(a) + y*np.sin(a)
    Pq = (np.floor(u/(period/nph)) % nph).astype(int) + 1
    Cc = np.zeros((N, N), int); Cc[:h, :h] = Pq
    return Cc + np.rot90(Cc, 1) + np.rot90(Cc, 2) + np.rot90(Cc, 3) - 1
