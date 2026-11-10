"""Colour groups of the straight-rod woodpiles of scan_woodpile_N24.json.

Each voxel map is rebuilt from its stored parameters with woodpile.woodpile (24 voxels, 6 per layer); the colour group is
the set of point operations that, with some voxel translation, leave every phase in place (rcsr_colour_pg.colour_group_full).
Output: three_d/woodpile_groups.json (per structure: colour point group, its order and the axial-vector verdict [x, y, z])."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import os
import sys
HERE = str(RELEASE_ROOT)
from three_d import woodpile as W
from three_d import coloring as C
from three_d.rcsr_colour_pg import colour_group_full, symbol

if __name__ == '__main__':
    N = 24
    rows = json.load(open(os.path.join(HERE, f'three_d/scan_woodpile_N{N}.json')))
    out = []
    for i, r in enumerate(rows):
        T = W.woodpile(N, N//4, r['c1'], r['c2'], r['w1'], r['w2'], 2, 1, 0)
        G = colour_group_full(T)
        allowed = [bool(a) for a in C.axial_allowed(G)[0]]
        out.append(dict(index=i, rods=r['rods'], matrix=r['matrix'], colour_point_group=symbol(G), colour_group_order=len(G), allowed=allowed, az=r['az']))
        print(i, symbol(G), len(G), allowed, flush=True)
    json.dump(out, open(os.path.join(HERE, 'three_d/woodpile_groups.json'), 'w'), indent=1)
