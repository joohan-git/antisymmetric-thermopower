"""Colour groups of the straight-rod woodpiles of scan_woodpile_N24.json.

Each voxel map is rebuilt from its stored parameters with woodpile.woodpile (24 voxels, 6 per layer); the colour group is
the set of point operations that, with some voxel translation, leave every phase in place (rcsr_colour_pg.colour_group_full).
Output: woodpile_groups.json (per structure: colour point group, its order and the axial-vector verdict [x, y, z])."""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import woodpile as W
import coloring as C
from rcsr_colour_pg import colour_group_full, symbol

if __name__ == '__main__':
    N = 24
    rows = json.load(open(os.path.join(HERE, f'scan_woodpile_N{N}.json')))
    out = []
    for i, r in enumerate(rows):
        T = W.woodpile(N, N//4, r['c1'], r['c2'], r['w1'], r['w2'], 2, 1, 0)
        G = colour_group_full(T)
        allowed = [bool(a) for a in C.axial_allowed(G)[0]]
        out.append(dict(index=i, rods=r['rods'], matrix=r['matrix'], colour_point_group=symbol(G), colour_group_order=len(G), allowed=allowed, az=r['az']))
        print(i, symbol(G), len(G), allowed, flush=True)
    json.dump(out, open(os.path.join(HERE, 'woodpile_groups.json'), 'w'), indent=1)
