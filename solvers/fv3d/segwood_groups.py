"""Colour-group symbols of the segmented woodpiles of scan_segwood_N16_s*.json.

Each voxel map is rebuilt from its stored parameters with woodpile.seg_woodpile (16 voxels, 4 per layer); the colour group is
the set of point operations that, with some voxel translation, leave every phase in place (rcsr_colour_pg.colour_group_full).
Output: segwood_groups.json (per structure: file, index, colour point group, its order, axial-vector verdict, stored order)."""
import glob
import json
import os
import sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import woodpile as W
import coloring as C
from rcsr_colour_pg import colour_group_full, symbol

if __name__ == '__main__':
    N = 16
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, 'scan_segwood_N16_s*.json'))):
        for i, r in enumerate(json.load(open(f))):
            T = W.seg_woodpile(N, N//4, r['c1'] % 1, r['c2'] % 1, r['w1'], r['w2'], r['a'], r['b'], tuple(r['o1']), tuple(r['o2']), 0, r['stack'])
            G = colour_group_full(T)
            out.append(dict(file=os.path.basename(f), index=i, colour_point_group=symbol(G), colour_group_order=len(G),
                            allowed=[bool(a) for a in C.axial_allowed(G)[0]], stored_group_order=r['group_order'], stored_allowed=r['allowed']))
    json.dump(out, open(os.path.join(HERE, 'segwood_groups.json'), 'w'), indent=1)
    print(len(out), 'structures', dict(Counter(o['colour_point_group'] for o in out)),
          'order mismatches', sum(o['colour_group_order'] != o['stored_group_order'] for o in out),
          'verdict mismatches', sum(o['allowed'] != o['stored_allowed'] for o in out))
