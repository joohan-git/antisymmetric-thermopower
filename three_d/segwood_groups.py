"""Colour-group symbols of the segmented woodpiles of scan_segwood_N16_s*.json.

Each voxel map is rebuilt from its stored parameters with woodpile.seg_woodpile (16 voxels, 4 per layer); the colour group is
the set of point operations that, with some voxel translation, leave every phase in place (rcsr_colour_pg.colour_group_full).
Output: three_d/segwood_groups.json (per structure: file, index, colour point group, its order, axial-vector verdict, stored order)."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import glob
import json
import os
import sys
from collections import Counter
HERE = str(RELEASE_ROOT)
from three_d import woodpile as W
from three_d import coloring as C
from three_d.rcsr_colour_pg import colour_group_full, symbol

if __name__ == '__main__':
    N = 16
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, 'three_d/scan_segwood_N16_s*.json'))):
        for i, r in enumerate(json.load(open(f))):
            T = W.seg_woodpile(N, N//4, r['c1'] % 1, r['c2'] % 1, r['w1'], r['w2'], r['a'], r['b'], tuple(r['o1']), tuple(r['o2']), 0, r['stack'])
            G = colour_group_full(T)
            out.append(dict(file=os.path.basename(f), index=i, colour_point_group=symbol(G), colour_group_order=len(G),
                            allowed=[bool(a) for a in C.axial_allowed(G)[0]], stored_group_order=r['group_order'], stored_allowed=r['allowed']))
    json.dump(out, open(os.path.join(HERE, 'three_d/segwood_groups.json'), 'w'), indent=1)
    print(len(out), 'structures', dict(Counter(o['colour_point_group'] for o in out)),
          'order mismatches', sum(o['colour_group_order'] != o['stored_group_order'] for o in out),
          'verdict mismatches', sum(o['allowed'] != o['stored_allowed'] for o in out))
