"""Aggregation of the three-dimensional colour-group atlas: straight woodpiles, segmented woodpiles and RCSR-derived nets.
Reads three_d/scan_woodpile_N24.json with three_d/woodpile_groups.json, scan_segwood_N16_s*.json and three_d/rcsr_color_N16_all.json (stored results, no RCSR file needed)
and writes atlas3d.json. 'violations' counts structures whose computed |S_a| contradicts the colour-group verdict."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json; import glob; import os; import numpy as np
HERE = str(RELEASE_ROOT)
atlas = []
for f in sorted(glob.glob(os.path.join(HERE, 'three_d/scan_segwood_N16_s*.json'))):
    for r in json.load(open(f)): atlas.append(dict(family='segmented woodpile', allowed=bool(r['allowed'][2]), az=abs(r['ax'][2]), inplane=float(np.hypot(*r['ax'][:2])), allowed_inplane=bool(r['allowed'][0] or r['allowed'][1])))
groups = json.load(open(os.path.join(HERE, 'three_d/woodpile_groups.json')))
for r, g in zip(json.load(open(os.path.join(HERE, 'three_d/scan_woodpile_N24.json'))), groups):
    atlas.append(dict(family='straight woodpile', allowed=bool(g['allowed'][2]), az=abs(r['az']), inplane=r['ax_inplane'], allowed_inplane=bool(g['allowed'][0] or g['allowed'][1]), colour_point_group=g['colour_point_group']))
for r in json.load(open(os.path.join(HERE, 'three_d/rcsr_color_N16_all.json'))):
    atlas.append(dict(family='RCSR net', allowed=bool(r['allowed'][2]), az=abs(r['ax'][2]), inplane=float(np.hypot(*r['ax'][:2])), allowed_inplane=bool(r['allowed'][0] or r['allowed'][1]), name=r['name']))
viol = [a for a in atlas if (not a['allowed'] and a['az'] > 1e-6) or (a['allowed'] and a['az'] < 1e-6) or (not a['allowed_inplane'] and a['inplane'] > 1e-6)]
print(len(atlas), 'structures;', sum(a['allowed'] for a in atlas), 'allowed;', len(viol), 'violations; max forbidden |S_a|', max(a['az'] for a in atlas if not a['allowed']))
json.dump(dict(atlas=atlas, n=len(atlas), violations=len(viol)), open(os.path.join(HERE, 'three_d/atlas3d.json'), 'w'), indent=1)
