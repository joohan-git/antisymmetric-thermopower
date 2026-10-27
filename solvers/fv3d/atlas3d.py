"""Aggregation of the three-dimensional colour-group atlas: straight woodpiles, segmented woodpiles and RCSR-derived nets.
Reads scan_woodpile_N24.json with woodpile_groups.json, scan_segwood_N16_s*.json and rcsr_color_N16_all.json (stored results, no RCSR file needed)
and writes atlas3d.json. 'violations' counts structures whose computed |S_a| contradicts the colour-group verdict."""
import json, glob, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
atlas = []
for f in sorted(glob.glob(os.path.join(HERE, 'scan_segwood_N16_s*.json'))):
    for r in json.load(open(f)): atlas.append(dict(family='segmented woodpile', allowed=bool(r['allowed'][2]), az=abs(r['ax'][2]), inplane=float(np.hypot(*r['ax'][:2])), allowed_inplane=bool(r['allowed'][0] or r['allowed'][1])))
groups = json.load(open(os.path.join(HERE, 'woodpile_groups.json')))
for r, g in zip(json.load(open(os.path.join(HERE, 'scan_woodpile_N24.json'))), groups):
    atlas.append(dict(family='straight woodpile', allowed=bool(g['allowed'][2]), az=abs(r['az']), inplane=r['ax_inplane'], allowed_inplane=bool(g['allowed'][0] or g['allowed'][1]), colour_point_group=g['colour_point_group']))
for r in json.load(open(os.path.join(HERE, 'rcsr_color_N16_all.json'))):
    atlas.append(dict(family='RCSR net', allowed=bool(r['allowed'][2]), az=abs(r['ax'][2]), inplane=float(np.hypot(*r['ax'][:2])), allowed_inplane=bool(r['allowed'][0] or r['allowed'][1]), name=r['name']))
viol = [a for a in atlas if (not a['allowed'] and a['az'] > 1e-6) or (a['allowed'] and a['az'] < 1e-6) or (not a['allowed_inplane'] and a['inplane'] > 1e-6)]
print(len(atlas), 'structures;', sum(a['allowed'] for a in atlas), 'allowed;', len(viol), 'violations; max forbidden |S_a|', max(a['az'] for a in atlas if not a['allowed']))
json.dump(dict(atlas=atlas, n=len(atlas), violations=len(viol)), open(os.path.join(HERE, 'atlas3d.json'), 'w'), indent=1)
