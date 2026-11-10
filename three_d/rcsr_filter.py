"""Selection of RCSR nets for the colour-group atlas, step by step.
Needs the RCSR net file (see README: three_d/rcsr3d_web.cgd or environment variable RCSR_CGD).
Filters: tetragonal; at most 8 vertices in the conventional cell (rcsr_net.expand, origin choice 2 where it exists);
0.5 <= c/a <= 1.6; the struts can be built from the EDGE_CENTER records (rcsr_net.edges). Output three_d/rcsr_filter.json with the
count and names after each filter and, for the excluded records, the reason."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import os; import sys; import json
import gemmi
HERE = str(RELEASE_ROOT)
from three_d import rcsr_net as RN

if __name__ == '__main__':
    nets = RN.read_cgd()
    steps = [('RCSR three-dimensional nets', [n['name'] for n in nets])]
    tet = [n for n in nets if (sg := gemmi.find_spacegroup_by_name(n['group'])) is not None and sg.crystal_system_str() == 'tetragonal']
    steps.append(('tetragonal', [n['name'] for n in tet]))
    small = [n for n in tet if len(RN.expand(n)[0]) <= 8]
    steps.append(('at most 8 vertices per cell', [n['name'] for n in small]))
    ca = [n for n in small if 0.5 <= n['cell'][2] / n['cell'][0] <= 1.6]
    steps.append(('0.5 <= c/a <= 1.6', [n['name'] for n in ca]))
    built, excluded = [], []
    for n in ca:
        try:
            pts, cns = RN.expand(n); RN.edges(n, pts, cns); built.append(n['name'])
        except ValueError as e:
            excluded.append({'name': n['name'], 'group': n['group'], 'reason': str(e)[:300]})
    steps.append(('struts determined by the EDGE_CENTER records', built))
    out = {'steps': [{'label': l, 'count': len(v), 'names': v if len(v) <= 300 else None} for l, v in steps], 'excluded': excluded}
    json.dump(out, open(os.path.join(HERE, 'three_d/rcsr_filter.json'), 'w'), indent=1)
    print([(s['label'], s['count']) for s in out['steps']], 'excluded', [e['name'] for e in excluded])
