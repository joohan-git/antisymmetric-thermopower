"""Material-preserving tetragonal point groups of fixed voxel geometries."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import collections
import hashlib
import json
import numpy as np
from three_d.fixed_rasters import ROOT, MAIN, W, exact_group, save


def classify(g):
    ops = [np.array(x) for x in g['operations']]
    n = len(ops)
    proper = all(round(np.linalg.det(x)) == 1 for x in ops)
    inv = any(np.array_equal(x, -np.eye(3)) for x in ops)
    if n == 16:
        name = '4/mmm'
    elif n == 8 and proper:
        name = '422'
    elif n == 8 and inv and g['has_C4_z']:
        name = '4/m'
    elif n == 8 and inv:
        name = 'mmm'
    elif n == 8 and g['has_C4_z']:
        name = '4mm'
    elif n == 8:
        name = '-42m'
    elif n == 4 and proper:
        name = '4' if g['has_C4_z'] else '222'
    elif n == 4 and inv:
        name = '2/m'
    elif n == 4:
        name = '-4' if any(round(np.trace(x)) == -1 and round(np.linalg.det(x)) == -1 and not np.array_equal(x @ x, np.eye(3)) for x in ops) else 'mm2'
    elif n == 2:
        name = '2' if proper else ('-1' if inv else 'm')
    elif n == 1:
        name = '1'
    else:
        raise ValueError(g)
    keys = {tuple(x.ravel()) for x in ops}
    assert all(tuple((a @ b).ravel()) in keys for a in ops for b in ops)
    g['point_group'] = name
    g['perpendicular_twofolds'] = [x.tolist() for x in ops if round(np.linalg.det(x)) == 1 and round(np.trace(x)) == -1 and x[2, 2] == -1]
    g['classification'] = 'C: exact equality of fixed voxel labels; point-group quotient by translations, restricted to tetragonal 4/mmm parent'
    return g


def main():
    result = {'definition': 'No phase permutations. Every voxel translation, not only multiples of N/8. Group order counts distinct point operations.', 'cases': []}
    files = [MAIN/'three_d/scan_woodpile_N24.json'] + sorted(MAIN.glob('three_d/scan_segwood_N16_s*.json'))
    sources = files + [MAIN/'three_d/woodpile.py', MAIN/'three_d/coloring.py']
    hashes = lambda: {(p.resolve().relative_to(RELEASE_ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result['source_hashes_before'] = hashes()
    for path in files:
        straight = path.name.startswith('scan_woodpile_')
        n = 24 if straight else 16
        for index, row in enumerate(json.loads(path.read_text())):
            keys = ('c1', 'c2', 'w1', 'w2') if straight else ('c1', 'c2', 'w1', 'w2', 'a', 'b')
            params = [row[k] for k in keys]
            labels = W.woodpile(n, n//4, *params) if straight else W.seg_woodpile(n, n//4, *params, row['o1'], row['o2'], 0, row['stack'])
            group = classify(exact_group(labels))
            result['cases'].append({'id': path.stem + ':' + str(index), 'index_zero_based': index, 'family': 'straight' if straight else 'segmented',
                'source_file': path.relative_to(RELEASE_ROOT).as_posix(), 'N': n, 'original_record': row, 'labels_sha256': hashlib.sha256(labels.astype(np.int8).tobytes()).hexdigest(),
                'group': group, 'FV_Sa_z_uV_K_P': row['az'] if straight else row['ax'][2]})
    result['summary'] = {}
    result['forbidden_segmented_rerasterization'] = []
    for case in result['cases']:
        if case['family'] != 'segmented' or case['group']['axial_allowed'][2]: continue
        row = case['original_record']
        for n in (32, 64):
            params = [row[k] for k in ('c1', 'c2', 'w1', 'w2', 'a', 'b')]
            labels = W.seg_woodpile(n, n//4, *params, row['o1'], row['o2'], 0, row['stack'])
            result['forbidden_segmented_rerasterization'].append({'id': case['id'], 'N': n,
                'group': classify(exact_group(labels)), 'note': 'New raster, not fixed-geometry subdivision.'})
    for family in ('straight', 'segmented'):
        rows = [x for x in result['cases'] if x['family'] == family]
        result['summary'][family] = {'count_C': len(rows), 'groups_C': dict(collections.Counter(x['group']['point_group'] for x in rows)),
            'z_allowed_C': sum(x['group']['axial_allowed'][2] for x in rows),
            'exceptions_to_4_or_2': [x['id'] for x in rows if x['group']['point_group'] not in ('4', '2')],
            'max_forbidden_abs_Sa_P': max((abs(x['FV_Sa_z_uV_K_P']) for x in rows if not x['group']['axial_allowed'][2]), default=0)}
    result['source_hashes_after'] = hashes()
    assert result['source_hashes_before'] == result['source_hashes_after']
    save('three_d/material_symmetry.json', result)
    print({family: {k: v for k, v in item.items() if k != 'exceptions_to_4_or_2'} for family, item in result['summary'].items()}, flush=True)
    for x in result['cases']:
        if x['family'] == 'segmented' and x['group']['point_group'] == '422':
            print(x['id'], x['group'], flush=True)


if __name__ == '__main__':
    main()
