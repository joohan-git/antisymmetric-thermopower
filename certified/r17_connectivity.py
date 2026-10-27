"""Exact periodic face-graph components and noncontractible winding cycles."""
import json
from collections import deque
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = [('woodpile', 'W2_split_clear'), ('woodpile', 'W1_split'),
         ('rcsr', 'cyc'), ('rcsr', 'unm'), ('rcsr', 'bbr')]


def topology(labels, selected=(1, 2)):
    shape = np.array(labels.shape, dtype=int)
    mask = np.isin(labels, selected)
    component = np.full(labels.shape, -1, dtype=np.int32)
    lifts = np.zeros(tuple(labels.shape)+(3,), dtype=np.int64)
    records = []
    steps = [(axis, sign) for axis in range(3) for sign in (-1, 1)]
    for seed_array in np.argwhere(mask):
        seed = tuple(seed_array)
        if component[seed] >= 0: continue
        index = len(records); component[seed] = index
        lifts[seed] = seed_array
        queue = deque([seed]); count = 0; phases = [0, 0, 0]; cycles = set()
        while queue:
            p = queue.popleft(); count += 1; phases[int(labels[p])] += 1
            for axis, sign in steps:
                q = list(p); q[axis] = (q[axis]+sign) % shape[axis]; q = tuple(q)
                if not mask[q]: continue
                target = lifts[p].copy(); target[axis] += sign
                if component[q] < 0:
                    component[q] = index; lifts[q] = target; queue.append(q)
                else:
                    assert component[q] == index
                    delta = target-lifts[q]
                    assert np.all(delta % shape == 0)
                    winding = delta//shape
                    if np.any(winding):
                        first = winding[np.flatnonzero(winding)[0]]
                        if first < 0: winding = -winding
                        cycles.add(tuple(int(x) for x in winding))
        basis = []
        for cycle in sorted(cycles):
            v = np.array(cycle, dtype=np.int64)
            independent = not basis or (len(basis) == 1 and np.any(np.cross(basis[0], v))) or (len(basis) == 2 and int(np.dot(np.cross(*basis), v)) != 0)
            if independent: basis.append(v)
            if len(basis) == 3: break
        axes = [any(v[d] != 0 for v in cycles) for d in range(3)]
        records.append({'id': index, 'voxel_count': count, 'label_counts': phases,
                        'winding_generators': [list(x) for x in sorted(cycles)],
                        'winding_rank': len(basis), 'percolates_xyz': axes})
    return {'classification': 'Exact combinatorial result, not a transport point estimate',
            'definition': 'Only shared voxel faces are edges; periodic lifted cycles detect winding, not mere boundary crossing.',
            'selected_labels': list(selected), 'component_count': len(records),
            'percolates_xyz': [any(r['percolates_xyz'][d] for r in records) for d in range(3)],
            'components': records}


def main():
    source = ROOT/'r13_geometries.json'; data = json.loads(source.read_text())
    cases = {}
    for category, name in NAMES:
        meta = data[category][name]; labels = np.array(meta['labels'], dtype=np.int32)
        both = topology(labels)
        cases[name] = {'source_category': category, 'shape': list(labels.shape),
                       'voxel_spacing_base': meta['voxel_spacing_base'],
                       'label_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
                       'pn_union': both, 'n_only': topology(labels, (1,)), 'p_only': topology(labels, (2,))}
        print(name, 'components', both['component_count'], 'percolation', both['percolates_xyz'], flush=True)
    out = {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cases': cases}
    (ROOT/'r17_connectivity.json').write_text(json.dumps(out, indent=2)+chr(10), encoding='utf-8')


if __name__ == '__main__': main()
