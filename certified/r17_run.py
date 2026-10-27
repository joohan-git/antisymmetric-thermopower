"""Fixed-raster insulator substitution at original r13 rectangular spacings."""
import argparse
import hashlib
import json
import time
import numpy as np
from r17_core import pair
from r17_connectivity import ROOT, NAMES
from r6_core import save


def main(levels, names, kappas=(.3, 1.)):
    path = ROOT/'r13_geometries.json'; data = json.loads(path.read_text())
    output = ROOT/'r17_transport.json'
    out = json.loads(output.read_text()) if output.exists() else {
        'classification': 'C = floating-point continuum variational interval; P = uniform-grid point estimate. No V claim.',
        'source': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'reference_temperature_K': 300, 'cases': {}}
    original = {category: json.loads((ROOT/f'r13_{category}.json').read_text()) for category in ('woodpile', 'rcsr')}
    for k in levels:
        for category, name in NAMES:
            if names and name not in names: continue
            meta = data[category][name]; base = np.array(meta['labels'], dtype=np.int32)
            rec = out['cases'].setdefault(name, {'shape_base': list(base.shape), 'voxel_spacing_base': meta['voxel_spacing_base'],
                'label_sha256': hashlib.sha256(base.tobytes()).hexdigest(), 'original_weak_phases': data['materials_sigma_S_kappa'],
                'original_r13_reference': original[category]['cases'][name], 'insulator': {}})
            for kap in kappas:
                phases = [(1e-6, 0., kap), (1e5, -180e-6, 1.3), (1e5, 200e-6, 1.2)]
                row = rec['insulator'].setdefault(str(kap), {'phases_sigma_S_kappa': phases, 'levels': []})
                if any(level['k'] == k for level in row['levels']): continue
                lab = np.kron(base, np.ones((k, k, k), dtype=np.int32)); h = np.array(meta['voxel_spacing_base'])/k
                print('START', name, 'kappa', kap, 'k', k, lab.shape, flush=True)
                started = time.perf_counter(); result = pair(lab, phases, h)
                result.update(k=k, seconds_P=time.perf_counter()-started)
                row['levels'].append(result); save(output.name, out)
    # The geometry file may have been re-saved with other entries since the stored run; the voxel labels of every
    # case used here must still be those of the stored results.
    for category, name in NAMES:
        if name in out['cases']:
            base = np.array(data[category][name]['labels'], dtype=np.int32)
            assert hashlib.sha256(base.tobytes()).hexdigest() == out['cases'][name]['label_sha256'], name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--k', type=int, nargs='+', default=[1, 2]); parser.add_argument('--cases', nargs='+')
    parser.add_argument('--kappa', type=float, nargs='+', default=[.3, 1.])
    args = parser.parse_args(); main(args.k, args.cases, args.kappa)
