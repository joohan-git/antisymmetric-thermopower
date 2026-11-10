"""Fixed-raster insulator substitution at original junction rectangular spacings."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import argparse
import hashlib
import json
import time
import numpy as np
from common.insulating_voxels import pair
from three_d.periodic_connectivity import ROOT, NAMES
from common.transport2d import save


def main(levels, names, kappas=(.3, 1.)):
    path = ROOT/'three_d/junction_geometry.json'; data = json.loads(path.read_text())
    output = ROOT/'three_d/insulating_transport.json'
    out = json.loads(output.read_text()) if output.exists() else {
        'classification': 'C = floating-point continuum variational interval; P = uniform-grid point estimate. No V claim.',
        'source': path.relative_to(RELEASE_ROOT).as_posix(), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'reference_temperature_K': 300, 'cases': {}}
    original = {category: json.loads((ROOT/f'three_d/junction_{category}.json').read_text()) for category in ('woodpile', 'rcsr')}
    for k in levels:
        for category, name in NAMES:
            if names and name not in names: continue
            meta = data[category][name]; base = np.array(meta['labels'], dtype=np.int32)
            rec = out['cases'].setdefault(name, {'shape_base': list(base.shape), 'voxel_spacing_base': meta['voxel_spacing_base'],
                'label_sha256': hashlib.sha256(base.tobytes()).hexdigest(), 'original_weak_phases': data['materials_sigma_S_kappa'],
                'junction_transport_reference': original[category]['cases'][name], 'insulator': {}})
            for kap in kappas:
                phases = [(1e-6, 0., kap), (1e5, -180e-6, 1.3), (1e5, 200e-6, 1.2)]
                row = rec['insulator'].setdefault(str(kap), {'phases_sigma_S_kappa': phases, 'levels': []})
                if any(level['k'] == k for level in row['levels']): continue
                lab = np.kron(base, np.ones((k, k, k), dtype=np.int32)); h = np.array(meta['voxel_spacing_base'])/k
                print('START', name, 'kappa', kap, 'k', k, lab.shape, flush=True)
                started = time.perf_counter(); result = pair(lab, phases, h)
                result.update(k=k, seconds_P=time.perf_counter()-started)
                row['levels'].append(result); save(output.relative_to(ROOT).as_posix(), out)
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
