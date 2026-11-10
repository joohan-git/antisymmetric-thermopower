"""Recompute the auxiliary ukg enclosures from its hash-verified RCSR geometry.

The stored JSON supplies the material inputs and physical voxel spacings.
Use --recompute to replace selected levels; other levels remain available.
"""
from pathlib import Path
import sys
RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

import argparse
import hashlib
import json
import time
import numpy as np
from common import rectangular_voxels, insulating_voxels
from common.transport2d import save
from three_d import fixed_rasters, rcsr_color


def geometry():
    """Return the radius-0.08a, 16-voxel net with its archived exact label hash."""
    net = next(n for n in rcsr_color.read_cgd() if n['name'] == 'ukg')
    points, coordination = rcsr_color.expand(net)
    segments = rcsr_color.edges(net, points, coordination)
    labels = rcsr_color.voxelize(net, segments, 16, .08).astype(np.int32)
    expected = 'ddcb2d410f8f97083434018d6ffc32f19820bf4c898718cc194bfc43db720203'
    assert hashlib.sha256(labels.tobytes()).hexdigest() == expected
    return labels


def main(levels, recompute):
    filename = 'three_d/ukg_check.json'
    data = json.loads((RELEASE_ROOT/filename).read_text(encoding='utf-8'))
    base = geometry()
    reference = data['insulating_transport']['ukg']
    assert list(base.shape) == reference['shape_base']
    assert hashlib.sha256(base.tobytes()).hexdigest() == reference['label_sha256']
    phases = reference['original_weak_phases']
    spacing = np.asarray(reference['voxel_spacing_base'])
    print('ukg geometry: exact archived label hash reproduced', flush=True)

    def record(rows, k, calculate):
        old = next((i for i, row in enumerate(rows) if row['k'] == k), None)
        if old is not None and not recompute:
            return
        result = calculate(); result['k'] = k
        if old is None:
            rows.append(result)
        else:
            rows[old] = result
        rows.sort(key=lambda row: row['k'])
        save(filename, data)

    for k in levels:
        labels = np.kron(base, np.ones((k, k, k), dtype=np.int32))
        for name, raster in [('ukg', labels), ('ukg_mirror_x', labels[::-1, :, :].copy())]:
            def fixed():
                fem = fixed_rasters.R.FE.homogenize(raster, phases)
                fv = fixed_rasters.R.fv_with_complement(raster, phases)
                return {'fem': fem, 'fv': fv,
                        'continuum_enclosure': fixed_rasters.R.enclosure(fv['continuum_energy_lower'], fem['scaled_energy'])}
            record(data['fixed_raster_periodic'][name]['levels'], k, fixed)
        # The physical-spacing and insulating series were archived through k=2.
        if k > 2:
            continue
        record(data['junction_rcsr']['ukg']['levels'], k,
               lambda: rectangular_voxels.pair(labels, phases, spacing/k))
        for case in reference['insulator'].values():
            def insulator():
                started = time.perf_counter()
                result = insulating_voxels.pair(labels, case['phases_sigma_S_kappa'], spacing/k)
                result['seconds_P'] = time.perf_counter()-started
                return result
            record(case['levels'], k, insulator)
        reference['junction_transport_reference'] = data['junction_rcsr']['ukg']
        save(filename, data)
        print('ukg completed k =', k, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, nargs='+', choices=(1, 2, 3), default=[1])
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    with fixed_rasters.R.threadpool_limits(limits=2):
        main(args.levels, args.recompute)
