"""Reproduce sensitivity, rounded inclusions, and finite-ring controls."""
import json
from pathlib import Path
import numpy as np
from r5_fem import homogenize, pinwheel, PHASES, material_energy
from r5_ring import annulus


def save(name, results):
    Path(__file__).with_name(name).write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')


def main():
    results = {}
    for conductivity in (1000, 100, 1, .01):
        results[str(conductivity)] = homogenize(pinwheel(96, 15, 3),
            [PHASES['p'], PHASES['n'], (conductivity, 0., .2)])
    save('r5_third_phase_sensitivity.json', results)
    results = {}
    for size in (48, 96, 192):
        x, y = np.meshgrid((np.arange(size)+.5)/size-.5,
                           (np.arange(size)+.5)/size-.5, indexing='ij')
        labels = np.zeros((size, size), dtype=int)
        for phase, radius, offset, angle in ((1, .26, 0, 35), (2, .32, 45, 93)):
            for k in range(4):
                phi, theta = np.deg2rad([offset+90*k, angle+90*k])
                dx, dy = x-radius*np.cos(phi), y-radius*np.sin(phi)
                u = dx*np.cos(theta)+dy*np.sin(theta)
                v = -dx*np.sin(theta)+dy*np.cos(theta)
                inside = (u/.09)**2+(v/.04)**2 < 1
                assert np.all(labels[inside] == 0)
                labels[inside] = phase
        assert np.array_equal(labels, np.rot90(labels))
        results[str(size)] = homogenize(labels, [PHASES['weak'], PHASES['p'], PHASES['n']])
    save('r5_smooth_three_phase.json', results)
    results = {}
    for label, nph, names, angle in (('two_phase', 2, ['p', 'metal'], 15),
                                    ('mirror_symmetric', 3, ['p', 'n', 'weak'], 45)):
        base = pinwheel(24, angle, nph)
        mats = np.array([np.kron(material_energy(PHASES[p]), np.eye(2)) for p in names])
        for tiles in (8, 16):
            labels = np.tile(base, (tiles, tiles))
            results[label+'_'+str(tiles)] = annulus(len(labels), mats, labels)
    save('r5_ring_controls.json', results)
    mats = np.array([np.kron(material_energy(PHASES[p]), np.eye(2)) for p in ('p', 'n')])
    base = pinwheel(24, 15, 3)
    base[base == 2] = 0
    results = {'periodic': homogenize(base, [PHASES['p'], PHASES['n']])}
    for tiles in (8, 16, 24):
        labels = np.tile(base, (tiles, tiles))
        results[str(tiles)] = annulus(len(labels), mats, labels)
    save('r5_ring_two_phase_merged.json', results)
    mats = np.array([np.kron(material_energy(PHASES[p]), np.eye(2)) for p in ('p', 'metal')])
    labels = np.tile(pinwheel(48, 15, 2), (8, 8))
    save('r5_ring_boundary_refinement.json', annulus(len(labels), mats, labels))


if __name__ == '__main__':
    main()
