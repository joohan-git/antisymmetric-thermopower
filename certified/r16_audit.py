"""Reconstruct each V certificate from saved integer trial arrays, without PDE solves."""
import json
import time
import hashlib
from fractions import Fraction
import numpy as np
import r16_verify as v
from flint import arb, arb_mat


def main():
    started = time.perf_counter(); records = []
    # A zero-containing ball must be squared by multiplication for this Arb API.
    test = arb_mat([[arb(2), arb('0 +/- 1e-30')], [arb('0 +/- 1e-30'), arb(3)]])
    assert v.spd(test)[0]
    for path in sorted(v.ROOT.glob('r16_verified_*.json')):
        start = time.perf_counter(); data = json.loads(path.read_text())
        archive = v.ROOT/data['trial_archive']
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == data['trial_sha256']
        trial = np.load(archive); labels = trial['labels']; dim = labels.ndim
        if dim == 2:
            assert np.array_equal(labels, np.rot90(labels))
        else:
            # Recheck the stored screw operation by integer label equality;
            # no FFT or tolerance participates in this proof replay.
            rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            matches = []
            symmetry = data['symmetry_exact']
            for operation, translation in zip(symmetry['operations'], symmetry['translations_voxels']):
                if np.array_equal(operation, rotation):
                    matches.append(np.array_equal(np.roll(np.rot90(labels, axes=(0, 1)), tuple(translation), axis=(0, 1, 2)), labels))
            assert matches and all(matches)
        materials = v.decimal_materials(data['phases_sigma_S_kappa_decimal_inputs'])
        upper, grams, den = v.q1_energy(labels, trial['Q1_primal_nodal_integer'], materials)
        assert str(den) == data['primal_denominator']
        assert v.integer_dump(grams) == data['primal_integer_grams']
        if dim == 2:
            dual, grams, den = v.q1_energy(labels, trial['Q1_dual_nodal_integer'], [a.inv() for a in materials])
            r = arb_mat(np.kron(np.eye(2, dtype=int), np.array([[0, -1], [1, 0]])).tolist())
            lower = r.transpose()*dual.inv()*r
            exact_divergence = 'identically zero by rotated gradient of Q1'
        else:
            assert np.count_nonzero(v.integer_divergence(trial['RT0_face_integer'])) == 0
            lower, unused, mean, grams, den = v.rt0_energy(labels, trial['RT0_face_integer'], materials)
            assert not mean.det().contains(0)
            exact_divergence = 0
        assert str(den) == data['dual_denominator']
        assert v.integer_dump(grams) == data['dual_integer_grams']
        upper = v.p4_symmetry(upper, dim); lower = v.p4_symmetry(lower, dim)
        assert v.spd(lower)[0] and v.spd(upper-lower)[0]
        n = dim*2; ha = arb_mat(n, n); ha[0, 0] = ha[1, 1] = v.ball(Fraction(1, 2))
        hc = arb_mat(n, n); hc[0, dim+1] = hc[dim+1, 0] = v.ball(Fraction(1, 4)); hc[1, dim] = hc[dim, 1] = v.ball(Fraction(-1, 4))
        proof = []
        for witness in data['ratio_dual_witnesses_V']:
            t = Fraction(witness['t_exact']); h = hc*witness['sign']-ha*v.ball(t)
            # Choose exact dyadic midpoints from the recorded balls, then
            # reconstruct Q=P-H exactly instead of trusting an approximate identity.
            p = arb_mat([[arb(x).mid() for x in row] for row in witness['P_ball']])
            p = (p+p.transpose())/2; q = p-h
            assert v.spd(p)[0] and v.spd(q)[0]
            value = v.trace_product(p, upper)-v.trace_product(q, lower)
            assert value <= 0
            proof.append(value.str(60, more=True))
        exact = [Fraction(x) for x in data['Sa_interval_uV_K_V_exact_rational']]
        witnesses = data['ratio_dual_witnesses_V']
        assert [x['sign'] for x in witnesses] == [1, -1]
        assert exact == [-Fraction(witnesses[1]['t_exact'])*200, Fraction(witnesses[0]['t_exact'])*200]
        printed = [Fraction(x) for x in data['Sa_interval_uV_K_V_decimal']]
        assert printed[0] <= exact[0] <= exact[1] <= printed[1]
        records.append({'file': path.name, 'classification': 'V', 'all_exact_integer_moments_match': True,
            'divergence_check': exact_divergence, 'rechecked_ratio_functionals_ball': proof,
            'exact_fourfold_label_symmetry_rechecked': True,
            'reported_endpoints_round_outward': True, 'audit_seconds_P': time.perf_counter()-start})
        print('rechecked', path.name, flush=True)
    v.save('r16_audit.json', {'records': records, 'all_passed': True, 'seconds_P': time.perf_counter()-started})


if __name__ == '__main__': main()
