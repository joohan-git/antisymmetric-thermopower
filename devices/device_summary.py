"""Derived device quantities and validation of energy order."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json; import hashlib
import numpy as np
from common.transport2d import ROOT, save
from common.finite_sample import POINT
from devices.tolerance import CERT

# Original stored inventory sizes; hashes identify the files in this release.
ARCHIVED_INPUT_BYTES = {
    'devices/winding.json': 28267,
    'devices/tolerance.json': 10452643,
    'devices/tolerance_enclosures.json': 45951,
    'devices/manufacturing.json': 6942602,
    'devices/contact.json': 61736,
    'devices/electrodes.json': 152916,
    'devices/ring.json': 23322,
}


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def summary_stats(rows):
    values=np.array([r['Sa'] for r in rows]);return {'number_type':POINT,'count':len(values),'mean_Sa_uV_K':float(values.mean()),
        'sample_std_Sa_uV_K':float(values.std(ddof=1)),'min_Sa_uV_K':float(values.min()),'max_Sa_uV_K':float(values.max()),
        'quantiles_05_50_95_uV_K':np.quantile(values,[.05,.5,.95]).tolist(),'count_below_40':int(np.sum(values<40))}


def main():
    t=read('devices/tolerance.json');m=read('devices/manufacturing.json');e=read('devices/electrodes.json')['cases'];c=read('devices/contact.json')['cases'];a=read('devices/tolerance_enclosures.json')['cases']
    out={'number_type_legend':{CERT:'continuum certified interval: stated fixed pixel map; variational discretization enclosure, not interval-arithmetic floating-point proof',POINT:'uniform-grid point estimate: not a continuum enclosure'},
        'ensemble':{},'contact_PF':{},'electrode_ratios':{},'data_inventory':{}}
    for symmetric in (True,False):
        out['ensemble'][f'unprotected_p4_{symmetric}']=summary_stats([r for r in t['ensemble'].values() if r['symmetry_preserved']==symmetric])
        out['ensemble'][f'protected_p4_{symmetric}']=summary_stats([r for r in m['protected_ensemble'].values() if r['symmetry_preserved']==symmetric])
    for key,r in c.items():
        out['contact_PF'][key]={'number_type':POINT,'PF_microW_mK2':r['Sa_uV_K']**2*1e-6*r['sigma_S_m']}
    for key,r in e.items():
        reference=a['finite_gap4pct_N50']['fem']['Sa'] if 'gap4pct' in key else 49.8304472
        if r['mirror']:reference=-reference
        out['electrode_ratios'][key]={'number_type':POINT,'fixed_reference_Sa_uV_K':reference,
            'central_ratio_to_reference':r['aligned_probes'][2]['V_left_minus_right_uV']/(reference*r['cells'][0]/r['cells'][1])}
    # Validate energy order for every interval used in the report.
    gaps=[]
    def walk(obj):
        if isinstance(obj,dict):
            if 'lower_energy' in obj and 'upper_energy' in obj:
                lo=np.array(obj['lower_energy']);hi=np.array(obj['upper_energy']);gaps.append(float(np.linalg.eigvalsh(hi-lo).min()));assert np.linalg.eigvalsh(lo).min()>0
            for value in obj.values():walk(value)
        elif isinstance(obj,list):
            for value in obj:walk(value)
    for file in ('devices/tolerance.json','devices/tolerance_enclosures.json','devices/manufacturing.json'):walk(read(file))
    assert min(gaps)>-1e-8
    out['variational_order_check']={'checked_matrices':len(gaps),'minimum_gap_eigenvalue':min(gaps)}
    for name in ('devices/winding.json','devices/tolerance.json','devices/tolerance_enclosures.json','devices/manufacturing.json','devices/contact.json','devices/electrodes.json','devices/ring.json'):
        p=ROOT/name;read(name);out['data_inventory'][name]={'archived_input_bytes':ARCHIVED_INPUT_BYTES[name],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    save('devices/device_summary.json',out)
    print(json.dumps({'ensemble':out['ensemble'],'order':out['variational_order_check']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
