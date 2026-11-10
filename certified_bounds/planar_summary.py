"""Collect the planar results (validation, sector enclosures, designs, four-phase overlay, idealized materials) into planar_summary.json."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
from pathlib import Path
import numpy as np
from common.transport2d import ROOT, save, sources, MATERIALS, richardson


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def interval(x,digits=3):
    return f'[{x[0]:.{digits}f}, {x[1]:.{digits}f}]'


def build():
    validation=read('validation/planar_refinement.json');sector=read('certified_bounds/sector_enclosures.json')
    search=read('planar_cells/material_search.json');designs=read('planar_cells/material_designs.json');four=read('planar_cells/four_phase_design.json');ideal=read('planar_cells/ideal_materials.json')
    analysis=read('certified_bounds/corner_exponents.json');perturb=read('validation/material_perturbations.json');controls=read('validation/material_controls.json')
    assert all(len(sector['cases'].get(k,[]))==3 for k in ('baseline','metal','best3','best4','ideal'))
    assert len(four['sector'])==3
    final={k:v[-1] for k,v in sector['cases'].items()};final['best4_overlay']=four['sector'][-1]
    normalizers={'baseline':200.,'metal':200.,'best3':200.,'best4':200.,'best4_overlay':200.,'ideal':ideal['best']['Smax_uV_K']}
    summary={'T0_K':300,'definition':'Sa=(Sxy-Syx)/2; array axis0=x, axis1=y',
             'source_hashes_before':validation['source_sha256_before'],'source_hashes_after':sources(),
             'realistic_materials':MATERIALS,'cases':{}}
    for name,r in final.items():
        rows=four['sector'] if name=='best4_overlay' else sector['cases'][name]
        p=r['fem'];enc=r['enclosure'];norm=normalizers[name]
        raw=enc['Sa_interval_uV_K'];abs_bounds=sorted(abs(x) for x in raw) if raw[0]*raw[1]>0 else [0.,max(abs(x) for x in raw)]
        summary['cases'][name]={'Sa_fine_uV_K':p['Sa'],'Sxx_fine_uV_K':p['Sxx'],'sigma_fine':p['sigma_iso'],
            'Sa_interval_uV_K':raw,'Sxx_interval_uV_K':enc['Sxx_interval_uV_K'],'sigma_interval':enc['sigma_interval'],
            'normalized_abs_Sa_interval':(np.array(abs_bounds)/norm).tolist(),'normalizer_uV_K':norm,
            'p2_extrapolation':{key:float((4*rows[-1]['fem'][key]-rows[-2]['fem'][key])/3) for key in ('Sa','Sxx','sigma_iso')},
            'observed_richardson':{key:richardson([dict(x['fem'],k=x['fem']['m']) for x in rows],key) for key in ('Sa','Sxx','sigma_iso')}}
    summary['best3_design']=designs['best3'];summary['best4_design']=four['best'];summary['ideal_materials']=ideal['best']['phases']
    summary['search_evaluations']={'initial_shapes':len(search['records']),'four_phase_overlay':len(four['records']),'ideal_materials':len(ideal['records'])}
    summary['source_hashes_unchanged']=summary['source_hashes_before']==summary['source_hashes_after']
    save('certified_bounds/planar_summary.json',summary)


if __name__=='__main__':build()
