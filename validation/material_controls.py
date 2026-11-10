"""Mirror controls and an ideal material contrast sweep."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import numpy as np
from common.transport2d import ROOT, save, solve, MATERIALS
from certified_bounds.corner_sectors import sector_pair


def main():
    four=json.loads((ROOT/'planar_cells/four_phase_design.json').read_text())
    ideal=json.loads((ROOT/'planar_cells/ideal_materials.json').read_text())
    out={}
    for name,d in [('best4_overlay',four),('ideal',ideal)]:
        base=np.array(d['best']['base_labels'] if name=='best4_overlay' else d['base_labels'])
        phases=[MATERIALS[m] for m in d['best']['materials']] if name=='best4_overlay' else d['best']['phases']
        g=np.kron(base,np.ones((4,4),int))
        out[name]={'original':{m:solve(g,phases,m) for m in ('fem','fv')},
                   'mirror':{m:solve(g[::-1,:].copy(),phases,m) for m in ('fem','fv')}}
    save('validation/material_controls.json',out)
    base=np.array(ideal['base_labels']);a=np.array(ideal['best']['phases'])
    mean=np.exp(np.log(a[:,[0,2]]).mean(axis=0));rows=[]
    for t in (0,.25,.5,.75,1):
        phases=a.copy();phases[:,[0,2]]=np.exp((1-t)*np.log(mean)+t*np.log(a[:,[0,2]]))
        r=sector_pair(base,phases.tolist(),2,2)
        rows.append({'contrast_fraction':t,'phases':phases.tolist(),'result':r})
    save('validation/ideal_material_trend.json',rows)


if __name__=='__main__':main()
