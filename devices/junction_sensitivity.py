"""Small p4 junction regularizations, solely to assess experiment sensitivity."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import numpy as np
from certified_bounds.design_enclosures import readonly_inputs, map_diagnostics
from common.transport2d import save, solve


def main():
    family,mats=readonly_inputs();n=64;base=family.F3_pinwheel_tiling(n,r=.6412,split=.9)
    phases=[(10.45,10e-6,.9012),mats['nBiTeSe'],mats['pBiSbTe']]
    sites=[]
    for i in range(n):
        for j in range(n):
            a,b,c,d=base[i,j],base[i-1,j],base[i-1,j-1],base[i,j-1]
            if a==c and b==d and a!=b:sites.append((i,j))
    out={'N':n,'sites':sites,'patch_definition':'Set four pixels adjacent to each alternating vertex to phase 0 or 1; both changes preserve p4','cases':{}}
    for label in (0,1):
        changed=base.copy()
        for i,j in sites:
            for di,dj in ((0,0),(-1,0),(-1,-1),(0,-1)):
                changed[(i+di)%n,(j+dj)%n]=label
        assert np.array_equal(changed,np.rot90(changed))
        case={'labels':changed.tolist(),'changed_pixels':int(np.count_nonzero(changed!=base)),
              'changed_fraction':float(np.mean(changed!=base)),'diagnostics':map_diagnostics(changed),'rows':[]}
        out['cases'][str(label)]=case
        for k in (2,4):
            g=np.kron(changed,np.ones((k,k),int));row={'k':k}
            for method in ('fem','fv'):
                r=solve(g,phases,method);row[method]=r
                print('patch',label,'k',k,method,r['Sa'],r['sigma_iso'],flush=True)
            case['rows'].append(row);save('devices/junction_sensitivity.json',out)


if __name__=='__main__':main()
