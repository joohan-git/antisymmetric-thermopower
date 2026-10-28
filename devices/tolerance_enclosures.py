"""Continuum check of rounding, selected random members, and finite-specimen bulk reference."""

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
from common.transport2d import ROOT, save, inverse_phases, enclosure_from_pair
from certified_bounds.corner_sectors import sector_pair, sector_homogenize
from certified_bounds.design_enclosures import readonly_inputs, metrics
from devices.contact import junction_gap
from devices.tolerance import CERT


def general_sa_interval(enc):
    # C blocks ordered Vx,Vy,Tx,Ty, with V scaled by BETA.
    low=np.array(enc['lower_energy']);high=np.array(enc['upper_energy']);mid=(low+high)/2;gap=high-low
    radius=.5*np.sqrt(np.maximum(np.outer(np.diag(gap),np.diag(gap)),0))
    def entry(i,j):return np.array([mid[i,j]-radius[i,j],mid[i,j]+radius[i,j]])
    def prod(x,y):z=np.outer(x,y);return np.array([z.min(),z.max()])
    def diff(x,y):return np.array([x[0]-y[1],x[1]-y[0]])
    a,b,d=entry(0,0),entry(0,1),entry(1,1);e,f,g,h=entry(0,2),entry(0,3),entry(1,2),entry(1,3)
    b2=np.array([0 if b[0]*b[1]<=0 else min(b*b),max(b*b)])
    determinant=diff(prod(a,d),b2);assert determinant[0]>0
    numerator=diff(prod(d,f),prod(a,g))+prod(b,diff(e,h))
    return (100*prod(numerator,1/determinant[::-1])).tolist()


def main():
    data=json.loads((ROOT/'devices/tolerance.json').read_text());ph=data['phases'];path=ROOT/'devices/tolerance_enclosures.json'
    out=json.loads(path.read_text()) if path.exists() else {'number_type':CERT,'cases':{}}
    configurations=[]
    for key,r in data['rounding'].items():
        if float(key)>0:configurations.append(('rounding_'+key,np.array(r['labels']),True,1))
    p4=sorted([(k,r) for k,r in data['ensemble'].items() if r['symmetry_preserved']],key=lambda x:x[1]['Sa'])
    for key,r in (p4[0],p4[-1]):configurations.append((key,np.array(r['labels']),True,1))
    key,r=min([(k,r) for k,r in data['ensemble'].items() if not r['symmetry_preserved']],key=lambda x:x[1]['Sa'])
    configurations.append((key,np.array(r['labels']),False,1))
    f,_=readonly_inputs();configurations.append(('finite_gap4pct_N50',junction_gap(f.F3_pinwheel_tiling(50,r=.6412,split=.9),.04),True,2))
    for key,a,p4,m in configurations:
        if key in out['cases']:continue
        if p4:
            r=sector_pair(a,ph,m,m);r['metrics']=metrics(r,ph);interval=r['metrics']['Sa_uV_K']
        else:
            primal=sector_homogenize(a,ph,m,m);dual=sector_homogenize(a,inverse_phases(ph),m,m);enc=enclosure_from_pair(primal,dual)
            interval=general_sa_interval(enc);r={'fem':primal,'general_Sa_interval_uV_K':interval,'lower_energy':enc['lower_energy'],'upper_energy':enc['upper_energy'],
                'formula':'General anisotropic 2x2 inverse: Sa/BETA=(d*f-a*g+b*(e-h))/(2*(a*d-b*b)); conservative interval arithmetic. p4 ratio formula not used.'}
        r['number_type']=CERT;r['N_base']=len(a);r['p4']=p4;out['cases'][key]=r;save(path.relative_to(ROOT).as_posix(),out);print(key,interval,flush=True)


if __name__=='__main__':main()
