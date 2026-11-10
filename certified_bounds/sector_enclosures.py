"""Corner-enriched continuum estimates and primal/dual bounds."""

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
from certified_bounds.corner_sectors import sector_homogenize, sector_pair
from common.transport2d import ROOT, save, original_cell3, PHASES, MATERIALS, material_energy


def main():
    destination=ROOT/'certified_bounds/sector_enclosures.json'
    out=json.loads(destination.read_text()) if destination.exists() else {'method':'Conforming mapped Q1 outside circular vertex patches; angular FEM with exact radial DtN inside patches; primal-dual energy enclosures',
         'tests':{},'cases':{}}
    for contrast in (1,10,10000):
        if 'checker_'+str(contrast) in out['tests']:continue
        rows=[]
        for m in (2,4,8):
            r=sector_homogenize(np.array([[0,1],[1,0]]),[(1.,0.,1.),(float(contrast),0.,1.)],m,m)
            r['exact_sigma']=float(np.sqrt(contrast));r['relative_error']=r['sigma_iso']/np.sqrt(contrast)-1;rows.append(r)
        out['tests']['checker_'+str(contrast)]=rows;save('certified_bounds/sector_enclosures.json',out)
    phases=[MATERIALS['p'],MATERIALS['Cu']]
    labels=np.zeros((2,2),int);labels[1,:]=1
    aa=np.array([material_energy(p) for p in phases]);normal=np.linalg.inv(np.linalg.inv(aa).mean(axis=0));tangent=aa.mean(axis=0)
    exact=np.zeros((4,4));exact[np.ix_([0,2],[0,2])]=normal;exact[np.ix_([1,3],[1,3])]=tangent
    if 'laminate' not in out['tests']:
        out['tests']['laminate']=[]
        for m in (2,4,8):
            r=sector_homogenize(labels,phases,m,m);r['relative_energy_error']=float(np.linalg.norm(np.array(r['scaled_effective_energy'])-exact)/np.linalg.norm(exact))
            out['tests']['laminate'].append(r)
    # Quadrature variation checks on a nonconstant material map.
    small=original_cell3(8,15,2)
    if 'quadrature' not in out['tests']:
        out['tests']['quadrature']=[sector_homogenize(small,[PHASES[x] for x in ('p','n','weak')],4,4,ngauss=g) for g in (6,10,14)]
    save('certified_bounds/sector_enclosures.json',out)
    designs=json.loads((ROOT/'planar_cells/material_designs.json').read_text());ideal=json.loads((ROOT/'planar_cells/ideal_materials.json').read_text())
    cases=[('baseline',original_cell3(),[PHASES[x] for x in ('p','n','weak')]),
           ('metal',original_cell3(),[MATERIALS[x] for x in ('p','n','Cu')]),
           ('best3',np.array(designs['best3']['base_labels']),[MATERIALS[x] for x in designs['best3']['materials']]),
           ('best4',np.array(designs['best4']['base_labels']),[MATERIALS[x] for x in designs['best4']['materials']]),
           ('ideal',np.array(ideal['base_labels']),ideal['best']['phases'])]
    for name,base,phases in cases:
        rows=out['cases'].setdefault(name,[])
        for m in (2,4,8):
            if any(r['fem']['m']==m for r in rows):continue
            r=sector_pair(base,phases,m,m);rows.append(r)
            p=r['fem'];e=r['enclosure']
            print(name,m,'Sa',p['Sa'],'Sxx',p['Sxx'],'sigma',p['sigma_iso'],'bounds',e['Sa_interval_uV_K'],e['Sxx_interval_uV_K'],'sec',p['seconds'],flush=True)
            save('certified_bounds/sector_enclosures.json',out)


if __name__=='__main__':main()
