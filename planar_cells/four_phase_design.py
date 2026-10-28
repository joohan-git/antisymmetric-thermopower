"""Incremental four-phase parametric ellipse refinement of the best 3-phase design."""

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
from scipy.stats import qmc
from scipy.optimize import minimize
from common.transport2d import ROOT, MATERIALS, save, solve, energy_enclosure, richardson
from certified_bounds.corner_sectors import sector_pair


def main():
    original=json.loads((ROOT/'planar_cells/material_designs.json').read_text())['best3']
    base=np.array(original['base_labels']);h=len(base)//2;quad=base[:h,:h].copy()
    x,y=np.meshgrid((np.arange(h)+.5)/h,(np.arange(h)+.5)/h,indexing='ij')
    records=[];cache={}
    def geometry(params):
        cx,cy=params[:2];a=.08+.5*params[2];b=.06+.35*params[3];t=np.pi*params[4]
        u=(x-cx)*np.cos(t)+(y-cy)*np.sin(t);v=-(x-cx)*np.sin(t)+(y-cy)*np.cos(t)
        q=quad.copy();q[(u/a)**2+(v/b)**2<1]=3
        full=np.zeros_like(base);full[:h,:h]=q+1
        return sum(np.rot90(full,j) for j in range(4))-1
    def evaluate(params,extra,k=2):
        labels=geometry(params);fr=np.bincount(labels.ravel(),minlength=4)/labels.size
        if fr.min()<.03125:return None
        key=(extra,labels.tobytes(),k)
        if key in cache:return cache[key]
        names=original['materials']+[extra];phases=[MATERIALS[m] for m in names];norm=max(abs(p[1]) for p in phases)*1e6
        g=np.kron(labels,np.ones((k,k),int));a=solve(g,phases);b=solve(g,phases,'fv')
        score=min(abs(a['Sa']),abs(b['Sa']))/norm if a['Sa']*b['Sa']>0 else 0
        r={'materials':names,'params':list(map(float,params)),'base_labels':labels.tolist(),'fractions':fr.tolist(),
           'score':score,'refine':k,'Sa_fem':a['Sa'],'Sa_fv':b['Sa'],'sigma_fem':a['sigma_iso'],'sigma_fv':b['sigma_iso']}
        records.append(r);cache[key]=r;return r
    starts=qmc.Sobol(5,scramble=True,seed=446).random_base2(8)
    for extra in ('Cu','Ni','PEDOT'):
        for params in starts:evaluate(params,extra)
        ranked=sorted((r for r in records if r['materials'][-1]==extra),key=lambda r:r['score'],reverse=True)[:2]
        for r in ranked:
            x0=np.array(r['params']);simplex=np.tile(x0,(6,1))
            for i in range(5):simplex[i+1,i]+=.1 if x0[i]<.85 else -.1
            def objective(x):
                if np.any(x<0) or np.any(x>1):return 1
                r=evaluate(x,extra);return -r['score'] if r else 1
            minimize(objective,x0,method='Nelder-Mead',options={'initial_simplex':simplex,'maxfev':160,'xatol':.001,'fatol':1e-6})
    for r in sorted(records,key=lambda r:r['score'],reverse=True)[:12]:evaluate(r['params'],r['materials'][-1],4)
    best=max((r for r in records if r['refine']==4),key=lambda r:r['score'])
    out={'algorithm':'p4 copies of a parametric ellipse replacing part of best3; Sobol plus Nelder-Mead',
         'source_design':original,'records':records,'best':best,'fem':[],'fv':[],'enclosures':[],'sector':[]}
    save('planar_cells/four_phase_design.json',out)
    base=np.array(best['base_labels']);phases=[MATERIALS[m] for m in best['materials']]
    print('best',best,flush=True)
    for k in (1,2,4,8,16):
        g=np.kron(base,np.ones((k,k),int))
        for method in ('fem','fv'):
            r=solve(g,phases,method);r['k']=k;out[method].append(r)
            print('four phase',k,method,r['Sa'],flush=True);save('planar_cells/four_phase_design.json',out)
        enc=energy_enclosure(g,phases,out['fem'][-1]);enc['k']=k;out['enclosures'].append(enc)
        save('planar_cells/four_phase_design.json',out)
    for m in (2,4,8):
        r=sector_pair(base,phases,m,m);out['sector'].append(r)
        print('four sector',m,r['fem']['Sa'],r['enclosure']['Sa_interval_uV_K'],flush=True)
        save('planar_cells/four_phase_design.json',out)
    out['richardson']={method:{key:richardson(out[method],key) for key in ('Sa','Sxx','sigma_iso')} for method in ('fem','fv')}
    save('planar_cells/four_phase_design.json',out)


if __name__=='__main__':main()
