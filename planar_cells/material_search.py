"""Independent shape/material search: Sobol sampling plus multistart Nelder-Mead.

No pixel label genetic operations or simulated annealing. Exact p4 is imposed by
rotating one quadrant. Candidates remain discrete pure phases at every solve.
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
from itertools import combinations
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from common.transport2d import MATERIALS, save, solve


def shape(params,nph,n=16,family='stripes'):
    p=np.clip(params,0,1)
    h=n//2
    x,y=np.meshgrid((np.arange(h)+.5)/h,(np.arange(h)+.5)/h,indexing='ij')
    theta=np.pi*p[0]
    u=x*np.cos(theta)+y*np.sin(theta)
    v=-x*np.sin(theta)+y*np.cos(theta)
    if family=='stripes':
        period=.25+1.5*p[1]
        coord=(u+1.2*(p[3]-.5)*(x-.5)*(y-.5)+.6*(p[4]-.5)*((x-.5)**2-(y-.5)**2))/period+p[2]
        weights=np.exp(3*(p[5:5+nph]-.5));weights=.08+(1-.08*nph)*weights/weights.sum()
        quad=np.searchsorted(np.cumsum(weights),coord%1).clip(0,nph-1)
    else:
        # Anisotropic periodic Voronoi sites in the fundamental quadrant.
        scores=[]
        ratio=10**(2*(p[0]-.5))
        theta=np.pi*p[1];co=np.cos(theta);si=np.sin(theta)
        for i in range(nph):
            dx=x-p[2+2*i];dy=y-p[3+2*i]
            du=co*dx+si*dy;dv=-si*dx+co*dy
            scores.append(ratio*du**2+dv**2/ratio)
        quad=np.argmin(scores,axis=0)
    full=np.zeros((n,n),int);full[:h,:h]=quad+1
    labels=sum(np.rot90(full,j) for j in range(4))-1
    assert np.array_equal(labels,np.rot90(labels))
    return labels


def run_realistic():
    started=time.perf_counter();names=list(MATERIALS)[:-1]
    records=[];candidate_pool=[];combination_summaries=[]
    sequence=qmc.Sobol(10,scramble=True,seed=261024).random_base2(7)
    cache={}
    def evaluate(params,materials,family,refine=1,stage='sobol'):
        labels=shape(params,len(materials),family=family)
        fractions=np.bincount(labels.ravel(),minlength=len(materials))/labels.size
        if fractions.min()<.03125:
            return None
        key=(tuple(materials),labels.tobytes(),refine)
        if key in cache:
            return cache[key]
        phases=[MATERIALS[m] for m in materials]
        refined=np.kron(labels,np.ones((refine,refine),int))
        a=solve(refined,phases,'fem');b=solve(refined,phases,'fv')
        norm=max(abs(p[1]) for p in phases)*1e6
        # Reject opposite signs as an unresolved discretization signal.
        robust=min(abs(a['Sa']),abs(b['Sa']))/norm if a['Sa']*b['Sa']>0 else 0.
        item={'materials':list(materials),'family':family,'params':list(map(float,params)),
              'fractions':fractions.tolist(),'refine':refine,'stage':stage,'score':robust,
              'Sa_fem':a['Sa'],'Sa_fv':b['Sa'],'ratio_fem':abs(a['Sa'])/norm,'ratio_fv':abs(b['Sa'])/norm,
              'sigma_fem':a['sigma_iso'],'sigma_fv':b['sigma_iso'],'Sxx_fem':a['Sxx'],'Sxx_fv':b['Sxx']}
        records.append(item);cache[key]=item
        return item
    def checkpoint():
        save('planar_cells/material_search.json',{'algorithm':'Sobol shapes plus multistart Nelder-Mead; score=min absolute FEM and FV Sa if signs agree',
             'base_N':16,'minimum_fraction':.03125,'seed':261024,'materials':MATERIALS,
             'elapsed_seconds':time.perf_counter()-started,'records':records,
             'combination_summaries':combination_summaries,'candidates':candidate_pool})
    for nph in (3,4):
        for materials in combinations(names,nph):
            local=[]
            for i,params in enumerate(sequence):
                family='stripes' if i%2==0 else 'voronoi'
                r=evaluate(params,materials,family)
                if r:local.append(r)
            top=sorted(local,key=lambda r:r['score'],reverse=True)[:5]
            checked=[]
            for r in top:
                rr=evaluate(r['params'],materials,r['family'],2,'screen_refine2')
                if rr:checked.append(rr)
            if not checked:continue
            best=max(checked,key=lambda r:r['score'])
            combination_summaries.append(best)
            candidate_pool.extend(checked[:2])
            print('combination',materials,'best',best['score'],'Sa',best['Sa_fem'],best['Sa_fv'],flush=True)
            checkpoint()
    # Independent continuous shape parameters, optimized with a deliberately broad initial simplex.
    for nph in (3,4):
        winners=sorted((r for r in combination_summaries if len(r['materials'])==nph),key=lambda r:r['score'],reverse=True)[:5]
        for winner in winners:
            family=winner['family'];materials=winner['materials']
            dim=5+nph if family=='stripes' else 2+2*nph
            x0=np.array(winner['params'][:dim]);simplex=np.tile(x0,(dim+1,1))
            for i in range(dim):simplex[i+1,i]+= .12 if x0[i]<.8 else -.12
            def objective(x):
                if np.any(x<0) or np.any(x>1):return 1+float(np.maximum(-x,0).sum()+np.maximum(x-1,0).sum())
                padded=np.pad(x,(0,10-len(x)),constant_values=.5)
                r=evaluate(padded,materials,family,2,'nelder_mead')
                return -r['score'] if r else 1.
            opt=minimize(objective,x0,method='Nelder-Mead',options={'initial_simplex':simplex,'maxfev':280,'xatol':.0005,'fatol':1e-6})
            best=evaluate(np.pad(opt.x,(0,10-len(opt.x)),constant_values=.5),materials,family,2,'nelder_mead_final')
            if best:candidate_pool.append(best)
            print('optimized',materials,family,best['score'] if best else None,'eval',opt.nfev,flush=True);checkpoint()
    # Re-rank using the same fixed geometries at N=64. Keep multiple material choices.
    finalists=[]
    for nph in (3,4):
        ranked=sorted((r for r in records if r['refine']==2 and len(r['materials'])==nph),key=lambda r:r['score'],reverse=True)
        selected=[];seen=set();per_material={}
        for r in ranked:
            labels=shape(r['params'],nph,family=r['family'])
            key=(tuple(r['materials']),labels.tobytes())
            if key in seen or per_material.get(tuple(r['materials']),0)>=3:continue
            seen.add(key);per_material[tuple(r['materials'])]=per_material.get(tuple(r['materials']),0)+1
            selected.append(r)
            if len(selected)==12:break
        for r in selected:
            rr=evaluate(r['params'],r['materials'],r['family'],4,'finalist_refine4')
            if rr:finalists.append(rr)
    finalists.sort(key=lambda r:r['score'],reverse=True)
    for r in finalists:r['base_labels']=shape(r['params'],len(r['materials']),family=r['family']).tolist()
    candidate_pool[:]=finalists;checkpoint()
    save('planar_cells/material_designs.json',{'algorithm':'Sobol + Nelder-Mead parametric shape optimization','finalists':finalists,
         'best3':next(r for r in finalists if len(r['materials'])==3),
         'best4':next(r for r in finalists if len(r['materials'])==4)})


if __name__=='__main__':
    run_realistic()
