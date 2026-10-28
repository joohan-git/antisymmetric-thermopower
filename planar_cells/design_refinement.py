"""Validate optimized fixed geometries, perturbation checks, and ideal material search."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from common.transport2d import ROOT, MATERIALS, PHASES, save, solve, energy_enclosure, richardson, material_energy, original_cell3, BETA
from planar_cells.material_search import shape


def certify_designs():
    designs=json.loads((ROOT/'planar_cells/material_designs.json').read_text())
    out={}
    for name in ('best3','best4'):
        design=designs[name];base=np.array(design['base_labels']);phases=[MATERIALS[m] for m in design['materials']]
        norm=max(abs(p[1]) for p in phases)*1e6
        case={'design':design,'phases':phases,'normalizer_uV_K':norm,'fem':[],'fv':[],'enclosures':[]};out[name]=case
        for k in (1,2,4,8,16):
            labels=np.kron(base,np.ones((k,k),int))
            for method in ('fem','fv'):
                r=solve(labels,phases,method);r['k']=k;r['ratio']=abs(r['Sa'])/norm;case[method].append(r)
                print(name,k,method,r['Sa'],r['Sxx'],r['sigma_iso'],flush=True)
                save('validation/material_designs.json',out)
            enc=energy_enclosure(labels,phases,case['fem'][-1]);enc['k']=k;case['enclosures'].append(enc)
            print('design certificate',name,k,enc['Sa_interval_uV_K'],flush=True)
            save('validation/material_designs.json',out)
        case['richardson']={m:{key:richardson(case[m],key) for key in ('Sa','Sxx','sigma_iso')} for m in ('fem','fv')}
        g=np.kron(base,np.ones((4,4),int))
        case['mirror']={m:solve(g[::-1,:].copy(),phases,m) for m in ('fem','fv')}
        save('validation/material_designs.json',out)


def from_energy(a):
    sigma=a[0,0]*.004/BETA**2
    seebeck=a[0,1]/a[0,0]*BETA
    kappa=300*.004*(a[1,1]-a[0,1]**2/a[0,0])
    return float(sigma),float(seebeck),float(kappa)


def perturbation():
    base=np.kron(original_cell3(),np.ones((2,2),int))
    phases=[PHASES[k] for k in ('p','n','weak')]
    matrices=np.array([material_energy(p) for p in phases]);mean=matrices.mean(axis=0)
    out={'contrast':[],'coupling':[],'gauge_shift':[]}
    for eps in (.03125,.0625,.125,.25,.5,1):
        mixed=[from_energy(mean+eps*(a-mean)) for a in matrices]
        r=solve(base,mixed);out['contrast'].append({'epsilon':eps,'Sa':r['Sa'],'Sa_over_eps3':r['Sa']/eps**3})
    for eps in (.0625,.125,.25,.5,1):
        r=solve(base,[(p[0],eps*p[1],p[2]) for p in phases])
        out['coupling'].append({'epsilon':eps,'Sa':r['Sa'],'Sa_over_eps':r['Sa']/eps})
    for shift in (-100e-6,0.,100e-6):
        r=solve(base,[(p[0],p[1]+shift,p[2]) for p in phases])
        out['gauge_shift'].append({'shift_uV':shift*1e6,'Sa':r['Sa'],'Sxx':r['Sxx']})
    # Three distinct material triples on a two-dimensional linear matrix span.
    dependent=[phases[0],phases[1],from_energy(.35*matrices[0]+.65*matrices[1])]
    out['three_phases_simultaneously_diagonalizable']={'phases':dependent,'result':solve(base,dependent)}
    save('validation/material_perturbations.json',out)


def ideal_search():
    design=json.loads((ROOT/'planar_cells/material_designs.json').read_text())['best3']
    base=np.array(design['base_labels']);labels=np.kron(base,np.ones((2,2),int))
    records=[];cache={};started=time.perf_counter()
    def decode(x):
        a=np.array(x).reshape(3,3)
        return [(10**(2+6*v[0]),(2*v[1]-1)*200e-6,10**(-1.3+4*v[2])) for v in a]
    def evaluate(x,stage):
        key=tuple(np.round(x,10))
        if key in cache:return cache[key]
        phases=decode(x);norm=max(abs(p[1]) for p in phases)*1e6
        a=solve(labels,phases);b=solve(labels,phases,'fv')
        score=min(abs(a['Sa']),abs(b['Sa']))/norm if a['Sa']*b['Sa']>0 else 0.
        r={'x':list(map(float,x)),'phases':phases,'score':score,'Sa_fem':a['Sa'],'Sa_fv':b['Sa'],
           'sigma_fem':a['sigma_iso'],'sigma_fv':b['sigma_iso'],'Smax_uV_K':norm,'stage':stage}
        records.append(r);cache[key]=r
        return r
    for i,x in enumerate(qmc.Sobol(9,scramble=True,seed=627).random_base2(10)):
        evaluate(x,'sobol')
        if i%128==127:
            print('ideal',i+1,max(r['score'] for r in records),flush=True)
    for start in sorted(records,key=lambda r:r['score'],reverse=True)[:4]:
        x0=np.array(start['x']);simplex=np.tile(x0,(10,1))
        for i in range(9):simplex[i+1,i]+=.08 if x0[i]<.85 else -.08
        def objective(x):
            if np.any(x<0) or np.any(x>1):return 1+np.maximum(-x,0).sum()+np.maximum(x-1,0).sum()
            return -evaluate(x,'nelder_mead')['score']
        opt=minimize(objective,x0,method='Nelder-Mead',options={'maxfev':240,'initial_simplex':simplex,'xatol':.001,'fatol':1e-5})
        print('ideal optimized',-opt.fun,flush=True)
    best=max(records,key=lambda r:r['score']);out={'base_labels':base.tolist(),'shape_source':'realistic best3 fixed geometry',
        'ranges':{'sigma':[1e2,1e8],'S_uV_K':[-200,200],'kappa':[10**-1.3,10**2.7]},
        'seed':627,'elapsed_seconds':time.perf_counter()-started,'records':records,'best':best,'fem':[],'fv':[],'enclosures':[]}
    save('planar_cells/ideal_materials.json',out)
    for k in (1,2,4,8,16):
        g=np.kron(base,np.ones((k,k),int))
        for method in ('fem','fv'):
            r=solve(g,best['phases'],method);r['k']=k;out[method].append(r)
            print('ideal validation',k,method,r['Sa'],r['sigma_iso'],flush=True)
            save('planar_cells/ideal_materials.json',out)
        enc=energy_enclosure(g,best['phases'],out['fem'][-1]);enc['k']=k;out['enclosures'].append(enc)
        save('planar_cells/ideal_materials.json',out)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='ideal':ideal_search()
    else:perturbation();certify_designs()
