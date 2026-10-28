"""Certification of the design geometries in planar_cells with the planar conforming sector/dual method."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
from pathlib import Path
import ast
import hashlib
import json
import types
import time
import numpy as np
from common.transport2d import ROOT, save, solve, rel
from certified_bounds.corner_sectors import sector_pair

T0=300.
SOURCE_PATHS=[ROOT/'planar_cells/families.py',ROOT/'common/mats.py']


def hashes():
    return {rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS}


def readonly_inputs():
    module=types.ModuleType('design_families'); module.__file__=str(SOURCE_PATHS[0])
    exec(compile(SOURCE_PATHS[0].read_text(encoding='utf-8'),str(SOURCE_PATHS[0]),'exec'),module.__dict__)
    # Read the literal material dictionary without triggering path-dependent imports or bytecode writes.
    tree=ast.parse(SOURCE_PATHS[1].read_text(encoding='utf-8'))
    mats=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAT' for t in n.targets))
    return module,mats


def schur(c):
    c=np.array(c)
    return c[2:,2:]-c[2:,:2]@np.linalg.solve(c[:2,:2],c[:2,2:])


def metrics(result,phases):
    """Schur complements are order preserving: L <= C <= U implies Schur(L) <= Schur(C) <= Schur(U)."""
    enc=result['enclosure'];p=result['fem']
    kap=[float(T0*.004*np.trace(schur(enc[k]))/2) for k in ('lower_energy','upper_energy')]
    assert 0<kap[0]<=kap[1]
    sigma=enc['sigma_interval'];sa=enc['Sa_interval_uV_K']
    ab=sorted(abs(x) for x in sa) if sa[0]*sa[1]>0 else [0.,max(abs(x) for x in sa)]
    pf=[ab[0]**2*1e-12*sigma[0],ab[1]**2*1e-12*sigma[1]]
    zt=[pf[0]*T0/kap[1],pf[1]*T0/kap[0]]
    halfspan=(max(x[1] for x in phases)-min(x[1] for x in phases))*5e5
    kp=float(T0*.004*np.trace(schur(p['scaled_effective_energy']))/2)
    pp=p['Sa']**2*1e-12*p['sigma_iso']
    return {'Sa_uV_K':sa,'Sxx_uV_K':enc['Sxx_interval_uV_K'],'sigma_S_m':sigma,
            'kappa_open_W_mK':kap,'PF_a_microW_mK2':[x*1e6 for x in pf],'zT_a':zt,
            'S_halfspan_uV_K':halfspan,'span_normalized_abs_Sa':[x/halfspan for x in ab],
            'point':{'Sa_uV_K':p['Sa'],'Sxx_uV_K':p['Sxx'],'sigma_S_m':p['sigma_iso'],
                     'kappa_open_W_mK':kp,'PF_a_microW_mK2':pp*1e6,'zT_a':pp*T0/kp,
                     'span_normalized_abs_Sa':abs(p['Sa'])/halfspan},
            'interval_rule':'Schur-monotone kappa bounds; conservative separate-interval products for PF and zT'}


def map_diagnostics(a):
    patterns={}
    for i in range(len(a)):
        for j in range(len(a)):
            q=tuple(int(x) for x in (a[i,j],a[i-1,j],a[i-1,j-1],a[i,j-1]))
            patterns[q]=patterns.get(q,0)+1
    return {'p4':bool(np.array_equal(a,np.rot90(a))),
            'fractions':(np.bincount(a.ravel())/a.size).tolist(),
            'alternating_two_phase_vertices':sum(v for q,v in patterns.items() if q[0]==q[2] and q[1]==q[3] and q[0]!=q[1]),
            'three_phase_vertices':sum(v for q,v in patterns.items() if len(set(q))==3),
            'corners':[{'phases':list(k),'count':v} for k,v in sorted(patterns.items())]}


def main():
    families,mats=readonly_inputs();path=ROOT/'certified_bounds/design_enclosures.json'
    out=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'source_hashes_before':hashes(),'T0_K':T0,'cases':{}}
    inputs={'A':{'phases':[(10.45,10e-6,.9012),mats['nBiTeSe'],mats['pBiSbTe']],
                 'function':'F3_pinwheel_tiling','parameters':{'r':.6412,'split':.9}},
            'B':{'phases':[mats['pBiSbTe'],mats['nBiTeSe'],mats['CPC']],
                 'function':'F2_rotsquares','parameters':{'theta':12,'s':.8}}}
    for name,meta in inputs.items():
        for n in (32,64):
            key=f'{name}_{n}';base=getattr(families,meta['function'])(n,**meta['parameters'])
            case=out['cases'].setdefault(key,dict(meta,N_base=n,labels=base.tolist(),diagnostics=map_diagnostics(base),sector=[],uniform=[]))
            assert case['diagnostics']['p4']
            for k in (1,2,4):
                if any(r['k']==k for r in case['uniform']):continue
                g=np.kron(base,np.ones((k,k),int))
                row={'k':k,'fem':solve(g,meta['phases']),'fv':solve(g,meta['phases'],'fv')}
                case['uniform'].append(row)
                print(key,'uniform',k,row['fem']['Sa'],row['fv']['Sa'],flush=True)
                save(path.relative_to(ROOT).as_posix(),out)
            for m in (2,4):
                if any(r['fem']['m']==m for r in case['sector']):continue
                r=sector_pair(base,meta['phases'],m,m);r['metrics']=metrics(r,meta['phases']);case['sector'].append(r)
                print(key,'sector',m,r['metrics'],flush=True);save(path.relative_to(ROOT).as_posix(),out)
            if name=='B':
                mirror=families.F2_rotsquares(n,theta=-12,s=.8)
                case['theta_mirror_label_identity']=bool(np.array_equal(mirror,base[::-1,:]))
                case['mirror_fv']=solve(np.kron(mirror,np.ones((2,2),int)),meta['phases'],'fv')
            out['source_hashes_after']=hashes();save(path.relative_to(ROOT).as_posix(),out)
    old=json.loads((ROOT/'certified_bounds/sector_enclosures.json').read_text(encoding='utf-8'))
    design=json.loads((ROOT/'planar_cells/material_designs.json').read_text(encoding='utf-8'))
    four=json.loads((ROOT/'planar_cells/four_phase_design.json').read_text(encoding='utf-8'))
    from common.transport2d import MATERIALS
    out['material_design_comparison']={
        'three':{'materials':design['best3']['materials'],'metrics':metrics(old['cases']['best3'][-1],[MATERIALS[x] for x in design['best3']['materials']])},
        'four':{'materials':four['best']['materials'],'metrics':metrics(four['sector'][-1],[MATERIALS[x] for x in four['best']['materials']])}}
    out['source_hashes_after']=hashes();out['sources_unchanged']=out['source_hashes_before']==out['source_hashes_after'];save(path.relative_to(ROOT).as_posix(),out)


if __name__=='__main__':main()
