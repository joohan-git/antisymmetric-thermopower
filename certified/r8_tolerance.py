"""Explicit fabrication perturbation models, with continuum brackets for junction gaps."""
import json
import numpy as np
from scipy.ndimage import binary_opening,distance_transform_edt,gaussian_filter
from r6_core import ROOT,save,solve
from r6_sector import sector_pair
from r7 import readonly_inputs,metrics,map_diagnostics
from r8_contact import junction_gap
from r8_finite import POINT

CERT='continuum_variational_enclosure'


def rounding(base,r):
    if r==0:return base.copy()
    N=len(base);radius=r*N;k=int(np.ceil(radius));x,y=np.indices((2*k+1,2*k+1))-k;disk=x*x+y*y<=radius**2+1e-12
    out=np.zeros_like(base)
    for p in (1,2):
        opened=binary_opening(np.tile(base==p,(3,3)),structure=disk)
        out[opened[N:2*N,N:2*N]]=p
    return out


def perturb(base,rng,symmetric):
    N=len(base);scores=[];fields=[]
    for p in (0,1,2):
        mask=np.tile(base==p,(3,3));sd=(distance_transform_edt(mask)-distance_transform_edt(~mask))[N:2*N,N:2*N]/N
        z=gaussian_filter(rng.normal(size=base.shape),.08*N,mode='wrap');z-=z.mean()
        if symmetric:z=sum(np.rot90(z,k) for k in range(4))/4
        z*=.02/max(abs(z).max(),1e-20);scores.append(sd+z);fields.append(z)
    return np.argmax(scores,axis=0).astype(int)


def main():
    f,m=readonly_inputs();ph=[(10.45,10e-6,.9012),m['nBiTeSe'],m['pBiSbTe']]
    path=ROOT/'r8_tolerance.json';out=json.loads(path.read_text()) if path.exists() else {'phases':ph,'gaps':{},'rounding':{},'ensemble':{},
        'definitions':{'gap':'At periodic crossing vertices (0,0.5),(0.5,0): replace centered square of side w by CPC phase0. N200 resolves every requested width exactly. Fixed raster outside patches.',
        'rounding':'Periodic disk morphological opening of each conductive phase1,2; removed convex corners filled by phase0; stated radius measured in unit-cell units.',
        'ensemble':'Each phase periodic signed distance perturbed by smooth zero-mean Gaussian field, max amplitude0.02 cell, correlation length0.08 cell; argmax determines labels. p4 cases rotate-average each field; random seed20261024. Does not guarantee topology preservation.',
        'certificate':'Conforming sector primal and dual variational enclosures for the stated fixed pixel geometry; quadrature and floating-point rounding not interval certified.'}}
    base=f.F3_pinwheel_tiling(200,r=.6412,split=.9)
    for w in (0.,.01,.02,.04):
        key=str(w)
        if key in out['gaps']:continue
        a=junction_gap(base,w);r=sector_pair(a,ph,1,1);r['metrics']=metrics(r,ph);r['number_type']=CERT;r['w_fraction']=w;r['N_base']=200;r['labels']=a.tolist();r['diagnostics']=map_diagnostics(a)
        out['gaps'][key]=r;save(path.name,out);print('gap',w,r['metrics']['Sa_uV_K'],flush=True)
    base=f.F3_pinwheel_tiling(100,r=.6412,split=.9)
    for radius in (0.,.01,.02,.03,.04,.05):
        key=str(radius)
        if key in out['rounding']:continue
        a=rounding(base,radius);rows=[]
        for k in (1,2):
            grid=np.kron(a,np.ones((k,k),int));rows.append({'k':k,'number_type':POINT,'fv':solve(grid,ph,'fv'),'fem':solve(grid,ph)})
        out['rounding'][key]={'radius_fraction':radius,'N_base':100,'number_type':POINT,'rows':rows,'diagnostics':map_diagnostics(a),'labels':a.tolist()};save(path.name,out);print('round',radius,rows[-1]['fv']['Sa'],rows[-1]['fem']['Sa'],flush=True)
    rng=np.random.default_rng(20261024)
    for symmetric in (True,False):
        for index in range(24):
            a=perturb(base,rng,symmetric);key=f'p4_{symmetric}_{index}'
            if key in out['ensemble']:continue
            result=solve(a,ph,'fv');result['number_type']=POINT;result['symmetry_preserved']=symmetric;result['sample']=index;result['diagnostics']=map_diagnostics(a)
            # Keep the geometry so extreme and representative members can be checked independently.
            result['labels']=a.tolist();out['ensemble'][key]=result;save(path.name,out);print('ensemble',key,result['Sa'],flush=True)
    # A few independently resolved members expose FV corner biases, rather than treating sampling spread as error bars.
    for symmetric in (True,False):
        rows=[(k,r) for k,r in out['ensemble'].items() if r['symmetry_preserved']==symmetric]
        order=sorted(rows,key=lambda x:x[1]['Sa'])
        for key,r in (order[0],order[len(order)//2],order[-1]):
            if 'fem_k2' in r:continue
            r['fem_k2']=solve(np.kron(np.array(r['labels']),np.ones((2,2),int)),ph)
            r['fv_k2']=solve(np.kron(np.array(r['labels']),np.ones((2,2),int)),ph,'fv');save(path.name,out)
    out['refinement_limit']='N200 m2 nr2 factorization exceeded available memory; reported N200 m1 nr1 enclosures already establish Sa>40 for each tested gap. No extrapolated interval is used.'
    save(path.name,out)


if __name__=='__main__':main()
