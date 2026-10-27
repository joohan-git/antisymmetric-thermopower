"""Periodic FV with zero-thickness limit of a neutral thermoelectric interface layer."""
import json,time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from r5_fem import material_energy,BETA,ENERGY_SCALE
from r6_core import ROOT,save
from r7 import readonly_inputs,schur
from r8_finite import POINT


def junction_gap(a,w):
    a=a.copy();N=len(a);x,y=np.meshgrid((np.arange(N)+.5)/N,(np.arange(N)+.5)/N,indexing='ij')
    for cx,cy in ((0.,.5),(.5,0.)):
        dx=abs((x-cx+.5)%1-.5);dy=abs((y-cy+.5)%1-.5)
        a[(dx<w/2-1e-12)&(dy<w/2-1e-12)]=0
    assert np.array_equal(a,np.rot90(a))
    return a


def homogenize_interface(labels,phases,rho=0.,rk=0.,L=.001):
    started=time.perf_counter();N=len(labels);n=N*N
    coeff=np.array([material_energy(p) for p in phases]);inv=np.linalg.inv(coeff)
    resistance=ENERGY_SCALE*np.diag([rho/BETA**2,300*rk])*N/L
    I,J=np.indices(labels.shape);p=(I*N+J).ravel();faces=[];row=[];col=[];entry=[]
    rhs=np.zeros((2*n,4));direct=np.zeros((4,4))
    for d,(di,dj) in enumerate(((1,0),(0,1))):
        q=(((I+di)%N)*N+(J+dj)%N).ravel();aa=labels.ravel();bb=labels[(I+di)%N,(J+dj)%N].ravel()
        conductance=np.linalg.inv(.5*(inv[aa]+inv[bb])+(aa!=bb)[:,None,None]*resistance)
        g=np.zeros((2,4));g[0,d]=1/N;g[1,2+d]=1/N
        block=np.einsum('ij,eab->eiajb',np.array([[1.,-1.],[-1.,1.]]),conductance).reshape(n,4,4)
        dofs=np.column_stack((2*p,2*p+1,2*q,2*q+1))
        row.append(np.broadcast_to(dofs[:,:,None],block.shape).ravel());col.append(np.broadcast_to(dofs[:,None,:],block.shape).ravel());entry.append(block.ravel())
        force=conductance@g
        np.add.at(rhs,2*p,force[:,0]);np.add.at(rhs,2*p+1,force[:,1]);np.add.at(rhs,2*q,-force[:,0]);np.add.at(rhs,2*q+1,-force[:,1])
        direct+=np.einsum('ia,eij,jb->ab',g,conductance,g)
        faces.append((p,q,conductance))
    K=coo_matrix((np.concatenate(entry),(np.concatenate(row),np.concatenate(col))),shape=(2*n,2*n)).tocsc();a=K[2:,2:];s=1/np.sqrt(a.diagonal())
    lu=splu((diags(s)@a@diags(s)).tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,options={'SymmetricMode':True})
    u=np.zeros_like(rhs);u[2:]=s[:,None]*lu.solve(s[:,None]*rhs[2:]);u[2:]+=s[:,None]*lu.solve(s[:,None]*(rhs[2:]-a@u[2:]))
    c=direct-rhs.T@u;fac=np.array([BETA,BETA,1,1]);physical=c*ENERGY_SCALE/fac[:,None]/fac[None,:]
    sigma=physical[:2,:2];S=np.linalg.solve(sigma,physical[:2,2:])*1e6
    return {'number_type':POINT,'N':N,'L_m':L,'rho_c_ohm_m2':rho,'R_K_m2K_W':rk,
        'Sa_uV_K':float((S[0,1]-S[1,0])/2),'Sxx_uV_K':float(S[0,0]),'S_uV_K':S.tolist(),
        'sigma_S_m':float(np.trace(sigma)/2),'kappa_open_W_mK':float(300*ENERGY_SCALE*np.trace(schur(c))/2),
        'scaled_effective_energy':c.tolist(),'residual':float(np.linalg.norm(K@u-rhs)/np.linalg.norm(rhs)),
        'seconds':time.perf_counter()-started}


def main():
    f,m=readonly_inputs();ph=[(10.45,10e-6,.9012),m['nBiTeSe'],m['pBiSbTe']]
    path=ROOT/'r8_contact.json';out=json.loads(path.read_text()) if path.exists() else {
        'number_type':POINT,'phases':ph,'model':'Every unlike-phase face: neutral interface S_i=0; delta/sigma_i=rho_c, delta/kappa_i=R_K; exact zero-thickness series-resistance limit. Baseline and changes use same FV grid.',
        'cases':{}}
    for kind in ('original','gap_4pct'):
        base=f.F3_pinwheel_tiling(100,r=.6412,split=.9)
        if kind!='original':base=junction_gap(base,.04)
        for L in (.001,.0003):
            for rk in (0.,1e-5):
                for rho in (0.,1e-9,1e-8,1e-7,1e-6):
                    key=f'{kind}_L{L}_RK{rk}_rho{rho}'
                    if key in out['cases']:continue
                    a=np.kron(base,np.ones((2,2),int));r=homogenize_interface(a,ph,rho,rk,L);r['geometry']=kind
                    out['cases'][key]=r;save(path.name,out);print(key,r['Sa_uV_K'],flush=True)
    for kind in ('original','gap_4pct'):
        base=f.F3_pinwheel_tiling(100,r=.6412,split=.9)
        if kind!='original':base=junction_gap(base,.04)
        for rho in (0.,1e-8,1e-6):
            key=f'fine_{kind}_{rho}'
            if key in out['cases']:continue
            r=homogenize_interface(np.kron(base,np.ones((4,4),int)),ph,rho,1e-5,.0003);r['geometry']=kind
            out['cases'][key]=r;save(path.name,out);print(key,r['Sa_uV_K'],flush=True)


if __name__=='__main__':main()
