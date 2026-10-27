"""Rectangular-voxel Q1/FV/RT0 thermoelectric homogenization for r13.

Q1 assembly comes from our independent r11 solver. FV uses the main matrix
harmonic mean but explicit face area, cell distance, and physical gradients.
RT0 uses physical flux densities, not integrated face currents.
"""
import r12_run as R
import time
import numpy as np
from scipy.sparse import coo_matrix

FE=R.R.FE


def fem(labels,phases,h=(1.,1.,1.),tol=2e-11):
    start=time.perf_counter();h=np.asarray(h,float)
    mat,dofs,coeff,ig,vol,sizes=FE.assemble(labels,phases,h=h)
    force=-np.einsum('id,padbe->piabe',ig,coeff.reshape(-1,2,3,2,3)).reshape(-1,16,6)
    rhs=np.zeros((mat.shape[0],6))
    np.add.at(rhs,dofs.ravel(),force[labels.ravel()].reshape(-1,6))
    u=np.zeros_like(rhs)
    u[2:],stats=FE.solve_spd(mat[2:,2:],rhs[2:],np.arange(2,mat.shape[0])%2,tol)
    volume=labels.size*vol
    c0=np.mean(coeff[labels.ravel()],axis=0)
    flux=c0-rhs.T@u/volume
    upper=c0-(rhs.T@u+u.T@rhs-u.T@(mat@u))/volume
    out=FE.properties((upper+upper.T)/2)
    out.update(classification='P: uniform rectangular-grid FEM; matrix is an admissible primal energy upper',
               shape=list(labels.shape),voxel_spacing=h.tolist(),solver=stats,
               flux_energy_relative=float(np.linalg.norm(flux-upper)/np.linalg.norm(upper)),
               reciprocity_relative=float(np.linalg.norm(flux-flux.T)/np.linalg.norm(flux)),
               seconds=time.perf_counter()-start)
    return out


def fv(labels,phases,h=(1.,1.,1.),tol=2e-11):
    start=time.perf_counter();h=np.asarray(h,float);vol=float(np.prod(h))
    a=np.array([FE.energy(p) for p in phases]);n=labels.size
    hm=np.array([[R.R.H3.hmean(x,y) for y in a] for x in a])
    idx=np.arange(n,dtype=np.int32).reshape(labels.shape)
    rows=[];cols=[];entries=[];faces=[]
    rhs=np.zeros((2*n,6))
    for d in range(3):
        p=idx.ravel();q=np.roll(idx,-1,axis=d).ravel()
        face=hm[labels.ravel(),np.roll(labels,-1,axis=d).ravel()]
        conductance=face*vol/h[d]**2
        faces.append((p,q,face))
        for x in range(2):
            for y in range(2):
                v=conductance[:,x,y]
                rows.extend([2*p+x,2*p+x,2*q+x,2*q+x])
                cols.extend([2*p+y,2*q+y,2*q+y,2*p+y])
                entries.extend([v,-v,v,-v])
        for fin in range(2):
            j=3*fin+d
            forcing=conductance[:,:,fin]*h[d]
            np.add.at(rhs[:,j],(2*p[:,None]+np.arange(2)).ravel(),forcing.ravel())
            np.add.at(rhs[:,j],(2*q[:,None]+np.arange(2)).ravel(),-forcing.ravel())
    mat=coo_matrix((np.concatenate(entries),(np.concatenate(rows),np.concatenate(cols))),shape=(2*n,2*n)).tocsr()
    del rows,cols,entries
    u=np.zeros_like(rhs)
    u[2:],stats=FE.solve_spd(mat[2:,2:],rhs[2:],np.arange(2,2*n)%2,tol)
    u=u.reshape(n,2,6)
    flux=np.empty((n,2,3,6))
    for d,(p,q,face) in enumerate(faces):
        g=(u[q]-u[p])/h[d]
        g[:,0,d]+=1.;g[:,1,3+d]+=1.
        flux[:,:,d,:]=np.einsum('nab,nbj->naj',face,g)
    c=flux.mean(axis=0).reshape(6,6)
    result=FE.properties((c+c.T)/2)
    result.update(classification='P: rectangular-grid FV with physical face conductances',
                  shape=list(labels.shape),voxel_spacing=h.tolist(),solver=stats,
                  reciprocity_relative=float(np.linalg.norm(c-c.T)/np.linalg.norm(c)))
    ff=flux.reshape(tuple(labels.shape)+(2,3,6))
    div=sum((ff[:,:,:,:,d,:]-np.roll(ff[:,:,:,:,d,:],1,axis=d))/h[d] for d in range(3))
    before=float(np.linalg.norm(div)*min(h)/max(np.linalg.norm(ff),1e-30))
    lam=np.zeros(labels.shape)
    for d,size in enumerate(labels.shape):
        shape=[1,1,1];shape[d]=size
        lam+=((2-2*np.cos(2*np.pi*np.arange(size)/size))/h[d]**2).reshape(shape)
    safe=lam.copy();safe[0,0,0]=1.
    phat=-np.fft.fftn(div,axes=(0,1,2))/safe[:,:,:,None,None];phat[0,0,0]=0
    psi=np.fft.ifftn(phat,axes=(0,1,2)).real
    for d in range(3):ff[:,:,:,:,d,:]-=(np.roll(psi,-1,axis=d)-psi)/h[d]
    div2=sum((ff[:,:,:,:,d,:]-np.roll(ff[:,:,:,:,d,:],1,axis=d))/h[d] for d in range(3))
    after=float(np.linalg.norm(div2)*min(h)/max(np.linalg.norm(ff),1e-30))
    inva=np.linalg.inv(a)[labels.ravel()];raw=np.zeros((6,6))
    for d in range(3):
        plus=ff[:,:,:,:,d,:].reshape(n,2,6)
        minus=np.roll(ff[:,:,:,:,d,:],1,axis=d).reshape(n,2,6)
        raw+=(np.einsum('nai,nab,nbj->ij',minus,inva,minus)+np.einsum('nai,nab,nbj->ij',plus,inva,plus))/(3*n)
        cross=np.einsum('nai,nab,nbj->ij',minus,inva,plus)/(6*n)
        raw+=cross+cross.T
    mean=ff.mean(axis=(0,1,2)).reshape(6,6);invmean=np.linalg.inv(mean)
    resistance=invmean.T@raw@invmean;lower=np.linalg.inv(resistance)
    result.update(complementary_resistance_upper=resistance.tolist(),continuum_energy_lower=((lower+lower.T)/2).tolist(),
                  RT0_divergence_relative_before_projection=before,RT0_divergence_relative_after_projection=after,
                  seconds=time.perf_counter()-start)
    return result


def pair(labels,phases,h):
    first=fem(labels,phases,h)
    print('  FEM',first['Sa_z_uV_K'],'seconds',first['seconds'],flush=True)
    second=fv(labels,phases,h)
    enc=R.R.enclosure(second['continuum_energy_lower'],first['scaled_energy'])
    print('  FV',second['Sa_z_uV_K'],'C',enc['Sa_z_interval_uV_K'],flush=True)
    return {'fem':first,'fv':second,'continuum_enclosure':enc}


def validate():
    out={'classification':'P numerical validation against exact laminate tensors', 'laminates':[]}
    phases=[R.PHASES[0],R.PHASES[2]];h=np.array([.7,.7,1.3])
    a=np.array([FE.energy(p) for p in phases])
    for axis in range(3):
        lab=np.zeros((4,4,4),np.int32);sl=[slice(None)]*3;sl[axis]=slice(2,None);lab[tuple(sl)]=1
        exact=np.zeros((6,6))
        for d in range(3):
            ids=[d,3+d]
            exact[np.ix_(ids,ids)]=np.linalg.inv(np.linalg.inv(a).mean(0)) if d==axis else a.mean(0)
        x=fem(lab,phases,h);y=fv(lab,phases,h)
        errors={name:float(np.linalg.norm(np.array(matrix)-exact)/np.linalg.norm(exact)) for name,matrix in
                [('FEM',x['scaled_energy']),('FV',y['scaled_energy']),('RT0',y['continuum_energy_lower'])]}
        assert max(errors.values())<1e-9
        out['laminates'].append({'axis':axis,'spacing':h.tolist(),'relative_errors':errors})
    old=R.read('r12_periodic.json')['cases']['W2']['levels'][0]
    labels=np.array(R.read('r12_geometries.json')['cases']['W2']['labels'],np.int32)
    x=fem(labels,R.PHASES);y=fv(labels,R.PHASES)
    out['cubic_reproduction_errors_uV_K']={name:float(np.linalg.norm(np.array(new['S_uV_K'])-np.array(old[name]['S_uV_K']))) for name,new in [('fem',x),('fv',y)]}
    assert max(out['cubic_reproduction_errors_uV_K'].values())<1e-7
    R.save('r13_validation.json',out)
    print(out,flush=True)
