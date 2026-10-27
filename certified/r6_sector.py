"""Conforming corner-sector enrichment with exact radial dependence.

Every pixel vertex has a circular patch. In that disk, angular finite elements
are separated from radius and the semi-discrete radial problem is solved exactly:
Ktheta Y = Mtheta Y lambda^2; DtN = Mtheta Y lambda Y.T Mtheta.
Outside disks, mapped Q1 elements fill the four corner squares of each pixel.
Their interface traces use the same angular basis. This removes the artificial
point contact truncation while retaining the variational upper-bound property.
All integration is high-order Gauss; floating point/quadrature are not interval certified.
"""
import time
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from numpy.polynomial.legendre import leggauss
from r5_fem import material_energy,BETA,ENERGY_SCALE
from r6_core import compact,inverse_phases,enclosure_from_pair


def ray(t):
    return np.array([1.,t]) if t<=1 else np.array([2-t,1.])


@lru_cache(maxsize=6)
def outer_template(m,nr,radius=.22,ngauss=10):
    tnodes=np.linspace(0,2,2*m+1);snodes=np.linspace(0,1,nr+1)
    coords=[];elements=[];shapes=[]
    gauss,weights=leggauss(ngauss)
    for corner in ((0,0),(1,0),(1,1),(0,1)):
        offset=np.array(corner);sign=1-2*offset
        node0=len(coords)
        for s in snodes:
            for t in tnodes:
                a=ray(t);norm=np.linalg.norm(a)
                coords.append(offset+sign*((1-s)*radius/norm+s*.5)*a)
        for j in range(nr):
            for i in range(2*m):
                s0,s1=snodes[j:j+2];t0,t1=tnodes[i:i+2]
                nodes=np.array([j*(2*m+1)+i,(j+1)*(2*m+1)+i,(j+1)*(2*m+1)+i+1,j*(2*m+1)+i+1])+node0
                elements.append(nodes);ke=np.zeros((4,4))
                for xi,wx in zip(gauss,weights):
                    for eta,wy in zip(gauss,weights):
                        u=(xi+1)/2;v=(eta+1)/2
                        s=s0+u*(s1-s0);t=t0+v*(t1-t0)
                        a=ray(t);at=np.array([0.,1.]) if t<1 else np.array([-1.,0.])
                        norm=np.linalg.norm(a);unit=a/norm
                        ut=at/norm-a*np.dot(a,at)/norm**3
                        rout=.5*norm;routt=.5*np.dot(a,at)/norm
                        rr=radius+s*(rout-radius)
                        xs=(rout-radius)*unit
                        xt=s*routt*unit+rr*ut
                        jac=np.column_stack((sign*xs*(s1-s0),sign*xt*(t1-t0)))
                        deriv=np.array([[-(1-v),-(1-u)],[1-v,-u],[v,u],[-v,1-u]])
                        grad=deriv@np.linalg.inv(jac)
                        ke+=wx*wy/4*abs(np.linalg.det(jac))*(grad@grad.T)
                shapes.append(ke)
    return np.array(coords),np.array(elements),np.array(shapes)


@lru_cache(maxsize=6)
def angular_template(m,ngauss=16):
    angles=np.array(sorted(set([float(np.arctan2(ray(t)[1],ray(t)[0])+q*np.pi/2)
                                  for q in range(4) for t in np.linspace(0,2,2*m+1)[:-1]])))
    # Each quarter uses 2m intervals, endpoint belongs to following quarter.
    assert len(angles)==8*m
    gauss,weights=leggauss(ngauss);mass=[];stiff=[]
    for i,theta0 in enumerate(angles):
        theta1=angles[i+1] if i+1<len(angles) else 2*np.pi
        mid=(theta0+theta1)/2
        tangent=abs(np.cos(mid))>=abs(np.sin(mid))
        def coord(theta):return np.tan(theta) if tangent else 1/np.tan(theta)
        z0,z1=coord(theta0),coord(theta1)
        mm=np.zeros((2,2));kk=mm.copy()
        for x,w in zip(gauss,weights):
            theta=(theta0+theta1)/2+x*(theta1-theta0)/2
            z=coord(theta);dz=1/np.cos(theta)**2 if tangent else -1/np.sin(theta)**2
            f=(z-z0)/(z1-z0);nn=np.array([1-f,f]);dd=np.array([-1.,1.])*dz/(z1-z0)
            mm+=w*(theta1-theta0)/2*np.outer(nn,nn);kk+=w*(theta1-theta0)/2*np.outer(dd,dd)
        mass.append(mm);stiff.append(kk)
    return angles,np.array(mass),np.array(stiff)


def dtn(coeff,quadrants,m):
    angles,mm,kk=angular_template(m);n=len(angles)
    mass=np.zeros((2*n,2*n));stiff=mass.copy()
    for i in range(n):
        dofs=np.array([2*i,2*i+1,2*((i+1)%n),2*((i+1)%n)+1])
        a=coeff[quadrants[i//(2*m)]]
        mass[np.ix_(dofs,dofs)]+=np.kron(mm[i],a)
        stiff[np.ix_(dofs,dofs)]+=np.kron(kk[i],a)
    values,vectors=eigh(stiff,mass,check_finite=False)
    assert values[2]>0
    values[:2]=0
    root=np.sqrt(np.maximum(values,0))
    b=(mass@vectors*root)@vectors.T@mass
    # Preserve exact constant-field nullspace despite eigensolver roundoff.
    constants=np.tile(np.eye(2),(n,1))/np.sqrt(n)
    project=np.eye(2*n)-constants@constants.T
    b=project@b@project
    return (b+b.T)/2,float(root[2])


def sector_homogenize(base,phases,m=4,nr=3,radius=.22,ngauss=10):
    started=time.perf_counter();n=len(base);count=n*n
    coords_local,elem_local,shape=outer_template(m,nr,radius,ngauss)
    shifts=np.array([(i,j) for i in range(n) for j in range(n)])
    allcoords=(coords_local[None,:,:]+shifts[:,None,:]).reshape(-1,2)%n
    rounded=np.round(allcoords,11);rounded[rounded==n]=0
    vertices,index=np.unique(rounded,axis=0,return_inverse=True);nv=len(vertices)
    localmap=index.reshape(count,len(coords_local))
    nodes=localmap[:,elem_local].reshape(-1,4)
    dofs=(2*nodes[:,:,None]+np.arange(2)).reshape(-1,8);ne=len(nodes)
    coeff=np.array([material_energy(p) for p in phases])
    values=np.einsum('eij,pab->peiajb',shape,coeff[base.ravel()]).reshape(ne,8,8)
    row=[np.broadcast_to(dofs[:,:,None],values.shape).ravel()]
    col=[np.broadcast_to(dofs[:,None,:],values.shape).ravel()]
    entries=[values.ravel()]
    g=np.zeros((len(elem_local),4,2,4))
    xy=coords_local[elem_local]/n
    g[:,:,0,:2]=xy;g[:,:,1,2:]=xy;g=g.reshape(len(elem_local),8,4)
    g=np.tile(g,(count,1,1))
    force=-np.einsum('eij,ejk->eik',values,g)
    rhs=np.zeros((2*nv,4));np.add.at(rhs,dofs.ravel(),force.reshape(-1,4))
    direct=np.einsum('eia,eij,ejb->ab',g,values,g)
    lookup={tuple(x):i for i,x in enumerate(vertices)}
    angles,unused,unused2=angular_template(m)
    circle=radius*np.column_stack((np.cos(angles),np.sin(angles)))
    gc=np.zeros((len(angles),2,4));gc[:,0,:2]=circle/n;gc[:,1,2:]=circle/n;gc=gc.reshape(-1,4)
    cache={};exponents=[]
    for i in range(n):
        for j in range(n):
            pattern=(int(base[i,j]),int(base[i-1,j]),int(base[i-1,j-1]),int(base[i,j-1]))
            if pattern not in cache:cache[pattern]=dtn(coeff,pattern,m)
            block,lam=cache[pattern];exponents.append(lam)
            cp=np.round((circle+np.array([i,j]))%n,11);cp[cp==n]=0
            bn=np.array([lookup[tuple(p)] for p in cp]);bd=(2*bn[:,None]+np.arange(2)).ravel()
            row.append(np.broadcast_to(bd[:,None],block.shape).ravel())
            col.append(np.broadcast_to(bd[None,:],block.shape).ravel())
            entries.append(block.ravel())
            np.add.at(rhs,bd,-block@gc);direct+=gc.T@block@gc
    stiffness=coo_matrix((np.concatenate(entries),(np.concatenate(row),np.concatenate(col))),shape=(2*nv,2*nv)).tocsc()
    del row,col,entries,values,g,force
    mat=stiffness[2:,2:];diagonal=1/np.sqrt(mat.diagonal())
    lu=splu((diags(diagonal)@mat@diags(diagonal)).tocsc(),
            permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
    correction=np.zeros_like(rhs);correction[2:]=diagonal[:,None]*lu.solve(diagonal[:,None]*rhs[2:])
    for unused in range(2):
        correction[2:]+=diagonal[:,None]*lu.solve(diagonal[:,None]*(rhs[2:]-mat@correction[2:]))
    eff=direct-rhs.T@correction
    factors=np.array([BETA,BETA,1.,1.]);physical=eff*ENERGY_SCALE/factors[:,None]/factors[None,:]
    sig=physical[:2,:2];alpha=physical[:2,2:];s=np.linalg.solve(sig,alpha)
    return compact({'m':m,'radial_elements':nr,'radius_per_pixel':radius,'outer_gauss_order':ngauss,
        'vertices':nv,'outer_elements':ne,'smallest_angular_exponent':min(exponents),
        'sigma':sig.tolist(),'alpha':alpha.tolist(),'S_uV_K':(1e6*s).tolist(),
        'Santi_main_uV_K':float(5e5*(s[0,1]-s[1,0])),
        'scaled_effective_energy':eff.tolist(),
        'residual_relative':float(np.linalg.norm(stiffness@correction-rhs)/max(np.linalg.norm(rhs),1e-30)),
        'onsager_relative':float(np.linalg.norm(eff-eff.T)/np.linalg.norm(eff)),
        'seconds':time.perf_counter()-started})


def sector_pair(base,phases,m=4,nr=3,radius=.22,ngauss=10):
    assert np.array_equal(base,np.rot90(base))
    a=sector_homogenize(base,phases,m,nr,radius,ngauss)
    b=sector_homogenize(base,inverse_phases(phases),m,nr,radius,ngauss)
    return {'fem':a,'enclosure':enclosure_from_pair(a,b)}
