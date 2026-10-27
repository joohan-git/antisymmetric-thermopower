"""Uniform Cartesian Q1 finite-domain thermoelectric solver, including insulating matrix.

All finite-domain outputs are uniform-grid point estimates, never continuum certificates.
Uses the energy scaling and constitutive ordering of r5_fem/r6_sector.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from r5_fem import BETA,ENERGY_SCALE

POINT='uniform_grid_point_estimate'


def geometry_shapes(hx,hy):
    signs=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    tensors=np.zeros((4,4,2,2));grads=[]
    for xi in (-1/np.sqrt(3),1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3),1/np.sqrt(3)):
            d=np.column_stack((signs[:,0]*(1+signs[:,1]*eta)/(2*hx),signs[:,1]*(1+signs[:,0]*xi)/(2*hy)))
            tensors+=hx*hy/4*np.einsum('id,je->ijde',d,d);grads.append(d)
    return tensors,np.array(grads),signs/np.array([2*hx,2*hy])


def solve_domain(labels,mats,box,active=None,thermal='vertical',center=None,ri=None,ro=None,electrodes=None):
    started=time.perf_counter();nx,ny=labels.shape;hx,hy=box[0]/nx,box[1]/ny
    ix,iy=np.indices(labels.shape)
    if active is None:active=np.ones_like(labels,dtype=bool)
    ix,iy=ix[active],iy[active];ne=len(ix)
    fullnodes=np.stack((ix*(ny+1)+iy,(ix+1)*(ny+1)+iy,(ix+1)*(ny+1)+iy+1,ix*(ny+1)+iy+1),axis=1)
    used,inverse=np.unique(fullnodes,return_inverse=True);nodes=inverse.reshape(ne,4);nv=len(used)
    coords=np.column_stack((used//(ny+1)*hx,used%(ny+1)*hy))
    element_labels=labels[active];coef=np.array(mats)[element_labels].reshape(ne,2,2,2,2)
    tensor,gaussgrad,gradcenter=geometry_shapes(hx,hy)
    phase_local=np.einsum('ijde,padbe->piajb',tensor,np.array(mats).reshape(-1,2,2,2,2)).reshape(-1,8,8)
    local=phase_local[element_labels]
    physical_dofs=(2*nodes[:,:,None]+np.arange(2)).reshape(ne,8)
    # Electrical line electrodes are floating equipotential node groups, with zero total current.
    dofmap=np.arange(2*nv)
    groups=[]
    for spec in electrodes or []:
        axis,position,low,high=spec
        along=coords[:,1-axis]
        group=np.flatnonzero(np.isclose(coords[:,axis],position,atol=min(hx,hy)*1e-5)&(along>=low-1e-12)&(along<=high+1e-12))
        if len(group):dofmap[2*group]=2*group[0];groups.append(group)
    unique,dofmap=np.unique(dofmap,return_inverse=True);dofs=dofmap[physical_dofs]
    rr=np.broadcast_to(dofs[:,:,None],local.shape).ravel();cc=np.broadcast_to(dofs[:,None,:],local.shape).ravel()
    stiffness=coo_matrix((local.ravel(),(rr,cc)),shape=(len(unique),len(unique))).tocsc()
    del rr,cc
    conducting=np.linalg.norm(np.array(mats)[:,:2,:2],axis=(1,2))>0
    electric_nodes=np.unique(nodes[conducting[element_labels]])
    inactive_electric=np.setdiff1d(np.arange(nv),electric_nodes)
    if thermal=='vertical':
        hot=np.flatnonzero(np.isclose(coords[:,1],box[1]));cold=np.flatnonzero(np.isclose(coords[:,1],0))
    elif thermal=='annulus':
        if center is None:center=np.array(box)/2
        hotset=set();coldset=set()
        for dx,dy,face in ((-1,0,(0,3)),(1,0,(1,2)),(0,-1,(0,1)),(0,1,(3,2))):
            xx=ix+dx;yy=iy+dy;outside=(xx<0)|(xx>=nx)|(yy<0)|(yy>=ny)
            clippedx=xx.clip(0,nx-1);clippedy=yy.clip(0,ny-1)
            exposed=outside|(~active[clippedx,clippedy])
            radii=np.hypot((xx+.5)*hx-center[0],(yy+.5)*hy-center[1])
            inner=exposed&(radii<(ri+ro)/2);outer=exposed&(~inner)
            hotset.update(nodes[inner][:,face].ravel());coldset.update(nodes[outer][:,face].ravel())
        hot=np.array(sorted(hotset),int);cold=np.array(sorted(coldset),int)
    else:raise ValueError(thermal)
    assert not len(np.intersect1d(hot,cold))
    fixed=np.unique(np.r_[dofmap[2*inactive_electric],dofmap[2*electric_nodes[:1]],dofmap[2*hot+1],dofmap[2*cold+1]])
    u=np.zeros(len(unique));u[dofmap[2*hot+1]]=1.
    free=np.setdiff1d(np.arange(len(u)),fixed)
    rhs=-stiffness[free][:,fixed]@u[fixed];mat=stiffness[free][:,free]
    scale=1/np.sqrt(mat.diagonal())
    lu=splu((diags(scale)@mat@diags(scale)).tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
    u[free]=scale*lu.solve(scale*rhs)
    u[free]+=scale*lu.solve(scale*(rhs-mat@u[free]))
    residual=np.linalg.norm(mat@u[free]-rhs)/max(np.linalg.norm(rhs),1e-30)
    reaction=stiffness@u
    hotheat=300*ENERGY_SCALE*reaction[dofmap[2*hot+1]].sum()
    coldheat=300*ENERGY_SCALE*reaction[dofmap[2*cold+1]].sum()
    nodal=u[dofmap].reshape(nv,2);el_u=nodal[nodes]
    grad=np.einsum('id,eia->ead',gradcenter,el_u)
    flux=-np.einsum('eadbf,ebf->ead',coef,grad)
    current=flux[:,0]*ENERGY_SCALE/BETA
    stats={'number_type':POINT,'grid_shape':[nx,ny],'elements':ne,'nodes':nv,'box_m':list(box),
           'residual':float(residual),'heat_input_W_per_m':float(hotheat),'heat_output_W_per_m':float(-coldheat),
           'seconds':time.perf_counter()-started,'floating_electrode_nodes':[len(x) for x in groups]}
    state={'nodal':nodal,'nodes':nodes,'used':used,'coords':coords,'ix':ix,'iy':iy,'hx':hx,'hy':hy,
           'labels':labels,'element_labels':element_labels,'active':active,'current':current,'grad':grad,
           'local':local,'el_u':el_u,'gaussgrad':gaussgrad,'center':np.array(box)/2,'groups':groups,
           'mats':np.array(mats),'coef':coef}
    return stats,state


def sample(state,points,gradient=False):
    points=np.array(points);hx,hy=state['hx'],state['hy'];nx,ny=state['labels'].shape
    ix=np.floor(points[:,0]/hx).astype(int).clip(0,nx-1);iy=np.floor(points[:,1]/hy).astype(int).clip(0,ny-1)
    lookup=np.full((nx+1)*(ny+1),-1,int);lookup[state['used']]=np.arange(len(state['used']))
    full=np.stack((ix*(ny+1)+iy,(ix+1)*(ny+1)+iy,(ix+1)*(ny+1)+iy+1,ix*(ny+1)+iy+1),axis=1)
    nodes=lookup[full];assert nodes.min()>=0
    values=state['nodal'][nodes];x=points[:,0]/hx-ix;y=points[:,1]/hy-iy
    if not gradient:
        weights=np.column_stack(((1-x)*(1-y),x*(1-y),x*y,(1-x)*y))
        return np.einsum('ei,eia->ea',weights,values)
    deriv=np.stack((np.column_stack((-(1-y),1-y,y,-y))/hx,np.column_stack((-(1-x),-x,x,1-x))/hy),axis=-1)
    return np.einsum('eid,eia->ead',deriv,values)


def winding_outputs(state,thickness=.001):
    x=(state['ix']+.5)*state['hx']-state['center'][0];y=(state['iy']+.5)*state['hy']-state['center'][1]
    rr=x*x+y*y;j=state['current'];area=state['hx']*state['hy']
    centroid=thickness*np.sum((x*j[:,1]-y*j[:,0])/rr)*area/(2*np.pi)
    xy=state['coords'][state['nodes']]-state['center'];angles=np.arctan2(xy[:,:,1],xy[:,:,0]);mid=np.arctan2(y,x)
    angles=mid[:,None]+(angles-mid[:,None]+np.pi)%(2*np.pi)-np.pi
    virtual=np.zeros_like(state['el_u']);virtual[:,:,0]=angles
    # The locally unwrapped nodal angle is a discrete closed 1-form of winding one.
    discrete=-thickness*ENERGY_SCALE/BETA*np.einsum('ei,eij,ej->',virtual.reshape(-1,8),state['local'],state['el_u'].reshape(-1,8))/(2*np.pi)
    return {'centroid_current_A':float(centroid),'conservative_cut_current_A':float(discrete),
            'postprocessing_difference_A':float(centroid-discrete)}
