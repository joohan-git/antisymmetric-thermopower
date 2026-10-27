"""Separate old annulus quadrature artifacts from conserved two-phase current."""
import json
import numpy as np
from r5_fem import PHASES,pinwheel,material_energy
from r6_core import ROOT,save
from r8_finite import solve_domain,winding_outputs,POINT


def run(tiles=8,k=1,contrast=100,offset=0,collar=False,mirror_control=False,frozen_boundary=False,collar_fraction=1.):
    base=pinwheel(24,15,2)
    if mirror_control:base=np.concatenate((base[:12],base[:12][::-1]),axis=0)
    base=np.roll(base,offset,axis=0);labels=np.kron(np.tile(base,(tiles,tiles)),np.ones((k,k),int));N=len(labels)
    ph=[PHASES['p'],(1e5*contrast,0.,.5*contrast)]
    mats=[np.kron(material_energy(p),np.eye(2)) for p in ph]
    x,y=np.meshgrid((np.arange(N)+.5)*.02/N-.01,(np.arange(N)+.5)*.02/N-.01,indexing='ij');r=np.hypot(x,y)
    active=(r>.004)&(r<.009)
    if frozen_boundary:
        n0=24*tiles;xx,yy=np.meshgrid((np.arange(n0)+.5)*.02/n0-.01,(np.arange(n0)+.5)*.02/n0-.01,indexing='ij')
        active=np.kron((np.hypot(xx,yy)>.004)&(np.hypot(xx,yy)<.009),np.ones((k,k),bool))
    if collar:labels[(r<.004+collar_fraction*.02/tiles)|(r>.009-collar_fraction*.02/tiles)]=0
    stats,state=solve_domain(labels,mats,(.02,.02),active,'annulus',ri=.004,ro=.009)
    stats.update(winding_outputs(state));stats.update(tiles_per_box=tiles,fixed_geometry_refinement=k,conductivity_contrast=contrast,offset_pixels=offset,homogeneous_p_boundary_collar=collar,mirror_control=mirror_control,frozen_boundary=frozen_boundary)
    reference=.001*32248.2197*49.8304472e-6
    stats['reference_A_homogeneous_current_A']=reference;stats['absolute_artifact_fraction_of_A']=abs(stats['conservative_cut_current_A'])/reference
    stats['collar_fraction_of_cell']=collar_fraction if collar else 0.;stats['remaining_metal_fraction']=float(np.mean(labels[active]==1))
    return stats


def main():
    path=ROOT/'r8_ring.json';out=json.loads(path.read_text()) if path.exists() else {'number_type':POINT,'cases':{}}
    configurations=[(f't{t}',dict(tiles=t)) for t in (8,16,24)]
    configurations += [(f'fixed8_k{k}',dict(tiles=8,k=k,frozen_boundary=True)) for k in (2,4)]
    configurations += [(f'offset{x}',dict(offset=x)) for x in (6,12)]
    configurations += [('collar8',dict(collar=True)),('collar16',dict(tiles=16,collar=True)),('contrast10',dict(contrast=10)),('contrast3',dict(contrast=3)),('mirror',dict(mirror_control=True))]
    configurations += [(f'collar{t}_quarter',dict(tiles=t,collar=True,collar_fraction=.25)) for t in (8,16,24)]
    configurations += [('contrast3_t16',dict(tiles=16,contrast=3)),('contrast3_t24',dict(tiles=24,contrast=3)),('contrast3_t16_collar',dict(tiles=16,contrast=3,collar=True,collar_fraction=.25))]
    configurations += [(f'contrast3_fixed8_k{k}',dict(k=k,contrast=3,frozen_boundary=True)) for k in (2,4)]
    configurations += [(f'contrast3_t16_collar_offset{o}',dict(tiles=16,contrast=3,collar=True,collar_fraction=.25,offset=o)) for o in (6,12)]
    for key,kw in configurations:
        if key in out['cases']:continue
        r=run(**kw);out['cases'][key]=r;save(path.name,out);print(key,r['centroid_current_A'],r['conservative_cut_current_A'],flush=True)


if __name__=='__main__':main()
