"""Transmission-corner angular eigenvalues and certified material-only bounds."""

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
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from common.transport2d import original_cell3, material_energy, PHASES, MATERIALS, save, ROOT, enclosure_from_pair


def exponent(phases,labels,nper=40):
    coeff=np.array([material_energy(p) for p in phases])
    ne=4*nper;h=2*np.pi/ne
    stiff=np.zeros((2*ne,2*ne));mass=stiff.copy()
    for i in range(ne):
        dofs=np.array([2*i,2*i+1,2*((i+1)%ne),2*((i+1)%ne)+1])
        a=coeff[labels[i//nper]]
        stiff[np.ix_(dofs,dofs)]+=np.kron(np.array([[1,-1],[-1,1]])/h,a)
        mass[np.ix_(dofs,dofs)]+=np.kron(np.array([[2,1],[1,2]])*h/6,a)
    eigen=eigh(stiff,mass,subset_by_index=(0,9),check_finite=False)[0]
    positive=np.sqrt(eigen[eigen>1e-7])
    def mismatch(lam):
        c=np.cos(lam*np.pi/2);s=np.sin(lam*np.pi/2);prod=np.eye(4)
        for label in labels:
            a=coeff[label]
            trans=np.block([[c*np.eye(2),s*np.linalg.inv(a)],[-s*a,c*np.eye(2)]])
            prod=trans@prod
        return np.linalg.svd(prod-np.eye(4),compute_uv=False)[-1]
    fe=float(positive[0])
    fit=minimize_scalar(mismatch,bounds=(.98*fe,1.01*fe),method='bounded',options={'xatol':1e-13})
    return {'lambda_FE':fe,'lambda_transfer':float(fit.x),'transfer_singular_value':float(fit.fun),'energy_power_2lambda':float(2*fit.x)}


def bound_for_fractions(phases,fractions):
    aa=np.array([material_energy(p) for p in phases]);f=np.array(fractions)
    upper=np.kron(np.einsum('i,iab->ab',f,aa),np.eye(2))
    dualupper=np.kron(np.einsum('i,iab->ab',f,np.linalg.inv(aa)),np.eye(2))
    enc=enclosure_from_pair({'scaled_effective_energy':upper.tolist()},
                            {'scaled_effective_energy':dualupper.tolist(),'seconds':0})
    m=float(np.linalg.eigvalsh(aa).min());big=float(np.linalg.eigvalsh(aa).max())
    enc['ellipticity_bound_Sa_uV_K']=200*(big-m)/(2*np.sqrt(big*m))
    enc['phase_scaled_min_eigenvalue']=m;enc['phase_scaled_max_eigenvalue']=big
    return enc


def main():
    base=original_cell3();out={'corners':{},'material_only_bounds':{}}
    for name,phases in [('baseline',[PHASES[k] for k in ('p','n','weak')]),('metal',[MATERIALS[k] for k in ('p','n','Cu')])]:
        patterns={}
        for i in range(len(base)):
            for j in range(len(base)):
                t=(int(base[i,j]),int(base[i-1,j]),int(base[i-1,j-1]),int(base[i,j-1]))
                canon=min(tuple(np.roll(t,l)) for l in range(4))
                patterns[canon]=patterns.get(canon,0)+1
        rows=[]
        for labels,count in patterns.items():
            r=exponent(phases,labels);r['quadrants']=list(map(int,labels));r['count']=count;rows.append(r)
        rows.sort(key=lambda r:r['lambda_transfer']);out['corners'][name]=rows
        print(name,'slowest',rows[0],flush=True)
        save('certified_bounds/corner_exponents.json',out)
    designs=json.loads((ROOT/'planar_cells/material_designs.json').read_text())
    for name in ('best3','best4'):
        d=designs[name];phases=[MATERIALS[m] for m in d['materials']]
        out['material_only_bounds'][name]=bound_for_fractions(phases,d['fractions'])
    ideal=ROOT/'planar_cells/ideal_materials.json'
    if ideal.exists():
        d=json.loads(ideal.read_text());base=np.array(d['base_labels']);fractions=np.bincount(base.ravel())/base.size
        out['material_only_bounds']['ideal']=bound_for_fractions(d['best']['phases'],fractions)
    four=ROOT/'planar_cells/four_phase_design.json'
    if four.exists():
        d=json.loads(four.read_text());phases=[MATERIALS[m] for m in d['best']['materials']]
        out['material_only_bounds']['best4_overlay']=bound_for_fractions(phases,d['best']['fractions'])
    aa=np.array([material_energy(p) for name,p in MATERIALS.items() if name!='polymer_void'])
    eig=np.linalg.eigvalsh(aa);lo=float(eig.min());hi=float(eig.max())
    out['all_six_material_global_bound']={'scaled_min_eigenvalue':lo,'scaled_max_eigenvalue':hi,
        'Sa_upper_uV_K':200*(hi-lo)/(2*np.sqrt(hi*lo))}
    save('certified_bounds/corner_exponents.json',out)


if __name__=='__main__':main()
