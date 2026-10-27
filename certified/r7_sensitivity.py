"""Small p4 junction regularizations, solely to assess experiment sensitivity."""
import numpy as np
from r7 import readonly_inputs,map_diagnostics
from r6_core import save,solve


def main():
    family,mats=readonly_inputs();n=64;base=family.F3_pinwheel_tiling(n,r=.6412,split=.9)
    phases=[(10.45,10e-6,.9012),mats['nBiTeSe'],mats['pBiSbTe']]
    sites=[]
    for i in range(n):
        for j in range(n):
            a,b,c,d=base[i,j],base[i-1,j],base[i-1,j-1],base[i,j-1]
            if a==c and b==d and a!=b:sites.append((i,j))
    out={'N':n,'sites':sites,'patch_definition':'Set four pixels adjacent to each alternating vertex to phase 0 or 1; both changes preserve p4','cases':{}}
    for label in (0,1):
        changed=base.copy()
        for i,j in sites:
            for di,dj in ((0,0),(-1,0),(-1,-1),(0,-1)):
                changed[(i+di)%n,(j+dj)%n]=label
        assert np.array_equal(changed,np.rot90(changed))
        case={'labels':changed.tolist(),'changed_pixels':int(np.count_nonzero(changed!=base)),
              'changed_fraction':float(np.mean(changed!=base)),'diagnostics':map_diagnostics(changed),'rows':[]}
        out['cases'][str(label)]=case
        for k in (2,4):
            g=np.kron(changed,np.ones((k,k),int));row={'k':k}
            for method in ('fem','fv'):
                r=solve(g,phases,method);row[method]=r
                print('patch',label,'k',k,method,r['Sa'],r['sigma_iso'],flush=True)
            case['rows'].append(row);save('r7_sensitivity.json',out)


if __name__=='__main__':main()
