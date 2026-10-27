"""Interface resistance in the two-phase p4 pinwheel of Fig. 2a: a resistive p/n interface is a third transport matrix.

Same zero-thickness neutral-interface model as r8_contact.py (S_i = 0, delta/sigma_i = rho_c, delta/kappa_i = R_K on every
unlike-phase face), applied to the 24 x 24 pinwheel of Fig. 2 refined 2- and 4-fold, with its mirror image and the
three-phase cell of Fig. 2b on the same grids. Output: r8_contact_two_phase.json (finite-volume point estimates).
"""
import json
import numpy as np
from r5_fem import pinwheel,PHASES
from r6_core import ROOT,save
from r8_contact import homogenize_interface
from r8_finite import POINT


def main():
    path=ROOT/'r8_contact_two_phase.json'
    out=json.loads(path.read_text()) if path.exists() else {'number_type':POINT,
        'model':'r8_contact.homogenize_interface on pinwheel(24, 15, n) refined by kron; phases of Fig. 2 (r5_fem.PHASES p, n, weak).',
        'cases':{}}
    cells={'two_phase':(pinwheel(24,15,2),['p','n']),'two_phase_mirror':(pinwheel(24,15,2)[::-1].copy(),['p','n']),
           'three_phase':(pinwheel(24,15,3),['p','n','weak'])}
    for L in (.001,.0003):
        for rk in (0.,1e-5):
            for rho in (0.,1e-9,1e-8,1e-7,1e-6):
                for k in (2,4):
                    for name,(base,ph) in cells.items():
                        key=f'{name}_k{k}_L{L}_RK{rk}_rho{rho}'
                        if key in out['cases']:continue
                        r=homogenize_interface(np.kron(base,np.ones((k,k),int)),[PHASES[x] for x in ph],rho,rk,L)
                        r.pop('scaled_effective_energy');r.update(cell=name,refinement=k)
                        out['cases'][key]=r;save(path.name,out);print(key,round(r['Sa_uV_K'],4),flush=True)


if __name__=='__main__':main()
