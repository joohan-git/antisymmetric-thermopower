"""Finite A specimens, floating line electrodes and non-invasive point probes."""

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
from common.fem2d import material_energy
from common.transport2d import ROOT, save, solve
from certified_bounds.design_enclosures import readonly_inputs
from devices.contact import junction_gap
from common.finite_sample import solve_domain, sample, POINT


def run(nx=8,ny=4,N=32,kind='A',electrode='point',mirror=False,refine=1):
    f,m=readonly_inputs();ph=[(10.45,10e-6,.9012),m['nBiTeSe'],m['pBiSbTe']]
    base=f.F3_pinwheel_tiling(N,r=.6412,split=.9)
    if kind=='gap':base=junction_gap(base,4/N)
    if kind=='gap4pct':base=junction_gap(base,.04)
    if kind=='two_phase':base[base==0]=2
    if kind=='mirror_symmetric':
        q=base[:N//2,:N//2];base=np.block([[q,q[:,::-1]],[q[::-1],q[::-1,::-1]]])
    if mirror:base=base[::-1,:]
    mats=[np.kron(material_energy(p),np.eye(2)) for p in ph]
    if kind=='homogeneous':
        cert=json.loads((ROOT/'certified_bounds/design_enclosures.json').read_text())['cases']['A_64']['sector'][-1]
        mats=[np.array(cert['fem']['scaled_effective_energy'])];base=np.zeros_like(base)
        per=cert['fem']
    else:per=solve(base,ph)
    if refine>1:
        base=np.kron(base,np.ones((refine,refine),int))
        if kind!='homogeneous':per=solve(base,ph)
    L=.001;W=nx*L;H=ny*L;labels=np.tile(base,(nx,ny));electrodes=[]
    if electrode=='full_line':electrodes=[(0,x,0,H) for x in (0.,W)]
    if electrode=='middle_line':electrodes=[(0,x,.4*H,.6*H) for x in (0.,W)]
    if electrode=='short_line':electrodes=[(0,x,H/2-.05*L,H/2+.05*L) for x in (0.,W)]
    stats,state=solve_domain(labels,mats,(W,H),electrodes=electrodes)
    cert_sa=49.8304472*(-1 if mirror else 1)
    def measurement(yl,yr):
        u=sample(state,[(0,yl*H),(W,yr*H)]);deltaT=float(u[1,1]-u[0,1]);v=float((u[0,0]-u[1,0])*200)
        corrected=v-per['Sxx']*deltaT
        return {'left_y_over_H':yl,'right_y_over_H':yr,'V_left_minus_right_uV':v,'deltaT_right_minus_left_K':deltaT,
            'V_temperature_corrected_uV':corrected,'ratio_to_A64_continuum_tensor_point':v/(cert_sa*W/H),
            'ratio_corrected_to_A64':corrected/(cert_sa*W/H),
            'ratio_to_same_grid_bulk_Sa':v/(per['Sa']*W/H) if abs(per['Sa'])>1e-7 else None}
    stats.update(kind=kind,cells=[nx,ny],base_pixels_per_cell=N,fixed_geometry_refinement=refine,pixels_per_cell=N*refine,electrode=electrode,mirror=mirror,periodic_same_grid=per,
        aligned_probes=[measurement(y,y) for y in (.1,.25,.5,.75,.9)],
        misaligned_probes=[measurement(.45,.55),measurement(.55,.45)])
    return stats


def main():
    path=ROOT/'devices/electrodes.json';out=json.loads(path.read_text()) if path.exists() else {'number_type':POINT,'cases':{}}
    configs=[]
    for nx,ny in ((8,4),(16,8)):
        for kind in ('A','gap','homogeneous','two_phase','mirror_symmetric'):
            configs.append((f'{kind}_{nx}',dict(nx=nx,ny=ny,kind=kind)))
        for electrode in ('middle_line','full_line'):
            configs.append((f'A_{nx}_{electrode}',dict(nx=nx,ny=ny,electrode=electrode)))
        configs.append((f'A_{nx}_mirror',dict(nx=nx,ny=ny,mirror=True)))
    configs.append(('A_8_N64',dict(N=64)))
    for nx,ny in ((8,4),(16,8)):
        configs.append((f'gap4pct_{nx}',dict(nx=nx,ny=ny,N=50,kind='gap4pct')))
        configs.append((f'gap4pct_{nx}_line',dict(nx=nx,ny=ny,N=50,kind='gap4pct',electrode='middle_line')))
        configs.append((f'gap4pct_{nx}_shortline',dict(nx=nx,ny=ny,N=50,kind='gap4pct',electrode='short_line')))
    configs.append(('gap4pct_8_refine2',dict(N=50,kind='gap4pct',refine=2)))
    configs.append(('homogeneous_8_fullline',dict(kind='homogeneous',electrode='full_line')))
    for key,kw in configs:
        if key in out['cases']:continue
        r=run(**kw);out['cases'][key]=r;save(path.relative_to(ROOT).as_posix(),out)
        print(key,[(p['left_y_over_H'],p['V_left_minus_right_uV']) for p in r['aligned_probes']],flush=True)


if __name__=='__main__':main()
