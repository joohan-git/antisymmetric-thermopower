"""Highest-priority test: open spiral voltage, radial-bath limit and thermal redistribution."""
import json
import numpy as np
from r6_core import ROOT,save
from r7 import schur
from r8_finite import solve_domain,sample,POINT


def run(n=2,N=256,ratio=1.,rstart=.003,rend=.008,mirror=False,width=.0004):
    data=json.loads((ROOT/'r7_results.json').read_text())['cases']['A_64']['sector'][-1]
    c=np.array(data['fem']['scaled_effective_energy']);sigma=data['fem']['sigma_iso'];sa=data['fem']['Sa']*1e-6;sd=data['fem']['Sxx']*1e-6
    kap=float(300*.004*np.trace(schur(c))/2)
    if mirror:
        reflect=np.diag([1,-1,1,-1]);c=reflect@c@reflect;sa=-sa
    box=.02;ri=.002;ro=.0093;x,y=np.meshgrid((np.arange(N)+.5)*box/N-box/2,(np.arange(N)+.5)*box/N-box/2,indexing='ij')
    radius=np.hypot(x,y);angle=np.arctan2(y,x);direction=np.sign(n);turns=abs(n)
    angle=(direction*angle)%(2*np.pi)
    nearest=np.full_like(x,1.)
    for k in range(turns):
        t=angle+2*np.pi*k;r=rstart+(rend-rstart)*t/(2*np.pi*turns)
        dr=(rend-rstart)/(2*np.pi*turns)
        nearest=np.minimum(nearest,abs(radius-r)/np.sqrt(1+(dr/r)**2))
    band=nearest<width/2
    band|=((x-rstart)**2+y*y<(width/2)**2)|((x-rend)**2+y*y<(width/2)**2)
    active=(radius>ri)&(radius<ro)
    # Electrical perfect-insulator limit, with finite thermal conductivity.
    matrix=np.zeros((4,4));matrix[2:,2:]=np.eye(2)*kap*ratio/(300*.004)
    labels=band.astype(int);stats,state=solve_domain(labels,[matrix,c],(box,box),active,'annulus',ri=ri,ro=ro)
    t=np.linspace(0,2*np.pi*turns,12001);r=rstart+(rend-rstart)*t/(2*np.pi*turns);theta=direction*t
    points=np.column_stack((r*np.cos(theta)+box/2,r*np.sin(theta)+box/2))
    u=sample(state,points[[0,-1]]);deltaT=u[1,1]-u[0,1]
    voltage=(u[0,0]-u[1,0])*200e-6;corrected=voltage-sd*deltaT
    grad=sample(state,points,True)[:,1]
    dr=(rend-rstart)/(2*np.pi*turns)
    dx=dr*np.cos(theta)-direction*r*np.sin(theta);dy=dr*np.sin(theta)+direction*r*np.cos(theta)
    path_integral=float(np.trapezoid(grad[:,1]*dx-grad[:,0]*dy,t))
    naive=n*sa*stats['heat_input_W_per_m']/kap
    pathprediction=sa*path_integral
    stats.update(turns=n,mirror=mirror,matrix_kappa_ratio=ratio,strip_width_m=width,rstart_m=rstart,rend_m=rend,
        Sa_uV_K=sa*1e6,Sxx_uV_K=sd*1e6,kappa_strip_W_mK=kap,
        voltage_start_minus_end_uV=voltage*1e6,terminal_deltaT_end_minus_start_K=float(deltaT),
        longitudinal_corrected_uV=corrected*1e6,winding_prediction_uV=naive*1e6,
        local_gradient_prediction_uV=pathprediction*1e6,path_integral_K=path_integral,
        ratio_to_winding_prediction=float(corrected/naive),ratio_to_local_gradient_prediction=float(corrected/pathprediction),
        input_material_note='A64 r7 upper energy tensor used as a fixed homogeneous model; finite-domain values are point estimates')
    return stats


def main():
    path=ROOT/'r8_winding.json';out=json.loads(path.read_text()) if path.exists() else {'number_type':POINT,'cases':{}}
    configs=[]
    for ratio in (1.,.03):
        for n in (1,2,3,-1):configs.append((f'n{n}_k{ratio}',dict(n=n,ratio=ratio)))
        configs.append((f'pitch_k{ratio}',dict(n=2,ratio=ratio,rstart=.004,rend=.0075)))
        configs.append((f'mirror_k{ratio}',dict(n=2,ratio=ratio,mirror=True)))
        for n in (1,3):configs.append((f'fine_n{n}_k{ratio}',dict(n=n,ratio=ratio,N=512)))
    for n in (1,2,3):configs.append((f'finest_n{n}_k0.03',dict(n=n,ratio=.03,N=1024)))
    configs.append(('fine_pitch_k0.03',dict(n=2,ratio=.03,N=512,rstart=.004,rend=.0075)))
    configs.append(('finest_pitch_k0.03',dict(n=2,ratio=.03,N=1024,rstart=.004,rend=.0075)))
    for key,kw in configs:
        if key in out['cases']:continue
        r=run(**kw);out['cases'][key]=r;save(path.name,out)
        print(key,'Vcorr',r['longitudinal_corrected_uV'],'ratios',r['ratio_to_winding_prediction'],r['ratio_to_local_gradient_prediction'],'sec',r['seconds'],flush=True)


if __name__=='__main__':main()
