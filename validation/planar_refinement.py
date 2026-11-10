"""Fixed pixel geometry refinement, analytical tests, and variational certificates."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import time
import numpy as np
from scipy.linalg import fractional_matrix_power
from common.transport2d import save, solve, sources, original_cell3, pinwheel, PHASES, MATERIALS, material_energy, energy_enclosure, richardson


def main():
    started = time.perf_counter()
    data = {'source_sha256_before': sources(), 'tests': {}, 'fixed_geometry': {}}
    def checkpoint():
        data['elapsed_seconds'] = time.perf_counter()-started
        save('validation/planar_refinement.json', data)
    for n in (24,48,96):
        data['tests']['geometry_identity_'+str(n)] = bool(np.array_equal(original_cell3(n,15,n//4),pinwheel(n,15,3)))
    for name, phases in [('moderate',[PHASES['p'],PHASES['weak']]),
                         ('metal',[MATERIALS['p'],MATERIALS['Cu']])]:
        base = np.zeros((12,12),int); base[6:] = 1
        matrices = [material_energy(p) for p in phases]
        normal = np.linalg.inv(sum(np.linalg.inv(x) for x in matrices)/2)
        tangent = sum(matrices)/2
        exact = np.zeros((4,4))
        exact[np.ix_([0,2],[0,2])] = normal
        exact[np.ix_([1,3],[1,3])] = tangent
        rows = []
        for method in ('fem','fv'):
            result = solve(base,phases,method)
            result['method'] = method
            result['relative_energy_error'] = float(np.linalg.norm(np.array(result['scaled_effective_energy'])-exact)/np.linalg.norm(exact))
            rows.append(result)
        data['tests']['laminate_'+name] = {'exact_energy':exact.tolist(),'results':rows}
    for phase in ('p','Cu'):
        data['tests']['homogeneous_'+phase] = {method:solve(np.zeros((8,8),int),[MATERIALS[phase]],method) for method in ('fem','fv')}
    checkpoint()
    checker = np.array([[0,1],[1,0]])
    for contrast in (10,10000):
        rows=[]
        for k in (1,2,4,8,16,32,64,128):
            labels = np.kron(checker,np.ones((k,k),int))
            row={'k':k,'N':len(labels),'exact_sigma':float(np.sqrt(contrast))}
            for method in ('fem','fv'):
                r=solve(labels,[(1.,0.,1.),(float(contrast),0.,1.)],method)
                row[method]={'sigma':r['sigma_iso'],'relative_error':r['sigma_iso']/np.sqrt(contrast)-1,'seconds':r['seconds']}
            rows.append(row)
            print('checker',contrast,k,row['fem']['sigma'],row['fv']['sigma'],flush=True)
            data['tests']['checker_'+str(contrast)]=rows; checkpoint()
    # Coupled two-phase checkerboard has a matrix geometric mean exact solution.
    phases=[MATERIALS['p'],MATERIALS['Cu']]
    a,b=map(material_energy,phases)
    ar=fractional_matrix_power(a,.5); air=np.linalg.inv(ar)
    exact2=ar@fractional_matrix_power(air@b@air,.5)@ar
    rows=[]
    for k in (2,8,32,128):
        row={'k':k}
        for method in ('fem','fv'):
            r=solve(np.kron(checker,np.ones((k,k),int)),phases,method)
            r['relative_energy_error']=float(np.linalg.norm(np.array(r['scaled_effective_energy'])-np.kron(exact2,np.eye(2)))/np.linalg.norm(np.kron(exact2,np.eye(2))))
            row[method]=r
        rows.append(row)
        data['tests']['checker_coupled']={'exact_energy':np.kron(exact2,np.eye(2)).tolist(),'rows':rows};checkpoint()
    base=original_cell3()
    for name,phases in [('baseline',[PHASES[x] for x in ('p','n','weak')]),
                        ('metal',[MATERIALS[x] for x in ('p','n','Cu')])]:
        case={'base_labels':base.tolist(),'phases':phases,'fem':[],'fv':[],'enclosures':[]}
        data['fixed_geometry'][name]=case
        for k in (1,2,4,8,16):
            labels=np.kron(base,np.ones((k,k),int))
            for method in ('fem','fv'):
                r=solve(labels,phases,method);r['k']=k;case[method].append(r)
                print(name,k,method,'Sa',r['Sa'],'Sxx',r['Sxx'],'sigma',r['sigma_iso'],'sec',r['seconds'],flush=True)
                checkpoint()
            enc=energy_enclosure(labels,phases,case['fem'][-1]);enc['k']=k;case['enclosures'].append(enc)
            print('certificate',name,k,enc['Sa_interval_uV_K'],enc['Sxx_interval_uV_K'],flush=True)
            checkpoint()
        for method in ('fem','fv'):
            case[method+'_richardson']={key:richardson(case[method],key) for key in ('Sa','Sxx','sigma_iso')}
        checkpoint()
    phases=[PHASES[x] for x in ('p','n','weak')]
    labels=np.kron(base,np.ones((4,4),int))
    for name,g,ps in [('mirror',labels[::-1,:].copy(),phases),
                      ('translated',np.roll(labels,(7,11),(0,1)),phases),
                      ('two_phase',np.where(labels==2,0,labels),phases[:2]),
                      ('constant_S',labels,[(p[0],123e-6,p[2]) for p in phases])]:
        data['tests'][name]={m:solve(g,ps,m) for m in ('fem','fv')}
    data['source_sha256_after']=sources();checkpoint()


if __name__=='__main__':
    main()
