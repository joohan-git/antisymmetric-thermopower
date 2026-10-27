"""Combined tolerances and protected-junction random ensembles."""
import json
import numpy as np
from r6_core import ROOT,save,solve,inverse_phases,enclosure_from_pair
from r6_sector import sector_pair,sector_homogenize
from r7 import metrics
from r8_contact import junction_gap
from r8_tolerance import CERT
from r8_tolerance_audit import general_sa_interval
from r8_finite import POINT


def main():
    data=json.loads((ROOT/'r8_tolerance.json').read_text());audit=json.loads((ROOT/'r8_tolerance_audit.json').read_text())['cases'];ph=data['phases']
    path=ROOT/'r8_manufacturing.json';out=json.loads(path.read_text()) if path.exists() else {'joint':{},'protected_ensemble':{},'protected_certificates':{},
        'definition':'For joint tests apply fixed CPC square gap after rounding. For protected ensemble apply w=4% CPC square after boundary perturbation: isolation region is protected from the random boundary error.'}
    for w in (.02,.04):
        for r in (.01,.03,.05):
            key=f'w{w}_r{r}'
            if key in out['joint']:continue
            base=np.array(data['rounding'][str(r)]['labels']);a=junction_gap(base,w)
            if np.array_equal(a,base):result=audit['rounding_'+str(r)].copy();result['reused_identical_geometry']='rounding_'+str(r)
            else:result=sector_pair(a,ph,1,1);result['metrics']=metrics(result,ph)
            result.update(number_type=CERT,w_fraction=w,rounding_fraction=r,N_base=100);out['joint'][key]=result;save(path.name,out);print('joint',key,result['metrics']['Sa_uV_K'],flush=True)
    for key,r in data['ensemble'].items():
        if key in out['protected_ensemble']:continue
        a=junction_gap(np.array(r['labels']),.04) if r['symmetry_preserved'] else np.array(r['labels']).copy()
        if not r['symmetry_preserved']:
            # Same square patches, without imposing p4 on the rest of the specimen.
            template=junction_gap(np.ones_like(a),.04);a[template==0]=0
        row=solve(a,ph,'fv');row.update(number_type=POINT,symmetry_preserved=r['symmetry_preserved'],labels=a.tolist());out['protected_ensemble'][key]=row;save(path.name,out)
    for sym in (True,False):
        key,r=min([(k,r) for k,r in out['protected_ensemble'].items() if r['symmetry_preserved']==sym],key=lambda kr:kr[1]['Sa'])
        if key in out['protected_certificates']:continue
        a=np.array(r['labels'])
        if sym:result=sector_pair(a,ph,1,1);result['Sa_interval_uV_K']=result['enclosure']['Sa_interval_uV_K']
        else:
            p=sector_homogenize(a,ph,1,1);d=sector_homogenize(a,inverse_phases(ph),1,1);enc=enclosure_from_pair(p,d)
            result={'fem':p,'lower_energy':enc['lower_energy'],'upper_energy':enc['upper_energy'],'Sa_interval_uV_K':general_sa_interval(enc)}
        result['number_type']=CERT;out['protected_certificates'][key]=result;save(path.name,out);print('protected',key,result['Sa_interval_uV_K'],flush=True)


if __name__=='__main__':main()
