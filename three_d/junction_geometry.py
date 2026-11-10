"""Frozen neck/separation geometry and aspect-preserving RCSR runs."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import argparse
import hashlib
import itertools
import numpy as np
from three_d import fixed_rasters as R
from common import rectangular_voxels as C


def neck(base):
    proposals=np.zeros((3,)+base.shape,bool)
    for phase in (1,2):
        mask=base==phase
        for axis in range(3):
            b,c=[i for i in range(3) if i!=axis]
            m10=np.roll(mask,1,axis=b);m01=np.roll(mask,1,axis=c);m11=np.roll(m10,1,axis=c)
            bad=(mask&m11&~m10&~m01)|(~mask&~m11&m10&m01)
            points=np.argwhere(bad)
            for db,dc in itertools.product((0,-1),repeat=2):
                q=points.copy();q[:,b]+=db;q[:,c]+=dc;q%=np.array(base.shape)
                proposals[phase][tuple(q.T)]=True
        octants=np.array([np.roll(mask,shift,axis=(0,1,2)) for shift in itertools.product((0,1),repeat=3)])
        pattern=np.sum(octants.astype(np.uint16)*(2**np.arange(8,dtype=np.uint16))[:,None,None,None],axis=0)
        points=np.argwhere(np.isin(pattern,[2**i+2**(7-i) for i in range(4)]))
        for offset in itertools.product((0,-1),repeat=3):
            q=(points+offset)%np.array(base.shape);proposals[phase][tuple(q.T)]=True
    proposals&=(base==0)[None,:,:,:]
    conflicts=proposals[1]&proposals[2]
    # Phase priority is relevant only if two selected necks share matrix voxels.
    # It is invariant under the verified colour-preserving screw operations.
    result=base.copy();result[proposals[2]]=2;result[proposals[1]]=1
    return result,{'weak_voxels_selected_by_both_phases':int(conflicts.sum()),
                   'priority_if_conflict':'label 1',
                   'rule':'fill only weak voxels incident to each n/p edge-only or pure vertex-only contact; never replace existing n/p voxels',
                   'neck_one_side_width_fraction':1/base.shape[0]}


def prepare():
    original=R.read('three_d/fixed_rasters.json')
    out={'naming':{'W1':'fixed_raster W2','W2':'fixed_raster Wopt'},'source_sha256_before':R.hashes(),
         'fixed_geometry_file_sha256':hashlib.sha256((R.ROOT/'three_d/fixed_rasters.json').read_bytes()).hexdigest(),
         'materials_sigma_S_kappa':R.PHASES,'woodpile':{},'rcsr':{}}
    for name,oldname in [('W1','W2'),('W2','Wopt')]:
        base=np.array(original['cases'][oldname]['labels'],np.int32)
        for kind in ('fused','split'):
            if kind=='fused':lab,details=neck(base)
            else:lab,details=R.collar(base,1)
            contacts=R.contacts(lab);group=R.exact_group(lab)
            assert group['has_C4_z']
            key=name+'_'+kind
            out['woodpile'][key]={'parent_fixed_raster':oldname,'kind':kind,'N_base':base.shape[0],
                    'shape':list(lab.shape),'labels':lab.tolist(),'voxel_spacing_base':[1.,1.,1.],
                    'label_sha256':hashlib.sha256(lab.tobytes()).hexdigest(),
                    'geometry_edit':details,'changed_volume_fraction':float(np.mean(base!=lab)),
                    'contacts':contacts,'symmetry':group}
            print(key,'changed',np.mean(base!=lab),'conflicts',details.get('weak_voxels_selected_by_both_phases',0),
                  'contacts',[(p['label'],p['edge_only_segments_total'],p['pure_vertex_only_points']) for p in contacts['phases']],flush=True)
    for name in ('cyc','unm','bbr'):
        meta=original['cases'][name];lab=np.array(meta['labels'],np.int32)
        h=[1.,1.,1/meta['z_stretch_vs_cgd']]
        out['rcsr'][name]={'labels':lab.tolist(),'shape':list(lab.shape),'voxel_spacing_base':h,
                          'c_over_a':meta['intended_c_over_a'],'radius_over_a':meta['radius_over_a'],
                          'symmetry':meta['symmetry']}
    out['source_sha256_after']=R.hashes();assert out['source_sha256_before']==out['source_sha256_after']
    R.save('three_d/junction_geometry.json',out)


def run(category,levels,names=None):
    inputs=R.read('three_d/junction_geometry.json')
    filename='three_d/junction_'+category+'.json'
    out=R.read(filename) if (R.ROOT/filename).exists() else {'classification_key':{'C':'continuum variational enclosure, floating point','P':'uniform-grid point estimate'},'cases':{}}
    for name,meta in inputs[category].items():
        if names and name not in names:continue
        rec=out['cases'].setdefault(name,{'base_shape':meta['shape'],'voxel_spacing_base':meta['voxel_spacing_base'],'levels':[]})
        base=np.array(meta['labels'],np.int32)
        for k in levels:
            if any(x['k']==k for x in rec['levels']):continue
            lab=np.kron(base,np.ones((k,k,k),np.int32));h=np.array(meta['voxel_spacing_base'])/k
            print('START',name,'k',k,lab.shape,'spacing',h.tolist(),flush=True)
            result=C.pair(lab,R.PHASES,h);result['k']=k
            rec['levels'].append(result)
            R.save(filename,out)


def clearance():
    out=R.read('three_d/junction_geometry.json')
    for kind in ('fused','split'):
        parent='W2_'+kind;meta=out['woodpile'][parent]
        base=np.array(meta['labels'],np.int32);cut=np.zeros_like(base,bool)
        mask=base==0
        for axis in range(3):
            b,c=[i for i in range(3) if i!=axis]
            m10=np.roll(mask,1,axis=b);m01=np.roll(mask,1,axis=c);m11=np.roll(m10,1,axis=c)
            bad=(mask&m11&~m10&~m01)|(~mask&~m11&m10&m01)
            points=np.argwhere(bad)
            for db,dc in itertools.product((0,-1),repeat=2):
                q=points.copy();q[:,b]+=db;q[:,c]+=dc;q%=np.array(base.shape)
                cut[tuple(q.T)]=True
        lab=base.copy();lab[cut]=0
        diag=R.contacts(lab);group=R.exact_group(lab)
        assert group['has_C4_z']
        entry=dict(meta)
        entry.update(labels=lab.tolist(),label_sha256=hashlib.sha256(lab.tobytes()).hexdigest(),
                     contacts=diag,symmetry=group,parent_regularized_case=parent,
                     additional_clearance={'rule':'fill weak phase in all four voxels incident to each remaining weak/metal/weak/metal alternating edge; removes n/p line-only interphase contact',
                                           'changed_voxels':int(np.count_nonzero(lab!=base)),
                                           'volume_fraction_relative_to_parent':float(np.mean(lab!=base))})
        original=np.array(R.read('three_d/fixed_rasters.json')['cases']['Wopt']['labels'],np.int32)
        entry['changed_volume_fraction']=float(np.mean(lab!=original))
        out['woodpile'][parent+'_clear']=entry
        print(parent+'_clear','changed',entry['changed_volume_fraction'],'contacts',[(p['label'],p['edge_only_segments_total'],p['pure_vertex_only_points']) for p in diag['phases']],flush=True)
    R.save('three_d/junction_geometry.json',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','validate','woodpile','rcsr','clearance'])
    p.add_argument('--levels',nargs='+',type=int,default=[1,2,3]);p.add_argument('--cases',nargs='+')
    a=p.parse_args()
    with R.R.threadpool_limits(limits=2):
        if a.stage=='prepare':prepare()
        elif a.stage=='clearance':clearance()
        elif a.stage=='validate':C.validate()
        else:run(a.stage,a.levels,a.cases)
