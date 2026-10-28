"""Independent fixed-raster verification of segmented woodpiles and RCSR nets.

Numerical engines are the validated
layered independent Q1 FEM and FV-flux RT0 complementary trial construction.
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
from three_d import layered_cells as R
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import types
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = R.ROOT
MAIN = R.MAIN
PHASES = R.PHASES


def save(name,value):
    path=ROOT/name
    assert path.resolve().is_relative_to(ROOT)
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+chr(10)
    temp=path.with_name(path.name+'.'+str(time.time_ns())+'.tmp')
    temp.write_text(payload,encoding='utf-8')
    for attempt in range(15):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==14:raise
            time.sleep(.15*(attempt+1))


R.save=save


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def module(filename):
    path = MAIN/filename
    mod = types.ModuleType('fixed_raster_'+path.stem)
    mod.__file__ = str(path)
    exec(compile(path.read_text(encoding='utf-8-sig'),str(path),'exec'),mod.__dict__)
    return mod


W = module('three_d/woodpile.py')
RC = module('three_d/rcsr_color.py')
COL = module('three_d/coloring.py')


def hashes():
    names = ['three_d/woodpile.py','three_d/rcsr_color.py','three_d/rcsr_net.py','three_d/coloring.py','common/homog3d.py','common/homog3d_it.py',
             'three_d/verify_segwood.json','three_d/opt_segwood.json','three_d/scan_woodpile_N24.json','three_d/rcsr_radius.json']
    out = {n: hashlib.sha256((MAIN/n).read_bytes()).hexdigest() for n in names}
    cgd = Path(RC.CGD)
    out['rcsr3d_web.cgd (RCSR, not redistributed)'] = hashlib.sha256(cgd.read_bytes()).hexdigest() if cgd.is_file() else 'absent (needed only for net-derived structures)'
    return out


def main_data():
    out = {name: json.loads((MAIN/name).read_text(encoding='utf-8')) for name in
           ('three_d/verify_segwood.json','three_d/opt_segwood.json','three_d/scan_woodpile_N24.json','three_d/rcsr_radius.json')}
    path = MAIN/'three_d/converge3d.json'
    out['converge3d_status'] = 'missing; ignored'
    if path.exists():
        before = path.stat()
        try:
            payload = path.read_bytes()
            contents = json.loads(payload)
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns):
                out['three_d/converge3d.json'] = contents
                out['converge3d_sha256'] = hashlib.sha256(payload).hexdigest()
                out['converge3d_status'] = 'complete valid JSON, stable during read; snapshot only'
        except (OSError,ValueError):
            out['converge3d_status'] = 'incomplete or changing; ignored'
    return out


def exact_group(lab):
    """All translations on the finite periodic lattice, not a stride subset."""
    target = lab.astype(float)
    target_fft = np.fft.fftn(target)
    norm = float(np.sum(target*target))
    ops,translations = [],[]
    for rot in COL.OPS:
        transformed = COL.apply(lab,rot)
        corr = np.fft.ifftn(target_fft*np.conj(np.fft.fftn(transformed))).real
        candidates = np.argwhere(abs(corr-norm) < max(1e-7,1e-11*norm))
        for shift in candidates:
            if np.array_equal(np.roll(transformed,tuple(shift),axis=(0,1,2)),lab):
                ops.append(rot)
                translations.append(shift.tolist())
                break
    allowed,proj = COL.axial_allowed(ops)
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    has_c4 = any(np.array_equal(g,rotation) for g in ops)
    dets = [int(round(np.linalg.det(g))) for g in ops]
    name = 'unclassified subgroup of 4/mmm'
    if len(ops) == 8 and all(d == 1 for d in dets) and has_c4:
        name = '422'
    elif len(ops) == 4 and all(d == 1 for d in dets) and has_c4:
        name = '4'
    elif len(ops) == 8 and has_c4 and any(np.array_equal(g,-np.eye(3,dtype=int)) for g in ops):
        name = '4/m'
    return {'order': len(ops),'point_group': name,'operations': [g.tolist() for g in ops],
            'translations_voxels': translations,'axial_allowed': allowed,
            'axial_projector': proj.tolist(),'has_C4_z': has_c4,
            'search': 'all 16 signed tetragonal point operations, every voxel translation by FFT and exact equality'}


def half_offsets(order):
    return [d for d in itertools.product((-1,0,1),repeat=3)
            if 0 < sum(abs(x) for x in d) <= order and next(x for x in d if x) > 0]


def connectivity(mask,order):
    ids = np.full(mask.shape,-1,np.int32)
    ids[mask] = np.arange(mask.sum(),dtype=np.int32)
    if not mask.any():
        return {'components': 0,'largest_fraction_of_phase': 0.},ids
    rows,cols = [],[]
    for shift in half_offsets(order):
        other = np.roll(ids,shift,axis=(0,1,2))
        good = mask & (other >= 0)
        rows.append(ids[good]);cols.append(other[good])
    row,col = np.concatenate(rows),np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row),np.int8),(row,col)),shape=(mask.sum(),mask.sum())).tocsr()
    count,labels = connected_components(graph,directed=False)
    sizes = np.bincount(labels)
    labelmap = np.full(mask.shape,-1,np.int32);labelmap[mask] = labels
    return {'components': int(count),'largest_fraction_of_phase': float(sizes.max()/mask.sum())},labelmap


def contacts(lab):
    """Local non-face contacts, plus changes in global periodic connectivity.

    An edge-only contact is an alternating 2x2 cross-section around an edge.
    A pure vertex-only contact has two occupied opposite octants and no
    same-phase edge path within that vertex's eight incident voxels.
    Counts include one periodic representative per edge segment or vertex.
    """
    out = {'classification': 'exact combinatorial diagnosis of the fixed voxel geometry',
           'definitions': {'edge_only': 'opposite same-phase octants in a 2x2 edge cross-section; both face intermediates differ',
                           'vertex_only': 'same-phase opposite octants with no local face-or-edge path',
                           'global_connectivity': 'periodic 6(face),18(face+edge),26(face+edge+vertex) neighbors'},
           'phases': []}
    for phase in range(3):
        mask = lab == phase
        conns = {}
        for order,name in ((1,'6'),(2,'18'),(3,'26')):
            conns[name],unused = connectivity(mask,order)
        edges = []
        for axis in range(3):
            b,c = [i for i in range(3) if i != axis]
            m10 = np.roll(mask,1,axis=b)
            m01 = np.roll(mask,1,axis=c)
            m11 = np.roll(m10,1,axis=c)
            bad = (mask & m11 & ~m10 & ~m01) | (~mask & ~m11 & m10 & m01)
            edges.append(int(bad.sum()))
        octants = np.array([np.roll(mask,shift,axis=(0,1,2)) for shift in itertools.product((0,1),repeat=3)])
        pattern = np.sum(octants.astype(np.uint16)*(2**np.arange(8,dtype=np.uint16))[:,None,None,None],axis=0)
        vertex_patterns = [int(2**i+2**(7-i)) for i in range(4)]
        vertices = int(np.isin(pattern,vertex_patterns).sum())
        out['phases'].append({'label': phase,'voxel_count': int(mask.sum()),'fraction': float(mask.mean()),
                              'edge_only_segments_by_axis': edges,'edge_only_segments_total': sum(edges),
                              'pure_vertex_only_points': vertices,'connectivity': conns})
    return out


def prepare():
    refs = main_data()
    initial = hashes()
    # The second DESIGN is record index 4, not the second refinement record.
    distinct = []
    for rec in refs['three_d/verify_segwood.json']:
        if not any(rec['params'] == x['params'] for x in distinct):
            distinct.append(rec)
    w2 = distinct[1]
    opt = refs['three_d/opt_segwood.json']['best']
    straight = refs['three_d/scan_woodpile_N24.json'][0]
    items = {}
    for name,n,params,obj in [('W2',16,w2['params'],w2),('Wopt',24,opt['x'],opt)]:
        lab = W.seg_woodpile(n,n//4,params[0]%1,params[1]%1,*params[2:],tuple(obj['o1']),tuple(obj['o2']),0,obj['stack'])
        items[name] = (lab,{'N_base': n,'params': params,'o1':obj['o1'],'o2':obj['o2'],'stack':obj['stack']})
    n=24
    lab=W.woodpile(n,n//4,straight['c1'],straight['c2'],straight['w1'],straight['w2'],2,1,0)
    items['straight']=(lab,{'N_base':n,'source':straight,
                          'material_difference': 'selected label 0 uses weak=(10.45,10e-6,0.9012). Original scan first record uses CPC=(1e3,10e-6,0.3); its nonzero diagonal values are not comparison targets.'})
    nets={x['name']:x for x in RC.read_cgd()} if Path(RC.CGD).is_file() else {}
    if not nets: print('RCSR file absent: net-derived structures skipped (see README)', flush=True)
    for name,rad in [(n_,r_) for n_,r_ in [('cyc',.08),('unm',.12),('bbr',.08)] if n_ in nets]:
        net=nets[name];pts,cns=RC.expand(net);segments=RC.edges(net,pts,cns)
        lab=RC.voxelize(net,segments,16,rad)
        metadata={'N_base':16,'radius_over_a':rad,'net':{'name':net['name'],'group':net['group'],'cell':net['cell'],
                  'nodes':[(int(cn),xyz.tolist()) for cn,xyz in net['nodes']]},
                  'expanded_points':pts.tolist(),'coordination_numbers':cns.tolist(),
                  'segments_fractional':[[a.tolist(),b.tolist()] for a,b in segments],
                  'main_metric':'cubic voxels, as in main homog3d; the voxelized geometry is interpreted with z aspect Nz/N',
                  'intended_c_over_a':net['cell'][2]/net['cell'][0],
                  'solved_z_over_x':lab.shape[2]/lab.shape[0],
                  'z_stretch_vs_cgd':(lab.shape[2]/lab.shape[0])/(net['cell'][2]/net['cell'][0])}
        items[name]=(lab,metadata)
    out={'source_sha256_before':initial,'materials_sigma_S_kappa':PHASES,
         'classification_key':{'C':'continuum variational enclosure or exact structural identity; floating-point evaluation',
                               'P':'uniform-grid point estimate, not continuum certified'},'cases':{},
         'main_reference_snapshot':refs}
    for name,(lab,metadata) in items.items():
        group=exact_group(lab)
        main_group=COL.colour_group(lab)
        diagnosis=contacts(lab)
        metadata.update({'shape':list(lab.shape),'labels':lab.tolist(),
                         'labels_sha256':hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest(),
                         'symmetry':group,'main_colour_group_order':len(main_group),
                         'main_axial_allowed':COL.axial_allowed(main_group)[0],'contacts':diagnosis})
        out['cases'][name]=metadata
        print('prepared',name,lab.shape,group['point_group'],group['axial_allowed'],
              'contacts',[(p['label'],p['edge_only_segments_total'],p['pure_vertex_only_points'],
              [p['connectivity'][q]['components'] for q in ('6','18','26')]) for p in diagnosis['phases']],flush=True)
        if not group['has_C4_z']:
            raise RuntimeError(('Cannot apply p4 interval reduction',name))
    out['source_sha256_after']=hashes()
    assert initial==out['source_sha256_after']
    assert out['cases']['straight']['symmetry']['point_group']=='422'
    R.save('three_d/fixed_rasters.json',out)


def run(names,levels,mirror=False):
    inputs=read('three_d/fixed_rasters.json')
    path=ROOT/'three_d/fixed_raster_transport.json'
    out=read(path.relative_to(ROOT).as_posix()) if path.exists() else {'source_geometry_sha256':hashlib.sha256((ROOT/'three_d/fixed_rasters.json').read_bytes()).hexdigest(),
                                              'cases':{},'source_sha256_before':hashes()}
    for name in names:
        src=inputs['cases'][name]
        base=np.array(src['labels'],dtype=np.int32)
        key=name+('_mirror_x' if mirror else '')
        if mirror:base=base[::-1,:,:].copy()
        rec=out['cases'].setdefault(key,{'base_name':name,'mirror_x':mirror,'base_shape':list(base.shape),
                  'materials_sigma_S_kappa':PHASES,'levels':[]})
        for k in levels:
            if any(x['k']==k for x in rec['levels']):continue
            lab=np.kron(base,np.ones((k,k,k),np.int32))
            print('start',key,k,lab.shape,flush=True)
            fem=R.FE.homogenize(lab,PHASES)
            print('FEM',key,k,fem['Sa_z_uV_K'],'seconds',fem['seconds'],flush=True)
            fv=R.fv_with_complement(lab,PHASES)
            enc=R.enclosure(fv['continuum_energy_lower'],fem['scaled_energy'])
            rec['levels'].append({'k':k,'fem':fem,'fv':fv,'continuum_enclosure':enc})
            out['source_sha256_after']=hashes()
            assert out['source_sha256_before']==out['source_sha256_after']
            R.save(path.relative_to(ROOT).as_posix(),out)
            print('done',key,k,'FV',fv['Sa_z_uV_K'],'C',enc['Sa_z_interval_uV_K'],flush=True)


def collar(base,k):
    """Replace a one-fine-voxel collar around conductive non-face contacts.

    This is a NEW geometry for a sensitivity experiment, never a bound on
    the original geometry. Each side of the singular edge/vertex is cut by h.
    """
    lab=np.kron(base,np.ones((k,k,k),np.int32))
    cut=np.zeros(lab.shape,bool)
    for phase in (1,2):
        mask=base==phase
        for axis in range(3):
            b,c=[i for i in range(3) if i!=axis]
            m10=np.roll(mask,1,axis=b);m01=np.roll(mask,1,axis=c);m11=np.roll(m10,1,axis=c)
            bad=(mask&m11&~m10&~m01)|(~mask&~m11&m10&m01)
            points=np.argwhere(bad)*k
            for along,db,dc in itertools.product(range(k),(-1,0),(-1,0)):
                q=points.copy();q[:,axis]+=along;q[:,b]+=db;q[:,c]+=dc
                q=q%np.array(lab.shape)
                cut[tuple(q.T)]=True
        octants=np.array([np.roll(mask,shift,axis=(0,1,2)) for shift in itertools.product((0,1),repeat=3)])
        pattern=np.sum(octants.astype(np.uint16)*(2**np.arange(8,dtype=np.uint16))[:,None,None,None],axis=0)
        points=np.argwhere(np.isin(pattern,[2**i+2**(7-i) for i in range(4)]))*k
        for offset in itertools.product((-1,0),repeat=3):
            q=(points+offset)%np.array(lab.shape)
            cut[tuple(q.T)]=True
    old=lab.copy();lab[cut]=0
    return lab,{'fine_voxel_width_over_cell':1/lab.shape[0],
                'total_collar_width_over_cell':2/lab.shape[0],
                'replaced_voxel_count':int(np.count_nonzero(lab!=old)),
                'replaced_volume_fraction':float(np.mean(lab!=old)),
                'replaced_phase_counts':[int(np.count_nonzero((lab!=old)&(old==p))) for p in range(3)]}


def sensitivity(names):
    inputs=read('three_d/fixed_rasters.json')
    out={'classification':'P comparison between distinct geometries; individual C enclosures refer only to each specified geometry',
         'description':'one fine-voxel weak-phase collar on each side of every conductive edge-only and pure vertex-only contact',
         'cases':{}}
    for name in names:
        base=np.array(inputs['cases'][name]['labels'],dtype=np.int32)
        lab,info=collar(base,2)
        group=exact_group(lab)
        assert group['has_C4_z']
        print('sensitivity start',name,info,flush=True)
        fem=R.FE.homogenize(lab,PHASES)
        fv=R.fv_with_complement(lab,PHASES)
        enc=R.enclosure(fv['continuum_energy_lower'],fem['scaled_energy'])
        diag=contacts(lab)
        out['cases'][name]={'k_relative_to_base':2,'shape':list(lab.shape),'alteration':info,
                            'labels':lab.tolist(),'symmetry':group,'contacts_after':diag,
                            'fem':fem,'fv':fv,'continuum_enclosure':enc}
        R.save('three_d/raster_sensitivity.json',out)
        print('sensitivity done',name,'FEM',fem['Sa_z_uV_K'],'FV',fv['Sa_z_uV_K'],
              'C',enc['Sa_z_interval_uV_K'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','run','sensitivity'])
    parser.add_argument('--cases',nargs='+',default=['W2','Wopt','cyc','unm','bbr','straight'])
    parser.add_argument('--levels',nargs='+',type=int,default=[1,2,3])
    parser.add_argument('--mirror',action='store_true')
    args=parser.parse_args()
    with R.threadpool_limits(limits=2):
        if args.stage=='prepare':prepare()
        if args.stage=='run':run(args.cases,args.levels,args.mirror)
        if args.stage=='sensitivity':sensitivity(args.cases)
