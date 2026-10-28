"""Variational identities, tensor traces and signed annulus formulas."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import ast
import hashlib
import json
import numpy as np
from scipy.linalg import eigh
from three_d import fixed_rasters as R
from common.transport2d import rel
from common.fem2d import pinwheel, material_energy
from certified_bounds.corner_sectors import angular_template


def linear_bounds(lower,upper,h):
    lower=np.asarray(lower);upper=np.asarray(upper)
    v,w=np.linalg.eigh((upper-lower+upper.T-lower.T)/2)
    root=(w*np.sqrt(np.maximum(v,0)))@w.T
    vals=np.linalg.eigvalsh(root@h@root)
    base=float(np.sum(h*lower.T))
    return [base+float(np.minimum(vals,0).sum()),base+float(np.maximum(vals,0).sum())]


def main():
    out={'number_types':{'C':'continuum energy enclosure in the stated ideal model; no interval arithmetic',
                         'P':'uniform-grid or single trial tensor estimate; numerical check value'},
         'trace_and_spectrum':{},'energy_interval_optimization':{},'RT0':{},'reference_data':{}}
    angles,mm,kk=angular_template(4)
    err=0.
    for i,t0 in enumerate(angles):
        t1=angles[i+1] if i+1<len(angles) else 2*np.pi
        mid=(t0+t1)/2;quad=int(np.floor(mid/(np.pi/2)))
        tangent=abs(np.cos(mid))>=abs(np.sin(mid))
        coord=lambda t:np.tan(t) if tangent else 1/np.tan(t)
        def ray_parameter(t):
            a=t-quad*np.pi/2
            return np.tan(a) if a<=np.pi/4+1e-12 else 2-1/np.tan(a)
        for fraction in np.linspace(.03,.97,17):
            t=t0+(t1-t0)*fraction
            f=(coord(t)-coord(t0))/(coord(t1)-coord(t0))
            g=(ray_parameter(t)-ray_parameter(t0))/(ray_parameter(t1)-ray_parameter(t0))
            err=max(err,abs(f-g))
    assert err<1e-12
    n=len(angles);mass=np.zeros((2*n,2*n));stiff=mass.copy()
    coeff=material_energy((1e5,200e-6,1.0))
    for i in range(n):
        ids=[2*i,2*i+1,2*((i+1)%n),2*((i+1)%n)+1]
        mass[np.ix_(ids,ids)]+=np.kron(mm[i],coeff)
        stiff[np.ix_(ids,ids)]+=np.kron(kk[i],coeff)
    values,vectors=eigh(stiff,mass)
    lam=np.sqrt(np.maximum(values,0))
    positive=lam[2:]
    multiplicity=int(np.count_nonzero(abs(positive-positive[0])<1e-7))
    assert multiplicity>=2
    out['trace_and_spectrum']={
        'shared_trace':'The curved outer Q1 element is Q1 in ray parameter t, not a straight chord. On the circle, t is affine in tan(theta) or cot(theta), exactly the coordinate used for the disk angular basis. Both field components share the same boundary DOFs; nodal equality therefore implies equality of the entire trace.',
        'trace_basis_max_sample_error_P':err,'disk_radius_fraction_of_pixel':.22,
        'angular_elements_for_m4':32,'outer_radial_elements_for_nr3':3,
        'outer_Gauss_order':10,'angular_Gauss_order':16,
        'angular_weak_problem':'integral Yprime^T s(theta) Zprime dtheta = lambda^2 integral Y^T s(theta) Z dtheta; periodic continuous traces; s Yprime continuous in the transmission sense',
        'inner_product':'(Y,Z)_s = integral Y^T s Z dtheta; s is the symmetric positive definite 2-by-2 field coefficient',
        'spectrum':'lambda^2 real nonnegative, complete mass-orthonormal eigenbasis; positive exponents need not be simple. Two constant field modes have lambda=0.',
        'log_terms':'Choose finite-energy (r/R)^lambda modes for lambda>0 and constants for lambda=0. Negative powers and the lambda=0 log(r) branch have infinite energy. Repeated positive self-adjoint eigenvalues are diagonalizable and do not force generalized log modes.',
        'homogeneous_first_positive_exponents_P':positive[:8].tolist(),
        'homogeneous_first_positive_multiplicity_P':multiplicity,
        'DtN':'Ktheta Y = Mtheta Y diag(lambda^2), Y^T Mtheta Y=I; Bcircle = Mtheta Y diag(lambda) Y^T Mtheta. No continuum eigenmode truncation: all discrete angular modes are retained.'}
    out['energy_interval_optimization']={
        'unknown':'symmetric C of size 2d by 2d, d=2 or 3; L <= C <= U in Loewner order',
        'symmetry':'p4 in 2D or C4 about z in 3D gives sigma_xy_plane=a I. Three-dimensional bounds are C4 group averaged; the objective is invariant, so optimization over all matrices between these bounds admits a symmetric optimizer.',
        'ordering':'[electric x,electric y,(electric z),thermal x,thermal y,(thermal z)], zero-based indices',
        'a':'(C00+C11)/2',
        'c':'(C0,d+1-C1,d)/2',
        'b':'(C0,d+C1,d+1)/2',
        'ratios':'Sa=beta*c/a; Sxx=beta*b/a; beta=200e-6 V/K. sigma_plane=1e5*a S/m.',
        'matrix_parametrization':'C=L+Dhalf X Dhalf; D=U-L; 0<=X<=I',
        'support_function':'F(H)=trace(H L)+sum positive_eigenvalues(Dhalf H Dhalf)',
        'upper_ratio':'unique zero F(H-t Ha)=0',
        'lower_ratio':'unique zero -F(-H+t Ha)=0',
        'H':'Hc: (0,d+1),(d+1,0)=+1/4; (1,d),(d,1)=-1/4. Hb: (0,d),(d,0),(1,d+1),(d+1,1)=+1/4. Ha00=Ha11=1/2.',
        'solver':'numpy.linalg.eigh/eigvalsh for roots/support; scipy.optimize.brentq after geometric bracket expansion. 3D xtol=1e-13; the earlier 2D code uses the scipy default. No generic nonlinear optimizer or entrywise inversion of L,U.'}
    out['RT0']={
        'face_density':'F_d=A_harmonic*(u_neighbor-u_cell+h_d G_d)/h_d',
        'FV_face_conductance':'A_harmonic * face_area/h_d = A_harmonic*voxel_volume/h_d^2',
        'interior_flux':'J_d(t)=(1-t)Fminus_d+t Fplus_d, t in [0,1] along coordinate d',
        'conformity':'normal trace is single valued; divergence=sum_d(Fplus_d-Fminus_d)/h_d is constant in each cell and zero after periodic Hodge projection',
        'projection':'solve discrete Laplacian psi=div F; replace F_d by F_d-(psi_neighbor-psi_cell)/h_d; Fourier symbol uses h_d^-2',
        'energy_per_direction':'(Fm^T s^-1 Fm+Fp^T s^-1 Fp)/3+(Fm^T s^-1 Fp+Fp^T s^-1 Fm)/6',
        'normalization':'six flux trials form mean-flux matrix Q; B=Q^-T <J^T s^-1 J> Q^-1; L=B^-1. Volume average includes each physical cell volume. For equal rectangular voxels it is an arithmetic mean.',
        'quadrature':'The 1/3 and 1/6 weights are exact integrals of linear RT0 products. Spatial isotropy gives no cross-direction inverse-coefficient terms.'}
    figures=R.ROOT
    text=(figures/'planar_cells/pinwheel.py').read_text(encoding='utf-8')
    tree=ast.parse(text);function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='cell')
    ns={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'pinwheel_readonly_cell','exec'),ns)
    figlabels=ns['cell'](24,15,6,3)
    prior=R.read('validation/planar_refinement.json')['fixed_geometry']['baseline']
    assert np.array_equal(figlabels,np.array(prior['base_labels']))
    assert np.array_equal(figlabels,pinwheel(24,15,3))
    summary=R.read('certified_bounds/planar_summary.json')['cases']['baseline']
    cert=R.read('certified_bounds/sector_enclosures.json')['cases']['baseline'][-1]
    design=R.read('certified_bounds/design_enclosures.json')['cases']['A_64']['sector'][-1]
    ring=R.read('devices/annulus_results.json');old=ring['periodic_cell']
    ri,ro,t=.004,.009,.001;mu0=4*np.pi*1e-7
    factor=mu0*(1/ri-1/ro)/(2*np.log(ro/ri))
    sigma=old['sigma'][0][0];sa=old['Santi_main_uV_K']*1e-6
    current=-t*sigma*sa;mag=factor*current
    hc=np.zeros((4,4));hc[0,3]=hc[3,0]=.25;hc[1,2]=hc[2,1]=-.25
    cint=linear_bounds(cert['enclosure']['lower_energy'],cert['enclosure']['upper_energy'],-t*20*hc)
    newpoint=-t*summary['sigma_fine']*summary['Sa_fine_uV_K']*1e-6
    out['reference_data']={
        'design_A64_winding_model_P':{'Sxx_uV_K':design['fem']['Sxx'],'Sa_uV_K':design['fem']['Sa'],
                    'sigma_S_m':design['fem']['sigma_iso'],'note':'design_enclosures upper trial energy tensor is fixed input to device spiral calculations, not the exact continuum tensor'},
        'design_A64_Sxx_C_uV_K':design['enclosure']['Sxx_interval_uV_K'],
        'Vcorr_definition':'deltaTterminal=T_end-T_start; Vraw=V_start-V_end; Vcorr=Vraw-Sxx*deltaTterminal',
        'Vcorr_path_prediction':'Vcorr=Sa*integral(T_y dx-T_x dy), locally open circuit. For CCW positive n and radial outward heat flow: n*Sa*qprime/kappa.',
        'Fig1b_baseline':{'source':'planar_cells/pinwheel.py, cell(24,15,6,3)',
                         'labels_equal_fem_and_fv':True,'sigma_C_S_m':summary['sigma_interval'],
                         'sigma_upper_trial_point_P_S_m':summary['sigma_fine'],'Sa_C_uV_K':summary['Sa_interval_uV_K'],
                         'Sa_upper_trial_point_P_uV_K':summary['Sa_fine_uV_K']},
        'Fig2_reference_grid':[24,24],
        'bilinear_ring_geometry':{'box_mm':[20,20],'inner_radius_mm':4,'outer_radius_mm':9,'thickness_mm':1,
            'meshing':'Cartesian Q1 squares kept when cell-center radius is strictly between ri and ro; exposed square faces define staircase inner/outer boundaries. Boundary nodes have fixed T; no curved boundary elements or projection to exact circles.',
            'finest_homogeneous_grid':[576,576]},
        'device_spiral_geometry':{'box_mm':[20,20],'inner_radius_mm':2,'outer_radius_mm':9.3,'strip_width_mm':.4,
            'meshing':'same cell-center inclusion and exposed Cartesian faces; bilinear interpolation of nodal fields for terminal voltage and path gradients'}}
    out['annulus_formula_check']={
        'classification':'P for bilinear tensor/numerical comparison; C only for the separately stated Fig1b continuum-tensor interval in the ideal homogeneous thin-sheet annulus',
        'convention':'Sa=(Sxy-Syx)/2; right-handed x,y,z; inner-hot DeltaT>0; CCW current and +z magnetic field positive',
        'signed_formulas':'I_CCW/DeltaT=-t*sigma*Sa; Bz/DeltaT=-mu0*t*sigma*Sa*(1/ri-1/ro)/(2*ln(ro/ri))',
        'bilinear_convention':'a_cross_z=-Sa; bilinear predicted() uses plus t*sigma*a_cross_z',
        'bilinear_sigma_P_S_m':sigma,'bilinear_Sa_P_uV_K':sa*1e6,'bilinear_a_cross_z_P_uV_K':-sa*1e6,
        'bilinear_formula_I_A_per_K':current,'bilinear_formula_B_nT_per_K':mag*1e9,
        'bilinear_saved_prediction_I_A_per_K':ring['homogeneous_prediction']['winding_current_A'],
        'bilinear_saved_prediction_B_nT_per_K':ring['homogeneous_prediction']['B_center_thin_sheet_T']*1e9,
        'bilinear_homogeneous_grid_I_relative_error':ring['cases']['homogeneous_validation']['winding_current_A']/current-1,
        'bilinear_homogeneous_grid_B_relative_error':ring['cases']['homogeneous_validation']['B_center_thin_sheet_T']/mag-1,
        'Fig1b_upper_tensor_point_I_A_per_K':newpoint,'Fig1b_upper_tensor_point_B_nT_per_K':newpoint*factor*1e9,
        'Fig1b_continuum_I_C_A_per_K':cint,'Fig1b_continuum_B_C_nT_per_K':(np.array(cint)*factor*1e9).tolist(),
        'warning':'bilinear uses the coarse 24-by-24 Q1 effective tensor. Its current and B are not the continuum predictions of the Fig1b tensor, despite identical material labels.'}
    paths=[R.ROOT/n for n in ('certified_bounds/corner_sectors.py','common/transport2d.py','three_d/layered_cells.py','devices/annulus_fem.py','devices/winding.py','common/finite_sample.py','certified_bounds/planar_summary.json','certified_bounds/sector_enclosures.json','devices/annulus_results.json','certified_bounds/design_enclosures.json')]
    paths+=[figures/'planar_cells/pinwheel.py']
    out['source_sha256']={rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    R.save('validation/variational_identities.json',out)
    print('trace error',err,'first eigenvalue multiplicity',multiplicity,flush=True)
    print(out['annulus_formula_check'],flush=True)


if __name__=='__main__':main()
