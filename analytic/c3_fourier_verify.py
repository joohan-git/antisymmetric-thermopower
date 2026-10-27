"""Independent weak-contrast calculation. Run with Python + NumPy + SciPy.

All conductivities use symmetric heat-flux scaling q/T0. Array axis 0 is x.
The geometry function is the reference pinwheel cell(24, 15, 6, 3) of solvers/fv2d/pinwheel.py.
Finite volumes use solvers/fv2d/homog2.py (read only). Results are written next to this script.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.fft import rfft2, irfft2, next_fast_len

ROOT = Path(__file__).resolve().parent
T0 = 300.


def cell(N=24, alpha=15, period=6, nph=3):
    h = N // 2
    x, y = np.meshgrid(np.arange(h)+.5, np.arange(h)+.5, indexing='ij')
    a = np.deg2rad(alpha)
    Pq = (np.floor((x*np.cos(a)+y*np.sin(a))/(period/nph)) % nph).astype(int)+1
    Cc = np.zeros((N, N), int)
    Cc[:h, :h] = Pq
    return Cc + np.rot90(Cc, 1) + np.rot90(Cc, 2) + np.rot90(Cc, 3) - 1


def smat(sigma, S, kappa):
    return np.array([[sigma, sigma*S], [sigma*S, kappa/T0+sigma*S*S]])


def projectors(N, nyquist='symmetric'):
    """Real, self-adjoint continuum Fourier projector with Gamma(0)=0.

    On even grids the cross multiplier at a Nyquist line is ambiguous. The
    symmetric convention averages the two equivalent signs; the cutoff option
    discards those lines. Both preserve reflection/C4 and converge to the same
    continuum operator. The symmetric Nyquist modes are contractions, not
    exact projections; that finite-grid distinction is reported explicitly.
    """
    x = (np.fft.fftfreq(N)*N)[:, None]
    y = (np.fft.rfftfreq(N)*N)[None, :]
    den = x*x + y*y
    den[0, 0] = 1
    xx, xy, yy = x*x/den, x*y/den, y*y/den
    if N % 2 == 0:
        xy[N//2, :] = 0
        xy[:, N//2] = 0
        if nyquist == 'cutoff':
            for a in (xx, xy, yy):
                a[N//2, :] = 0
                a[:, N//2] = 0
    return np.array([[xx, xy], [xy, yy]])


def geometry(labels, nyquist='symmetric'):
    """g[b] = <chi_b ax((P chi_0)(P chi_1))>, ax(A)=(A01-A10)/2.

    Evaluates the double Fourier sum with FFTs. Only the traceless entries of
    P chi are needed, keeping memory small. Fourier transforms use numpy/scipy
    normalization (forward unnormalized; inverse divided by grid size).
    """
    N = len(labels)
    P = projectors(N, nyquist)
    mult_d, mult_o = P[0, 0]-P[1, 1], P[0, 1]
    del P
    hat = rfft2(labels == 0)
    d0 = irfft2(hat*mult_d, s=(N, N))
    o0 = irfft2(hat*mult_o, s=(N, N))
    hat = rfft2(labels == 1)
    d1 = irfft2(hat*mult_d, s=(N, N))
    o1 = irfft2(hat*mult_o, s=(N, N))
    z = .5*(d0*o1-d1*o0)
    return np.array([z[labels == b].sum()/N**2 for b in range(3)])


def geometry_exact_fourier(labels, cutoff):
    """Exact pixel Fourier coefficients, symmetric cutoff, no product aliasing.

    F_a is truncated to |kx|,|ky|<=cutoff. Its quadratic product has bandwidth
    2*cutoff. A grid with N>4*cutoff exactly integrates its pairing with the
    Fourier series of chi_b. Thus this is the finite double sum in the report,
    including the pixel sinc factors, rather than collocation quadrature.
    """
    base = len(labels)
    N = next_fast_len(4*cutoff+1, real=True)
    x = np.rint(np.fft.fftfreq(N)*N).astype(int)[:, None]
    y = np.rint(np.fft.rfftfreq(N)*N).astype(int)[None, :]
    window = (np.abs(x)<=cutoff) & (np.abs(y)<=cutoff)
    pair_window = (np.abs(x)<=2*cutoff) & (np.abs(y)<=2*cutoff)
    box_factor = np.exp(-1j*np.pi*(x+y)/base)*np.sinc(x/base)*np.sinc(y/base)
    def chi_hat(a):
        f = np.fft.fft2(labels==a)/base**2
        return N**2*f[x % base, y % base]*box_factor
    P = projectors(N)
    mult_d, mult_o = (P[0, 0]-P[1, 1])*window, P[0, 1]*window
    del P
    h = chi_hat(0)
    d0 = irfft2(h*mult_d, s=(N, N))
    o0 = irfft2(h*mult_o, s=(N, N))
    h = chi_hat(1)
    d1 = irfft2(h*mult_d, s=(N, N))
    o1 = irfft2(h*mult_o, s=(N, N))
    z = .5*(d0*o1-d1*o0)
    del d0, o0, d1, o1
    g = [float(np.mean(z*irfft2(chi_hat(a)*pair_window, s=(N, N)))) for a in [0, 1]]
    return np.array([*g, -sum(g)]), N


def material_factor(s0, delta):
    r = np.linalg.inv(s0)
    A, B = delta[0]-delta[2], delta[1]-delta[2]
    t = np.array([np.trace(r@A), np.trace(r@B)])
    Q = A@r@B-B@r@A
    return t, Q


def predicted(s0, delta, g):
    t, Q = material_factor(s0, delta)
    c3 = Q[0, 1]*np.dot(t, g[:2])
    return dict(t=t.tolist(), Q=Q.tolist(), c3=float(c3),
                Sa3_uV_K=float(c3/s0[0, 0]*1e6))


def sqrt_spd(A):
    w, v = np.linalg.eigh(A)
    return (v*np.sqrt(w))@v.T


def fft_full(labels, s0, delta, eps, tol=2e-14, maxiter=1000):
    """Full fixed-point solution of E = Ebar - eps Gamma(delta E).

    Four unit loads are solved simultaneously until the local-field residual
    converges, without using any cubic coefficients. The linear residual and
    effective reciprocity / C4 errors are returned. Whiten field units first.
    """
    N = len(labels)
    W = sqrt_spd(s0)
    Wi = np.linalg.inv(W)
    D = np.einsum('ab,ibc,cd->iad', Wi, delta, Wi)[labels]
    P = projectors(N)
    Ebar = np.eye(4).reshape(2, 2, 4)
    E = np.broadcast_to(Ebar, (N, N, 2, 2, 4)).copy()
    def apply_gamma(F):
        h = rfft2(F, axes=(0, 1))
        out = np.einsum('ijxy,xyajb->xyaib', P, h, optimize=True)
        return irfft2(out, s=(N, N), axes=(0, 1))
    err = float('inf')
    for it in range(maxiter):
        DE = np.einsum('xyab,xybik->xyaik', D, E, optimize=True)
        new = Ebar-eps*apply_gamma(DE)
        err = float(np.max(np.abs(new-E)))
        E = new
        if err < tol:
            break
    else:
        raise RuntimeError(f'FFT did not converge: {eps}, {err}')
    DE = np.einsum('xyab,xybik->xyaik', D, E, optimize=True)
    Cw = np.eye(4)+eps*DE.mean(axis=(0, 1)).reshape(4, 4)
    W4 = np.kron(W, np.eye(2))
    C = W4@Cw@W4
    a = .5*(C[0, 0]+C[1, 1])
    c = .5*(C[0, 3]-C[1, 2])
    return dict(eps=float(eps), a=float(a), c=float(c), Sa_uV_K=float(c/a*1e6),
                c_over_eps3=float(c/eps**3), Sa_over_eps3_uV_K=float(c/a/eps**3*1e6),
                iterations=it+1, field_residual=err,
                reciprocity_error=float(np.max(np.abs(Cw-Cw.T))),
                C=C.tolist())


def reference_materials():
    s = np.array([smat(1e5, 200e-6, 1.), smat(1e5, -200e-6, 1.),
                  smat(1e3, 0., .2)])
    s0 = (s[0]+s[1])/2
    return s0, s-s0, s


def upsample(ph, k):
    return ph.repeat(k, axis=0).repeat(k, axis=1)


def save(name, obj):
    (ROOT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run_geometry():
    s0, delta, s = reference_materials()
    out = dict(s0=s0.tolist(), delta=delta.tolist(), s_actual=s.tolist(),
               labels='0=p, 1=n, 2=weak; first array index=x',
               fractions=(np.bincount(cell().ravel())/24**2).tolist(),
               geometry=[], checks={})
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        t0 = time.time()
        g = geometry(upsample(cell(), k))
        rec = dict(k=k, N=24*k, g=g.tolist(), gsum=float(g[:2].sum()),
                   **predicted(s0, delta, g), seconds=time.time()-t0)
        out['geometry'].append(rec)
        print('geometry', k, g, rec['Sa3_uV_K'], flush=True)
        save('geometry_numbers.json', out)
    for name, ph in [('mirror', cell()[::-1, :]), ('p4mm', cell(alpha=45)),
                     ('two_phase', cell(nph=2)), ('merged_weak_into_p', np.where(cell()==2, 0, cell()))]:
        g = geometry(upsample(ph, 16))
        out['checks'][name] = dict(g=g.tolist(), **predicted(s0, delta, g))
    out['nyquist_cutoff'] = []
    for k in [4, 8, 16, 32, 64]:
        g = geometry(upsample(cell(), k), 'cutoff')
        out['nyquist_cutoff'].append(dict(k=k, g=g.tolist(), **predicted(s0, delta, g)))
    save('geometry_numbers.json', out)


def run_fft():
    s0, delta, s = reference_materials()
    out = dict(runs=[], checks={})
    for k in [1, 4, 8]:
        ph = upsample(cell(), k)
        prediction = predicted(s0, delta, geometry(ph))
        for eps in [.2, .1, .05, .025, -.025, .0125, -.0125, .00625, -.00625]:
            rec = dict(k=k, prediction=prediction, **fft_full(ph, s0, delta, eps))
            out['runs'].append(rec)
            print('fft', k, eps, rec['c_over_eps3'], rec['field_residual'], flush=True)
            save('fft_numbers.json', out)
    for name, ph in [('mirror', cell()[::-1, :]), ('p4mm', cell(alpha=45)),
                     ('two_phase', cell(nph=2))]:
        out['checks'][name] = fft_full(upsample(ph, 4), s0, delta, .1)
    # An exact simultaneous-congruence relation: s22/s11 is constant.
    sp = s.copy()
    sp[2, 1, 1] = sp[2, 0, 0]*s0[1, 1]/s0[0, 0]
    dp = sp-s0
    out['special_linear_relation'] = dict(s=sp.tolist(),
        weak_kappa=float(T0*sp[2, 1, 1]),
        prediction=predicted(s0, dp, geometry(upsample(cell(), 8))),
        full=fft_full(upsample(cell(), 4), s0, dp, .2))
    # Equal reference-metric traces: cubic zero with noncommuting matrices.
    dtrace = np.array([[[.3, .2], [.2, -.3]], [[-.1, .25], [.25, .1]],
                       [[-.2, -.1], [-.1, .2]]])
    out['equal_trace_relation'] = dict(delta=dtrace.tolist(),
        prediction=predicted(np.eye(2), dtrace, geometry(upsample(cell(), 8))),
        runs=[fft_full(upsample(cell(), 4), np.eye(2), dtrace, e) for e in [.2, .1, .05]])
    # Dependent endpoints do not force the cubic coefficient of an arbitrary
    # reference interpolation to vanish: the intermediate phases may span 3D.
    send = np.array([np.diag([2., 1.]), np.diag([1., 3.]), np.eye(2)])
    sref = np.array([[1., .2], [.2, 1.]])
    out['outside_span_reference_counterexample'] = dict(s0=sref.tolist(),
        endpoints=send.tolist(), prediction=predicted(sref, send-sref, geometry(upsample(cell(), 8))),
        full=fft_full(upsample(cell(), 4), sref, send-sref, .05))
    save('fft_numbers.json', out)


def run_fv():
    source = ROOT.parent/'solvers/fv2d/homog2.py'
    # Execute the source in memory: importing must not write a __pycache__ in
    # the read-only original directory.
    import types
    H = types.ModuleType('original_homog2')
    exec(compile(source.read_text(), str(source), 'exec'), H.__dict__)
    s0, delta, _ = reference_materials()
    W = sqrt_spd(s0)
    Wi = np.linalg.inv(W)
    out = dict(source=source.relative_to(ROOT.parent).as_posix(), runs=[], checks={})
    for k in [1, 2, 4, 8, 16]:
        ph = upsample(cell(), k)
        eps_values = [.1, .05, .025, -.025, .0125, -.0125] if k<16 else [.025, -.025, .0125, -.0125]
        for eps in eps_values:
            # The FV routine accepts arbitrary matrix conductivities. Solve
            # the whitened symmetric problem for conditioning, then undo the
            # congruence. For the diagonal reference, blocks scale directly.
            mats = [Wi@(s0+eps*d)@Wi for d in delta]
            start = time.time()
            sig, L, _ = H.homogenize(ph, mats)
            a = float(s0[0, 0]*np.trace(sig)/2)
            c = float(W[0, 0]*W[1, 1]*(L[0, 1]-L[1, 0])/2)
            rec = dict(k=k, eps=eps, a=a, c=c, c_over_eps3=c/eps**3,
                       Sa_uV_K=c/a*1e6, Sa_over_eps3_uV_K=c/a/eps**3*1e6,
                       seconds=time.time()-start)
            out['runs'].append(rec)
            print('fv', k, eps, rec['c_over_eps3'], rec['seconds'], flush=True)
            save('fv_numbers.json', out)
    for name, ph in [('mirror', cell()[::-1, :]), ('p4mm', cell(alpha=45)),
                     ('two_phase', cell(nph=2))]:
        eps = .1
        mats = [Wi@(s0+eps*d)@Wi for d in delta]
        sig, L, _ = H.homogenize(upsample(ph, 4), mats)
        a = float(s0[0, 0]*np.trace(sig)/2)
        c = float(W[0, 0]*W[1, 1]*(L[0, 1]-L[1, 0])/2)
        out['checks'][name] = dict(a=a, c=c, c_over_eps3=c/eps**3, Sa_uV_K=c/a*1e6)
    save('fv_numbers.json', out)


def run_units():
    import types
    source = ROOT.parent/'solvers/fv2d/homog2.py'
    H = types.ModuleType('original_homog2')
    exec(compile(source.read_text(), str(source), 'exec'), H.__dict__)
    s0, delta, _ = reference_materials()
    eps = .1
    mats = s0+eps*delta
    raw = [H.mloc(m[0, 0], m[0, 1]/m[0, 0],
                  T0*(m[1, 1]-m[0, 1]**2/m[0, 0])) for m in mats]
    sig, L, S = H.homogenize(cell(), raw)
    W = sqrt_spd(s0)
    Wi = np.linalg.inv(W)
    _, Lw, _ = H.homogenize(cell(), [Wi@m@Wi for m in mats])
    c = float((L[0, 1]-L[1, 0])/2)
    cw = float(W[0, 0]*W[1, 1]*(Lw[0, 1]-Lw[1, 0])/2)
    out = dict(eps=eps, unscaled_c=c, whitened_c=cw, difference=c-cw,
               unscaled_Sa_uV_K=float((S[0, 1]-S[1, 0])*5e5),
               matrix_conversion_error=float(max(np.max(np.abs(u-np.diag([1, T0])@m))
                                                 for u, m in zip(raw, mats))))
    save('units_check.json', out)
    print(out)


def run_exact_fourier():
    s0, delta, _ = reference_materials()
    out = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        g, N = geometry_exact_fourier(cell(), 12*k)
        rec = dict(k=k, cutoff=12*k, quadrature_N=N, g=g.tolist(), **predicted(s0, delta, g))
        out.append(rec)
        print('exact Fourier', k, g, rec['Sa3_uV_K'], flush=True)
        save('exact_fourier_numbers.json', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['geometry', 'fft', 'fv', 'exact', 'units', 'all'])
    args = parser.parse_args()
    for mode, fn in [('geometry', run_geometry), ('fft', run_fft), ('fv', run_fv),
                     ('exact', run_exact_fourier), ('units', run_units)]:
        if args.mode in (mode, 'all'):
            fn()
