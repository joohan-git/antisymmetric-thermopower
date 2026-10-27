import sys, numpy as np
sys.path.insert(0, '.')
import homog2 as H
def cell3(N, alpha, period, nph=3):
    h = N//2; x, y = np.meshgrid(np.arange(h)+0.5, np.arange(h)+0.5, indexing='ij')
    a = np.deg2rad(alpha); u = x*np.cos(a) + y*np.sin(a)
    P = (np.floor(u/(period/nph)) % nph).astype(int) + 1
    C = np.zeros((N, N), int); C[:h, :h] = P
    return C + np.rot90(C, 1) + np.rot90(C, 2) + np.rot90(C, 3) - 1
A = H.mloc(1e5, 200e-6, 1.0); Cn = H.mloc(1e5, -200e-6, 1.0); D = H.mloc(1e3, 0.0, 0.2)
# patch homogenize to also return full Meff
import types
src = open('homog2.py').read().replace("return sig, L, np.linalg.solve(sig, L)", "return sig, L, np.linalg.solve(sig, L), Meff")
mod = types.ModuleType('h3'); exec(src, mod.__dict__)
for N in [48, 96, 144]:
    T = cell3(N, 15, period=N//4)
    sig, L, S, M = mod.homogenize(T, [A, Cn, D])
    CVT = M[0, :, 1, :]; CTV = M[1, :, 0, :]
    print(N, 'Sxx %.3f Santi %.3f uV/K' % (S[0,0]*1e6, 0.5*(S[0,1]-S[1,0])*1e6),
          ' Onsager max|C_TV^T - T0*C_VT|/|T0*C_VT| = %.2e' % (np.abs(CTV.T - 300*CVT).max()/np.abs(300*CVT).max()),
          ' sig sym %.1e' % (abs(sig[0,1]-sig[1,0])/sig[0,0]), flush=True)
# mirror image: transpose map = reflection across diagonal
T = cell3(96, 15, period=24); Tm = T[::-1, :]
for lab, tt in [('original', T), ('mirror', Tm)]:
    sig, L, S, M = mod.homogenize(tt, [A, Cn, D]); print(lab, 'Santi %.3f' % (0.5*(S[0,1]-S[1,0])*1e6))
