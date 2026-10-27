"""Quick consistency checks of the main claims (runs in well under a minute).

1. Two-phase p4 pinwheel cell: the axial Seebeck coefficient S_a vanishes.
2. Three-phase p4 pinwheel cell: S_a is non-zero.
3. Its mirror image reverses the sign of S_a exactly.
4. A p4mm (mirror-symmetric) three-phase cell gives S_a = 0.
5. Three-dimensional 4_1 helix: two phases give zero axial z component, three phases do not, and the mirror image reverses it.
6. Weak-contrast cubic term (analytic/c3.py) on the 24 x 24 pinwheel: the closed form equals the general contraction
   to 1e-12, and the geometry factor G vanishes for two phases and for the p4mm cell and reverses sign for the mirror image.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'solvers', 'fv2d'))
sys.path.insert(0, os.path.join(HERE, 'solvers', 'fv3d'))
sys.path.insert(0, os.path.join(HERE, 'analytic'))
import homog2 as H
import homog3d as H3
import c3 as C3


def pinwheel(N, alpha, period, nph):
    """Stripe pinwheel cell of Fig. 2: stripes at angle alpha in one quadrant, completed by 90-degree rotations."""
    h = N // 2
    x, y = np.meshgrid(np.arange(h) + 0.5, np.arange(h) + 0.5, indexing='ij')
    a = np.deg2rad(alpha)
    u = x * np.cos(a) + y * np.sin(a)
    q = (np.floor(u / (period / nph)) % nph).astype(int) + 1
    c = np.zeros((N, N), int); c[:h, :h] = q
    return c + np.rot90(c, 1) + np.rot90(c, 2) + np.rot90(c, 3) - 1


def sa2d(cell, mats):
    sig, L, S = H.homogenize(cell, mats)
    return 0.5 * (S[0, 1] - S[1, 0]) * 1e6


p, n, w = H.mloc(1e5, 200e-6, 1.0), H.mloc(1e5, -200e-6, 1.0), H.mloc(1e3, 0.0, 0.2)
two = sa2d(pinwheel(24, 15, 6, 2), [p, n])
three = sa2d(pinwheel(24, 15, 6, 3), [p, n, w])
mirror = sa2d(pinwheel(24, 15, 6, 3)[::-1, :], [p, n, w])
mm = sa2d(pinwheel(24, 45, 6, 3), [p, n, w])
print(f'2D two-phase p4     S_a = {two: .3e} uV/K')
print(f'2D three-phase p4   S_a = {three: .3f} uV/K (finite volumes on the 24 x 24 raster)')
print(f'2D mirror image     S_a = {mirror: .3f} uV/K')
print(f'2D three-phase p4mm S_a = {mm: .3e} uV/K')
assert abs(two) < 1e-9 and abs(mm) < 1e-9
assert abs(three) > 1.0 and abs(three + mirror) < 1e-9 * max(1.0, abs(three))

# 3D: 4_1 helix built from a layer without symmetry
N = 8
rng = np.random.default_rng(0)
layer3 = rng.integers(0, 3, (N, N)); layer2 = (layer3 > 0).astype(int)


def helix(layer, q=2):
    h = np.zeros((N, N, 4 * q), int)
    for k in range(4):
        h[:, :, k * q:(k + 1) * q] = np.rot90(layer, k)[:, :, None]
    return h


wm, nm, pm = H3.sloc(10.45, 10e-6, 0.90), H3.sloc(1e5, -180e-6, 1.3), H3.sloc(1e5, 200e-6, 1.2)
az = lambda T, mats: H3.axial(H3.homogenize3d(T, mats)[2])[2] * 1e6
a2 = az(helix(layer2), [nm, pm])
a3 = az(helix(layer3), [wm, nm, pm])
a3m = az(helix(layer3)[::-1, :, :], [wm, nm, pm])
print(f'3D two-phase helix    S_a,z = {a2: .3e} uV/K')
print(f'3D three-phase helix  S_a,z = {a3: .4f} uV/K')
print(f'3D mirror image       S_a,z = {a3m: .4f} uV/K')
assert abs(a2) < 1e-9 and abs(a3) > 1e-3 and abs(a3 + a3m) < 1e-8 * max(1.0, abs(a3))

# Weak-contrast cubic term: geometry factor G_abc and the closed form of the material factor
def geometry_factor(lab, phases):
    KX, KY, K2 = C3.freqs(lab.shape[0])
    return C3.G3([(lab == i).astype(float) for i in phases], KX, KY, K2)


s0 = np.array([[1.0, 0.3], [0.3, 0.8]])
D1 = np.array([[0.7, -0.2], [-0.2, 0.1]]); D2 = np.array([[-0.3, 0.5], [0.5, 0.9]])
G = geometry_factor(pinwheel(24, 15, 6, 3), (0, 1))
closed, series = C3.closed_form(s0, D1, D2, G), C3.series_c3(s0, [D1, D2], G)
Gm = geometry_factor(pinwheel(24, 15, 6, 3)[::-1, :], (0, 1))
Gmm = geometry_factor(pinwheel(24, 45, 6, 3), (0, 1))
G2 = geometry_factor(pinwheel(24, 15, 6, 2), (0,))
print(f'cubic term c3: closed form {closed: .6e}, contraction {series: .6e}')
print(f'geometry factor G_112 = {G[0, 0, 1]: .4e}; mirror {Gm[0, 0, 1]: .4e}; p4mm {np.abs(Gmm).max():.1e}; two phases {abs(G2[0, 0, 0]):.1e}')
assert abs(closed) > 1e-6 and abs(closed - series) < 1e-12 * abs(closed)
assert np.abs(G + Gm).max() < 1e-12 and np.abs(Gmm).max() < 1e-12 and abs(G2[0, 0, 0]) < 1e-12
print('All checks passed.')
