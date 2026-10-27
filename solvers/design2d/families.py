"""Continuous p4 three-phase geometry families, rasterised at N (cell = unit square, periodic).
Labels 0,1,2 index the material triple. Each family checks exact rot90 invariance after rasterisation."""
import numpy as np

def grid(N):
    x = (np.arange(N)+0.5)/N
    return np.meshgrid(x, x, indexing='ij')

def symmetrize_check(T):
    assert np.array_equal(np.rot90(T), T), 'rasterised map not p4'
    return T

def rot_quadrant(P):
    """Build p4 cell from top-left quadrant map P (h x h) by rotation about cell centre."""
    h = P.shape[0]; N = 2*h
    C = -np.ones((N, N), int); C[:h, :h] = P
    for k in range(1, 4):
        R = np.rot90(C, k); C = np.where(C < 0, R, C)
    return symmetrize_check(C)

def F1_stripes(N, alpha, w0, w1):
    """Quadrant filled with stripes at angle alpha; one period split into fractions w0,w1,(1-w0-w1)."""
    h = N//2; x, y = np.meshgrid((np.arange(h)+0.5)/h, (np.arange(h)+0.5)/h, indexing='ij')
    a = np.deg2rad(alpha); u = (x*np.cos(a) + y*np.sin(a)) * 4.0  # 4 periods across quadrant
    f = u - np.floor(u)
    P = np.where(f < w0, 0, np.where(f < w0+w1, 1, 2))
    return rot_quadrant(P)

def F2_rotsquares(N, theta, s):
    """Rotating squares on a square lattice (spacing a=0.5, 2x2 squares per cell), squares centred at (i*a, j*a).
    Parity (i+j)%2==0 squares rotated +theta -> phase 0 (e.g. p-type); parity 1 rotated -theta -> phase 1 (e.g. n-type);
    voids/matrix -> phase 2 (soft conductor). Colouring the two square sublattices differently breaks the p4g glide -> p4.
    theta=0 restores mirrors. 4-fold axis at cell centre (a +theta square)."""
    X, Y = grid(N); a = 0.5
    lab = np.full((N, N), 2)
    for i in range(-1, 4):
        for j in range(-1, 4):
            cx, cy = i*a, j*a
            par = (i+j) % 2
            th = np.deg2rad(theta) * (1 if par == 0 else -1)
            dx, dy = X-cx, Y-cy
            u = dx*np.cos(th) + dy*np.sin(th); v = -dx*np.sin(th) + dy*np.cos(th)
            inside = (np.abs(u) < s*a/2) & (np.abs(v) < s*a/2)
            lab[inside] = par
    return lab

def F3_pinwheel_tiling(N, r, split):
    """Basketweave/pinwheel tiling: central square (phase 2) of side r, surrounded by four 'arms'
    rotating around it; each arm split along its length at fraction `split` into phase 0 (inner) / 1 (outer).
    Built on quadrant with rotation; handedness from arm offset."""
    X, Y = grid(N); c = 0.5
    dx, dy = X-c, Y-c
    lab = np.full((N, N), -1)
    sq = (np.abs(dx) < r/2) & (np.abs(dy) < r/2)
    lab[sq] = 2
    # arm 0: rectangle x in [-r/2, 0.5], y in [r/2, 0.5] ... use polar-like pinwheel: for each point outside square,
    # assign arm by which side's half-plane it lies in with handed offset: arm k covers region rotated by 90k
    for k in range(4):
        th = np.pi/2*k
        u = dx*np.cos(th) + dy*np.sin(th); v = -dx*np.sin(th) + dy*np.cos(th)
        arm = (v >= r/2) & (u >= -r/2) & (u < 0.5 - r/2 + 1e-12) & (v < 0.5)
        dist = (u + r/2) / (1.0 - r)   # 0 at inner end, 1 at outer end (handed: arms run along +u)
        sel = arm & (lab < 0)
        lab[sel] = np.where(dist[sel] < split, 0, 1)
    lab[lab < 0] = 1
    return lab

def is_p4(T): return np.array_equal(np.rot90(T), T)
def is_mirror(T): return np.array_equal(T, T.T) or np.array_equal(T, T[::-1, :]) or np.array_equal(T, T[:, ::-1]) or np.array_equal(T, np.rot90(T.T, 2))
