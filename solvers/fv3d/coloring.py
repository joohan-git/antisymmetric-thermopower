"""Colourings of the tetragonal woodpile rod packing and their colour point groups.
Parent: 4 layers; layer0 rods along x at y=1/4,3/4; layer1 along y at x=1/4,3/4; layer2 along x at y=0,1/2; layer3 along y at x=0,1/2.
Each of the 8 rods is p (2) or n (1); the matrix is the third phase (0).
Colour group = point operations g of 4/mmm (axis permutations/sign flips that keep z as the stacking axis) for which
g(T) equals T up to a lattice translation. An axial vector component a_i survives g iff det(g) * g_ii... (computed below)."""
import numpy as np, itertools
def rods_cell(N, w, colors):
    nz = N//4; T = np.zeros((N, N, 4*nz), int); c = (np.arange(N)+0.5)/N
    centres = [(0, (0.25, 0.75)), (1, (0.25, 0.75)), (0, (0.0, 0.5)), (1, (0.0, 0.5))]   # (direction 0=x rods,1=y rods), positions
    k = 0
    for layer, (direc, pos) in enumerate(centres):
        for p in pos:
            d = np.abs(((c - p + 0.5) % 1.0) - 0.5) < w/2
            sl = T[:, :, layer*nz:(layer+1)*nz]
            if direc == 0: sl[:, d, :] = colors[k]      # rod along x: varies in y
            else: sl[d, :, :] = colors[k]                # rod along y: varies in x
            k += 1
    return T
# 16 point operations of 4/mmm as signed axis permutations keeping z: (perm of x,y) x (sign x, sign y) x (sign z)
OPS = []
for perm in [(0, 1), (1, 0)]:
    for sx, sy, sz in itertools.product([1, -1], repeat=3):
        R = np.zeros((3, 3), int); R[0, perm[0]] = sx; R[1, perm[1]] = sy; R[2, 2] = sz; OPS.append(R)
def apply(T, R):
    """apply signed axis permutation R to a periodic voxel array (about the origin voxel corner)"""
    perm = [int(np.nonzero(R[i])[0][0]) for i in range(3)]; sgn = [int(R[i, perm[i]]) for i in range(3)]
    A = np.transpose(T, perm)
    for ax, s in enumerate(sgn):
        if s < 0: A = np.flip(A, axis=ax)          # reflection about the cell corner: (i+0.5)/N -> 1-(i+0.5)/N
    return A
def equal_up_to_translation(A, B, steps):
    N0, N1, N2 = A.shape
    for tx in range(0, N0, steps[0]):
        for ty in range(0, N1, steps[1]):
            for tz in range(0, N2, steps[2]):
                if np.array_equal(np.roll(A, (tx, ty, tz), axis=(0, 1, 2)), B): return True
    return False
def colour_group(T):
    N = T.shape[0]; steps = (max(N//8, 1), max(N//8, 1), max(T.shape[2]//8, 1))
    if T.shape[0] != T.shape[1]: raise ValueError
    return [R for R in OPS if equal_up_to_translation(apply(T, R), T, steps)]
def axial_allowed(G):
    """components of a time-even axial vector invariant under all g: a' = det(g) g a"""
    P = sum(np.linalg.det(R)*R for R in G)/len(G)
    return [bool(abs(P[i, i]) > 1e-9) for i in range(3)], P
