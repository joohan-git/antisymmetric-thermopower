"""Handed angular colouring of tetragonal RCSR nets.
Net: RCSR 3D cgd expanded with the space-group operations; struts from the EDGE_CENTER records (rcsr_net.py).
Colouring: strut with in-plane direction angle theta: p if (theta mod 90) in (0, 45), n if in (45, 90); vertical struts p; matrix = weak phase.
90-degree rotations keep the colouring; mirrors containing z and two-fold axes perpendicular to z swap p and n for struts off the
0 and 45 degree directions, so a net with such struts and a 4-fold axis has a ferroaxial colour group; nets without them are two-phase."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gemmi
CGD = os.environ.get('RCSR_CGD', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'rcsr3d_web.cgd'))   # RCSR 3D net file, see README
from rcsr_net import read_cgd, expand, edges   # nodes in origin choice 2, struts from the EDGE_CENTER records
def voxelize(net, segs, N, rad):
    a, b, c = net['cell'][:3]; Nz = max(4, int(round(N*c/a))); Nz += (-Nz) % 4
    g = (np.arange(N) + 0.5)/N; gz = (np.arange(Nz) + 0.5)/Nz
    X, Y, Z = np.meshgrid(g*a, g*b, gz*c, indexing='ij'); P = np.stack([X, Y, Z], -1)
    lab = np.zeros((N, N, Nz), int)              # 0 = weak matrix
    isp = np.zeros_like(lab, bool); isn = np.zeros_like(lab, bool)
    Lm = np.array([a, b, c])
    for p0, p1 in segs:
        A = p0*Lm; B = p1*Lm; d = B - A; dxy = d[:2]
        if np.linalg.norm(dxy) < 1e-6*np.linalg.norm(d): col = 'p'
        else:
            th = np.degrees(np.arctan2(dxy[1], dxy[0])) % 90.0
            col = 'p' if 1e-6 < th < 45 - 1e-6 else ('n' if th > 45 + 1e-6 else 'p')
        for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
            A2 = A + sh*Lm; dd = P - A2; t = np.clip((dd @ d)/(d @ d), 0, 1)
            dist = np.linalg.norm(dd - t[..., None]*d, axis=-1)
            m = dist < rad*a
            (isp if col == 'p' else isn)[m] = True
    lab[isn] = 1; lab[isp] = 2                   # p has priority (symmetric rule)
    return lab
