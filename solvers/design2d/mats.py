import sys; sys.path.insert(0, '../fv2d')
import homog2 as H
# (sigma S/m, S V/K, kappa W/mK) approximate room-temperature values
MAT = {'pBiSbTe': (1e5, 200e-6, 1.2), 'nBiTeSe': (1e5, -180e-6, 1.3), 'Cu': (6e7, 1.8e-6, 400.), 'Ni': (1.4e7, -19e-6, 90.),
       'PEDOT': (1e5, 20e-6, 0.3), 'CPC': (1e3, 10e-6, 0.3), 'insul': (1e-6, 0.0, 0.2)}
def m(name): s, S, k = MAT[name]; return H.mloc(s, S, k)
