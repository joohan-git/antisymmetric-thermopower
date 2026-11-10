"""Winding law versus the matrix/strip thermal-conductivity ratio (point estimates; same strip, source and endpoints as devices/winding.py)."""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import sys; import json
sys.dont_write_bytecode = True
from devices.winding import run
out = {}
for key, kw in [('n2_k0.003_N1024', dict(n=2, ratio=.003, N=1024)), ('n2_k10_N1024', dict(n=2, ratio=10., N=1024)),
                ('n1_k0.003_N1024', dict(n=1, ratio=.003, N=1024)), ('n3_k0.003_N1024', dict(n=3, ratio=.003, N=1024))]:
    r = run(**kw); out[key] = {k: r[k] for k in ('longitudinal_corrected_uV', 'winding_prediction_uV', 'local_gradient_prediction_uV', 'ratio_to_winding_prediction', 'ratio_to_local_gradient_prediction', 'heat_input_W_per_m')}
    print(key, out[key], flush=True)
    json.dump(out, open(r'devices/winding_ratio.json', 'w'), indent=1)
