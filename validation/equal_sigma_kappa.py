"""Equal-sigma-kappa control (Supplementary Note 3): three phases with one product sigma*kappa give S_a = 0 in two dimensions.

The p4 pinwheel of Fig. 2b with its weak phase replaced by sigma = 1e3 S/m, S = 0, kappa = 100 W/(m K), so that all
three phases have sigma*kappa = 1e5; certified primal/dual enclosures at m = 2, 4, 8 as in sector_enclosures.py.
Output: equal_sigma_kappa.json.
"""

from pathlib import Path as _Path
import sys as _sys
RELEASE_ROOT = _Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(RELEASE_ROOT))
if __name__ == '__main__':
    import os as _os
    _os.chdir(RELEASE_ROOT)
import json
from certified_bounds.corner_sectors import sector_pair
from common.transport2d import ROOT, save, original_cell3, PHASES


def main():
    path=ROOT/'validation/equal_sigma_kappa.json'
    phases=[PHASES['p'],PHASES['n'],(1e3,0.,100.)]
    out=json.loads(path.read_text()) if path.exists() else {'phases':phases,'sigma_kappa':[p[0]*p[2] for p in phases],'rows':[]}
    for m in (2,4,8):
        if any(r['fem']['m']==m for r in out['rows']):continue
        r=sector_pair(original_cell3(),phases,m,m);out['rows'].append(r)
        print(m,'Sa',r['fem']['Sa'],'interval',r['enclosure']['Sa_interval_uV_K'],flush=True);save(path.relative_to(ROOT).as_posix(),out)


if __name__=='__main__':main()
