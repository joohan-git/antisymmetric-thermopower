"""Equal-sigma-kappa control (Supplementary Note 3): three phases with one product sigma*kappa give S_a = 0 in two dimensions.

The p4 pinwheel of Fig. 2b with its weak phase replaced by sigma = 1e3 S/m, S = 0, kappa = 100 W/(m K), so that all
three phases have sigma*kappa = 1e5; certified primal/dual enclosures at m = 2, 4, 8 as in r6_sector_runs.py.
Output: r6_equal_sigma_kappa.json.
"""
import json
from r6_sector import sector_pair
from r6_core import ROOT,save,original_cell3,PHASES


def main():
    path=ROOT/'r6_equal_sigma_kappa.json'
    phases=[PHASES['p'],PHASES['n'],(1e3,0.,100.)]
    out=json.loads(path.read_text()) if path.exists() else {'phases':phases,'sigma_kappa':[p[0]*p[2] for p in phases],'rows':[]}
    for m in (2,4,8):
        if any(r['fem']['m']==m for r in out['rows']):continue
        r=sector_pair(original_cell3(),phases,m,m);out['rows'].append(r)
        print(m,'Sa',r['fem']['Sa'],'interval',r['enclosure']['Sa_interval_uV_K'],flush=True);save(path.name,out)


if __name__=='__main__':main()
