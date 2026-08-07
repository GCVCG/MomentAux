"""Report the multispectral EuroSAT sensor-fusion grid.

Three sources, identical tiles and identical subset indices, differing only
in which Sentinel-2 detectors they read:
    rgb  the 3 visible bands
    nir  the 10 non-visible bands (red edge, NIR, SWIR, cirrus, aerosol)
    all  both, i.e. sensor-fused input

crossed with {baseline, moment prior}. Two questions:

1. SENSOR FUSION. Is all > max(rgb, nir)? Under the classical taxonomy the
   visible and infrared bands should be COMPLEMENTARY (different physical
   information), so fusing them should compound. This is the multi-source
   version of our stack outcome, asked of the data rather than of a
   training intervention.

2. DOES THE PRIOR'S CURRENCY SURVIVE SENSOR FUSION? The moment prior
   supplies oriented-energy STRUCTURE, which is spatial and therefore in
   principle orthogonal to spectral coverage. If the currency account is
   right, its gain should persist on all three sources and should NOT be
   absorbed by adding bands. If instead the prior's gain shrinks once the
   extra bands are present, the two are substitutes and the account is
   wrong here.
"""
import argparse, json, os, statistics, itertools

BANDS = ("rgb", "nir", "all")
LABEL = {"rgb": "visible (3 bands)", "nir": "non-visible (10 bands)",
         "all": "sensor-fused (13 bands)"}


def acc(runs, cell):
    v = []
    d = os.path.join(runs, cell)
    if not os.path.isdir(d):
        return v
    for sd in sorted(os.listdir(d)):
        f = os.path.join(d, sd, "final.json")
        if os.path.isfile(f):
            try:
                v.append(100.0 * json.load(open(f))["final_test_acc"])
            except Exception:
                pass
    return v


def fmt(v):
    if not v:
        return "     --   "
    if len(v) == 1:
        return f"{v[0]:6.2f}    "
    return f"{statistics.fmean(v):6.2f}±{statistics.stdev(v)/len(v)**.5:4.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--pcts", default="5,10,25")
    a = ap.parse_args()
    for pct in [int(p) for p in a.pcts.split(",")]:
        print(f"\n=== EuroSAT multispectral, {pct}% of the training split ===")
        print(f"{'source':<24}{'baseline':>12}{'+ prior':>14}{'Δ prior':>10}")
        got = {}
        for b in BANDS:
            n = acc(a.runs, f"sf_eurosatms_{b}_none_{pct}pct")
            x = acc(a.runs, f"sf_eurosatms_{b}_aux_{pct}pct")
            got[b] = (n, x)
            d = (statistics.fmean(x) - statistics.fmean(n)) if n and x else None
            print(f"{LABEL[b]:<24}{fmt(n):>12}{fmt(x):>14}"
                  f"{('%+7.2f' % d) if d is not None else '     --':>10}")
        for arm, i in (("baseline", 0), ("prior", 1)):
            r, n_, al = (got['rgb'][i], got['nir'][i], got['all'][i])
            if r and n_ and al:
                best = max(statistics.fmean(r), statistics.fmean(n_))
                print(f"  sensor-fusion gain ({arm}): "
                      f"all {statistics.fmean(al):.2f} vs best single {best:.2f} "
                      f"= {statistics.fmean(al)-best:+.2f}")


if __name__ == "__main__":
    main()
