"""Notebook 16 (part G): learning curve. How does the hybrid's skill against ridge change as stations are added?"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; PROC, RES = HERE / "proc", HERE / "res"
D = pd.read_csv(PROC / "learning_curve_forecasts.csv").dropna(subset=["obs", "hyb", "ridge", "sarima", "trend"])
D["best_ref"] = np.where(D.lead <= 2, D.sarima, D.trend)
for c in ("hyb", "ridge", "best_ref"): D["e_" + c] = (D[c] - D.obs) ** 2
def skill(g, a, b): return 1 - g["e_" + a].mean() / g["e_" + b].mean()

# 1. Per subset, over all leads and by lead group
rows = []
for sub, g in D.groupby("subset"):
    for grp, sel in (("all leads", g.lead > 0), ("leads 1-2", g.lead <= 2), ("leads 3-6", g.lead >= 3)):
        h = g[sel]; rows.append(dict(subset=sub, k=g.k.iloc[0], leads=grp, hyb_vs_ridge=skill(h, "hyb", "ridge"),
                                     hyb_vs_ref=skill(h, "hyb", "best_ref"), ridge_vs_ref=skill(h, "ridge", "best_ref")))
S = pd.DataFrame(rows)
K = S.groupby(["leads", "k"])[["hyb_vs_ridge", "hyb_vs_ref", "ridge_vs_ref"]].agg(["mean", "min", "max"]).round(3)
print("Skill by number of stations (mean, min, max over station groups)"); print(K.to_string())

# 2. Same station, different group sizes: removes the effect of which stations are in a group
P = []
for (sub, st), g in D.groupby(["subset", "station"]):
    P.append(dict(subset=sub, k=g.k.iloc[0], station=st, hyb_vs_ridge=skill(g, "hyb", "ridge"), hyb_vs_ref=skill(g, "hyb", "best_ref")))
P = pd.DataFrame(P)
W = P.groupby(["station", "k"]).hyb_vs_ridge.mean().unstack().round(3)
print("\nHybrid vs ridge for each station, by the number of stations used in training"); print(W.to_string())
slopes = {st: float(np.polyfit(g.k, g.hyb_vs_ridge, 1)[0]) for st, g in P.groupby("station")}
print("\nChange in hybrid-vs-ridge skill per added station (slope):", {k: round(v, 4) for k, v in slopes.items()})
allslope = float(np.polyfit(P.k, P.hyb_vs_ridge, 1)[0]); print("Pooled slope:", round(allslope, 4))
# simple extrapolation to 10 and 23 stations (linear in k; only a rough guide)
cur = float(P[P.k == 6].hyb_vs_ridge.mean())
print(f"Hybrid vs ridge at 6 stations: {cur:.3f}; rough linear extrapolation: 10 stations {cur + 4 * allslope:.3f}, 23 stations {cur + 17 * allslope:.3f}")
# bootstrap the pooled slope over station groups
rng = np.random.default_rng(19); subs = P.subset.unique(); bs = []
for _ in range(1000):
    pick = rng.choice(subs, len(subs), replace=True); q = pd.concat([P[P.subset == s] for s in pick])
    bs.append(np.polyfit(q.k, q.hyb_vs_ridge, 1)[0])
lo, hi = np.percentile(bs, [2.5, 97.5]); print(f"95% interval of the pooled slope: {lo:.4f} to {hi:.4f}")
S.to_csv(RES / "nb19_learning_curve_by_subset.csv", index=False); P.to_csv(RES / "nb19_learning_curve_by_station.csv", index=False)
json.dump({"pooled_slope": allslope, "slope_ci": [float(lo), float(hi)], "slopes_by_station": slopes, "hyb_vs_ridge_at_6": cur}, open(RES / "nb19_summary.json", "w"), indent=2)
