"""Notebook 16 (part F): hybrid on ridge (stage 4) and combination of the strong models (stage 5)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import nnls
HERE = Path(__file__).parent; PROC, RES = HERE / "proc", HERE / "res"
STATIONS = ["KIAMO-Accra", "Koforidua", "Kumasi", "Navrongo", "Tamale", "Wa"]; LEADS = range(1, 7); SEEDS = range(5)
K = ["station", "year", "month", "lead"]; DEV_END = 2012; B_BOOT, BLOCK = 1000, 12

D = pd.read_csv(PROC / "nb16b_scored.csv")
N = pd.read_csv(PROC / "ridge_hybrid_forecasts.csv"); N["hybridge"] = N[[f"hybridge_s{s}" for s in SEEDS]].mean(axis=1)
D = D.merge(N[K + ["hybridge"]], on=K).dropna(subset=["hybridge"]).reset_index(drop=True)
D["dev"] = D.year <= DEV_END
print("Scored pairs:", len(D), "(Part B scored 8226)")
def mse(g, c): return float(np.mean((g[c] - g.obs) ** 2))
def rmse_net(g, c): return np.mean([np.sqrt(np.mean((s[c] - s.obs) ** 2)) for _, s in g.groupby("station")])

STRONG = ["ridge", "svr", "gru_s5", "newhyb", "hybridge"]
def combine(cols, name):
    out = np.full(len(D), np.nan)
    for L in LEADS:
        for Y in sorted(D.year.unique()):
            tr = D[(D.lead == L) & (D.year < Y)]; te = (D.lead == L) & (D.year == Y)
            if tr.year.nunique() < 3: w = np.ones(len(cols)) / len(cols)
            else:
                w, _ = nnls(tr[cols].values, tr.obs.values); w = w / w.sum() if w.sum() > 0 else np.ones(len(cols)) / len(cols)
            out[te.values] = D.loc[te, cols].values @ w
    D[name] = out
combine(STRONG, "comb_nnls"); D["comb_eq"] = D[STRONG].mean(axis=1)
# Choose between learned weights and equal weights on 2005-2012 only, for all leads together
dv = D[D.dev]; CHOICE = "comb_nnls" if mse(dv, "comb_nnls") < mse(dv, "comb_eq") else "comb_eq"
D["comb_strong"] = D[CHOICE]; print("Combination chosen on 2005-2012:", CHOICE)
NAMES = {"best_ref": "Best reference", "sarima": "Seasonal ARIMA", "lstms5": "LSTM alone with SEAS5", "newhyb": "Lead-aware hybrid with SEAS5",
         "hybridge": "Hybrid on ridge (LSTM corrects ridge)", "ridge": "Ridge with SEAS5", "svr": "SVR", "gru_s5": "ARIMA-GRU with SEAS5",
         "comb_eq": "Equal-weight combination of strong models", "comb_nnls": "Learned-weight combination of strong models", "comb_strong": "Chosen combination"}
rows = []
for per, sel in (("all", D.index == D.index), ("development", D.dev), ("confirmation", ~D.dev)):
    for L in LEADS:
        g = D[sel & (D.lead == L)]
        for c in NAMES: rows.append(dict(period=per, lead=L, key=c, model=NAMES[c], rmse=rmse_net(g, c), n=len(g)))
RM = pd.DataFrame(rows)
for per in ("all", "confirmation"): print(f"\nRMSE ({per})"); print(RM[RM.period == per].pivot(index="model", columns="lead", values="rmse").loc[list(NAMES.values())].round(3).to_string())

def dm_test(d, h, bw=12):
    d = np.asarray(d, float); n = len(d); dbar = d.mean(); lr = np.mean((d - dbar) ** 2)
    for l in range(1, bw + 1): lr += 2 * (1 - l / (bw + 1)) * np.mean((d[l:] - dbar) * (d[:-l] - dbar))
    dm = dbar / np.sqrt(max(lr, 1e-12) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dbar, dm, 2 * (1 - stats.t.cdf(abs(dm), n - 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty_like(p); m = 0
    for r, i in enumerate(o): m = max(m, (len(p) - r) * p[i]); out[i] = min(1.0, m)
    return out
def boot(g, a, b, rng):
    s = g.assign(ea=(g[a] - g.obs) ** 2, eb=(g[b] - g.obs) ** 2).groupby("start_key")[["ea", "eb"]].sum()
    n = len(s); nb = int(np.ceil(n / BLOCK)); ea, eb = s.ea.values, s.eb.values; out = []
    for _ in range(B_BOOT):
        st = rng.integers(0, n - BLOCK + 1, nb); ix = np.concatenate([np.arange(x, x + BLOCK) for x in st])[:n]
        out.append(1 - ea[ix].sum() / eb[ix].sum())
    return np.percentile(out, [2.5, 97.5])
def compare(a, b, sel, family):
    rng = np.random.default_rng(17); rr = []
    for L in LEADS:
        g = D[sel & (D.lead == L)]
        d = ((g[a] - g.obs) ** 2 - (g[b] - g.obs) ** 2).groupby(g.start_key).mean().sort_index()
        dbar, dm, p = dm_test(d.values, L); lo, hi = boot(g, a, b, rng)
        rr.append(dict(family=family, model=NAMES[a], against=NAMES[b], lead=L, skill=1 - mse(g, a) / mse(g, b), lo=lo, hi=hi, dm=dm, p=p))
    return rr
PLAN = [("hybridge", "ridge"), ("hybridge", "svr"), ("hybridge", "best_ref"), ("comb_strong", "ridge"), ("comb_strong", "svr"), ("comb_strong", "best_ref")]
EXTRA = [("hybridge", "lstms5"), ("hybridge", "newhyb"), ("hybridge", "sarima"), ("comb_strong", "hybridge"), ("svr", "ridge")]
TESTS = []
for per, sel in (("all", D.index == D.index), ("confirmation", ~D.dev)):
    main = sum((compare(a, b, sel, "main") for a, b in PLAN), [])
    for r, ph in zip(main, holm([r["p"] for r in main])): r.update(period=per, p_holm=ph)
    TESTS += main
    for a, b in EXTRA:
        ex = compare(a, b, sel, "other")
        for r, ph in zip(ex, holm([r["p"] for r in ex])): r.update(period=per, p_holm=ph)
        TESTS += ex
T = pd.DataFrame(TESTS); pd.set_option("display.width", 250)
for per in ("all", "confirmation"):
    print(f"\nTests ({per})"); print(T[T.period == per][["family", "model", "against", "lead", "skill", "lo", "hi", "p", "p_holm"]].round(3).to_string(index=False))
ST = pd.DataFrame([dict(station=st, lead=L, hybridge_vs_ridge=1 - mse(g, "hybridge") / mse(g, "ridge"), comb_vs_ridge=1 - mse(g, "comb_strong") / mse(g, "ridge"), comb_vs_best_ref=1 - mse(g, "comb_strong") / mse(g, "best_ref"))
                   for st in STATIONS for L in LEADS for g in [D[(D.station == st) & (D.lead == L)]]])
for c in ("hybridge_vs_ridge", "comb_vs_ridge", "comb_vs_best_ref"): print("\n" + c); print(ST.pivot(index="station", columns="lead", values=c).round(3).to_string())
RM.to_csv(RES / "nb18_rmse_by_lead_model.csv", index=False); T.to_csv(RES / "nb18_tests.csv", index=False); ST.to_csv(RES / "nb18_by_station.csv", index=False)
json.dump({"scored_pairs": len(D), "combination_chosen": CHOICE, "strong_members": STRONG}, open(RES / "nb18_summary.json", "w"), indent=2)
D.to_csv(PROC / "nb18_scored.csv", index=False)
