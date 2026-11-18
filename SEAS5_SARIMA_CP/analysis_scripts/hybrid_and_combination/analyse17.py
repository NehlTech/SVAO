"""Notebook 16 (part D): do the sea-temperature inputs help? Same rules as part B."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import nnls
HERE = Path(__file__).parent; PROC, RES = HERE / "proc", HERE / "res"
STATIONS = ["KIAMO-Accra", "Koforidua", "Kumasi", "Navrongo", "Tamale", "Wa"]; LEADS = range(1, 7); SEEDS = range(5)
K = ["station", "year", "month", "lead"]; DEV_END = 2012; B_BOOT, BLOCK = 1000, 12

D = pd.read_csv(PROC / "nb16b_scored.csv")
N = pd.read_csv(PROC / "lstm_hybrid_sst_forecasts.csv")
for fam in ["lstms5sst", "hybts5sst", "hybs5sst"]: N[fam] = N[[f"{fam}_s{s}" for s in SEEDS]].mean(axis=1)
D = D.merge(N[K + ["lstms5sst", "hybts5sst", "hybs5sst"]], on=K).dropna().reset_index(drop=True)
D["dev"] = D.year <= DEV_END
print("Scored pairs:", len(D), "(Part B scored 8226)")
def mse(g, c): return float(np.mean((g[c] - g.obs) ** 2))
def rmse_net(g, c): return np.mean([np.sqrt(np.mean((s[c] - s.obs) ** 2)) for _, s in g.groupby("station")])

CH = {L: ("hybs5sst" if mse(D[D.dev & (D.lead == L)], "hybs5sst") <= mse(D[D.dev & (D.lead == L)], "hybts5sst") else "hybts5sst") for L in LEADS}
D["newhyb_sst"] = [r[CH[r.lead]] for _, r in D.iterrows()]
print("Base chosen on 2005-2012:", {L: ("SARIMA" if v == "hybs5sst" else "trend climatology") for L, v in CH.items()})

def combine(a, b, name):
    out = np.full(len(D), np.nan)
    for L in LEADS:
        for Y in sorted(D.year.unique()):
            tr = D[(D.lead == L) & (D.year < Y)]; te = (D.lead == L) & (D.year == Y)
            if tr.year.nunique() < 3: w = np.array([0.5, 0.5])
            else:
                w, _ = nnls(tr[[a, b]].values, tr.obs.values); w = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])
            out[te.values] = D.loc[te, [a, b]].values @ w
    D[name] = out
combine("newhyb_sst", "ridge", "comb_ridge_sst")

NAMES = {"best_ref": "Best reference", "sarima": "Seasonal ARIMA", "lstms5": "LSTM alone with SEAS5", "lstms5sst": "LSTM alone with SEAS5 and sea temperature",
         "newhyb": "Lead-aware hybrid with SEAS5", "newhyb_sst": "Lead-aware hybrid with SEAS5 and sea temperature",
         "comb_ridge": "Hybrid + ridge combination", "comb_ridge_sst": "Hybrid (with sea temperature) + ridge combination", "ridge": "Ridge with SEAS5", "svr": "SVR"}
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
PLAN = [("newhyb_sst", "sarima"), ("newhyb_sst", "best_ref"), ("newhyb_sst", "lstms5sst")]
EXTRA = [("newhyb_sst", "newhyb"), ("lstms5sst", "lstms5"), ("newhyb_sst", "ridge"), ("newhyb_sst", "lstms5"), ("comb_ridge_sst", "best_ref"), ("comb_ridge_sst", "ridge"), ("comb_ridge_sst", "comb_ridge")]
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
ST = pd.DataFrame([dict(station=st, lead=L, skill_vs_best_ref=1 - mse(g, "newhyb_sst") / mse(g, "best_ref"), skill_vs_newhyb=1 - mse(g, "newhyb_sst") / mse(g, "newhyb"))
                   for st in STATIONS for L in LEADS for g in [D[(D.station == st) & (D.lead == L)]]])
print("\nBy station: skill of sea-temperature hybrid vs the same hybrid without it"); print(ST.pivot(index="station", columns="lead", values="skill_vs_newhyb").round(3).to_string())
RM.to_csv(RES / "nb17_rmse_by_lead_model.csv", index=False); T.to_csv(RES / "nb17_tests.csv", index=False); ST.to_csv(RES / "nb17_by_station.csv", index=False)
json.dump({"scored_pairs": len(D), "base_chosen": {str(L): ("sarima" if v == "hybs5sst" else "trend_clim") for L, v in CH.items()}}, open(RES / "nb17_summary.json", "w"), indent=2)
