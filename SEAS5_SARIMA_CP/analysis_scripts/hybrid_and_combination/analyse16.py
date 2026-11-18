"""Notebook 16 (part B): compare the hybrid with its parts, choose the hybrid's base for each lead
on the development start months (2005-2012), confirm on 2013-2025, and test forecast combinations."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import nnls
HERE = Path(__file__).parent; PROC, RES = HERE / "proc", HERE / "res"
STATIONS = ["KIAMO-Accra", "Koforidua", "Kumasi", "Navrongo", "Tamale", "Wa"]; LEADS = range(1, 7); SEEDS = range(5)
DEV_END = 2012; B_BOOT, BLOCK = 1000, 12
K = ["station", "year", "month", "lead"]

R = pd.read_csv(PROC / "reference_forecasts.csv")[K + ["obs", "trend_clim", "sarima"]]
H = pd.read_csv(PROC / "hybrid_forecasts.csv"); N = pd.read_csv(PROC / "lstm_alone_and_trend_hybrid_forecasts.csv")
P = pd.read_csv(PROC / "postprocessed_forecasts.csv")[K + ["ridge"]]
X = pd.read_csv(PROC / "extra_baselines_forecasts.csv")
D = R.merge(H.drop(columns=["start", "valid_month"]), on=K).merge(N.drop(columns=["start", "valid_month"]), on=K).merge(P, on=K)
for fam in ["hyb", "hybs5", "lstm", "lstms5", "hybts5"]: D[fam] = D[[f"{fam}_s{s}" for s in SEEDS]].mean(axis=1)
X["gru_s5"] = X[[f"gru_s5_s{s}" for s in SEEDS]].mean(axis=1)
D = D.merge(X[K + ["svr", "gru_s5"]], on=K)
D = D.dropna(subset=["obs", "trend_clim", "sarima", "ridge", "svr", "hyb", "hybs5", "lstm", "lstms5", "hybts5", "gru_s5"]).reset_index(drop=True)
D["best_ref"] = np.where(D.lead <= 2, D.sarima, D.trend_clim)
D["dev"] = D.year <= DEV_END
print("Scored pairs:", len(D), "| development:", int(D.dev.sum()), "| confirmation:", int((~D.dev).sum()))

def rmse_net(g, c): return np.mean([np.sqrt(np.mean((s[c] - s.obs) ** 2)) for _, s in g.groupby("station")])
def mse(g, c): return float(np.mean((g[c] - g.obs) ** 2))

# ---- 1. Choose the hybrid's base for each lead on the development years ----
CH = {}
for L in LEADS:
    g = D[D.dev & (D.lead == L)]
    CH[L] = "hybs5" if mse(g, "hybs5") <= mse(g, "hybts5") else "hybts5"
D["newhyb"] = [r[CH[r.lead]] for _, r in D.iterrows()]
print("Base chosen on 2005-2012:", {L: ("SARIMA" if v == "hybs5" else "trend climatology") for L, v in CH.items()})

# ---- 2. Combinations: NNLS weights from earlier start years only ----
def combine(a, b, name):
    out = np.full(len(D), np.nan); W = []
    for L in LEADS:
        for Y in sorted(D.year.unique()):
            tr = D[(D.lead == L) & (D.year < Y)]; te = (D.lead == L) & (D.year == Y)
            if tr.year.nunique() < 3: w = np.array([0.5, 0.5])
            else:
                w, _ = nnls(tr[[a, b]].values, tr.obs.values); w = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])
            out[te.values] = D.loc[te, [a, b]].values @ w; W.append(dict(combo=name, lead=L, year=Y, w_a=w[0], w_b=w[1]))
    D[name] = out; return pd.DataFrame(W)
WTS = pd.concat([combine("newhyb", "sarima", "comb_sar"), combine("newhyb", "ridge", "comb_ridge")])

NAMES = {"trend_clim": "Trend climatology", "sarima": "Seasonal ARIMA", "best_ref": "Best reference", "lstm": "LSTM alone", "lstms5": "LSTM alone with SEAS5",
         "hyb": "Hybrid ARIMA-LSTM", "hybs5": "Hybrid with SEAS5", "hybts5": "Hybrid with SEAS5 on trend climatology", "newhyb": "Lead-aware hybrid with SEAS5",
         "comb_sar": "Hybrid + SARIMA combination", "comb_ridge": "Hybrid + ridge combination", "gru_s5": "ARIMA-GRU with SEAS5", "ridge": "Ridge with SEAS5", "svr": "SVR"}

# ---- 3. RMSE tables ----
rows = []
for per, sel in (("all", D.index == D.index), ("development", D.dev), ("confirmation", ~D.dev)):
    for L in LEADS:
        g = D[sel & (D.lead == L)]
        for c in NAMES: rows.append(dict(period=per, lead=L, key=c, model=NAMES[c], rmse=rmse_net(g, c), mse=mse(g, c), n=len(g)))
RM = pd.DataFrame(rows)
for per in ("all", "development", "confirmation"):
    print(f"\nRMSE ({per}), mean over stations"); print(RM[RM.period == per].pivot(index="model", columns="lead", values="rmse").loc[list(NAMES.values())].round(3).to_string())

# ---- 4. Tests ----
def dm_test(d, h, bw=12):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); dbar = d.mean(); lr = np.mean((d - dbar) ** 2)
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
D["start_key"] = D.year * 100 + D.month
def compare(a, b, sel, family):
    rng = np.random.default_rng(16); rr = []
    for L in LEADS:
        g = D[sel & (D.lead == L)]
        d = ((g[a] - g.obs) ** 2 - (g[b] - g.obs) ** 2).groupby(g.start_key).mean().sort_index()
        dbar, dm, p = dm_test(d.values, L); lo, hi = boot(g, a, b, rng)
        rr.append(dict(family=family, model=NAMES[a], against=NAMES[b], lead=L, skill=1 - mse(g, a) / mse(g, b), lo=lo, hi=hi, dm=dm, p=p))
    return rr
PLAN = [("newhyb", "sarima"), ("newhyb", "best_ref"), ("newhyb", "lstms5")]       # main family (18 tests)
EXTRA = [("newhyb", "lstm"), ("newhyb", "hybs5"), ("newhyb", "hyb"), ("hyb", "lstm"), ("hybs5", "lstms5"), ("lstms5", "best_ref"), ("lstm", "best_ref"),
         ("comb_sar", "best_ref"), ("comb_sar", "newhyb"), ("comb_ridge", "ridge"), ("comb_ridge", "best_ref"), ("ridge", "newhyb"), ("newhyb", "gru_s5")]
TESTS = []
for per, sel in (("all", D.index == D.index), ("confirmation", ~D.dev)):
    main = sum((compare(a, b, sel, "main") for a, b in PLAN), [])
    for r, ph in zip(main, holm([r["p"] for r in main])): r.update(period=per, p_holm=ph)
    TESTS += main
    for a, b in EXTRA:
        ex = compare(a, b, sel, "other")
        for r, ph in zip(ex, holm([r["p"] for r in ex])): r.update(period=per, p_holm=ph)
        TESTS += ex
T = pd.DataFrame(TESTS)
pd.set_option("display.width", 250)
for per in ("all", "confirmation"):
    print(f"\nTests ({per})")
    print(T[T.period == per][["family", "model", "against", "lead", "skill", "lo", "hi", "p", "p_holm"]].round(3).to_string(index=False))

# ---- 5. By station and season (main model vs best reference, vs LSTM alone with SEAS5) ----
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
D["vmonth"] = ((D.month - 1 + D.lead - 1) % 12) + 1; D["season"] = D.vmonth.map(SEASON)
ST = []
for st in STATIONS:
    for L in LEADS:
        g = D[(D.station == st) & (D.lead == L)]
        ST.append(dict(station=st, lead=L, skill_vs_best_ref=1 - mse(g, "newhyb") / mse(g, "best_ref"), skill_vs_lstms5=1 - mse(g, "newhyb") / mse(g, "lstms5"), skill_vs_sarima=1 - mse(g, "newhyb") / mse(g, "sarima")))
ST = pd.DataFrame(ST); print("\nLead-aware hybrid skill vs best reference by station"); print(ST.pivot(index="station", columns="lead", values="skill_vs_best_ref").round(3).to_string())
SE = []
for s in ["DJF", "MAM", "JJA", "SON"]:
    g = D[D.season == s]; SE.append(dict(season=s, skill_vs_best_ref=1 - mse(g, "newhyb") / mse(g, "best_ref"), skill_vs_lstms5=1 - mse(g, "newhyb") / mse(g, "lstms5")))
SE = pd.DataFrame(SE); print(SE.round(3).to_string(index=False))
print("\nMean combination weights on the hybrid:"); print(WTS[WTS.year >= 2008].groupby(["combo", "lead"]).w_a.mean().unstack().round(2).to_string())

RM.to_csv(RES / "nb16b_rmse_by_lead_model.csv", index=False); T.to_csv(RES / "nb16b_tests.csv", index=False)
ST.to_csv(RES / "nb16b_by_station.csv", index=False); SE.to_csv(RES / "nb16b_by_season.csv", index=False); WTS.to_csv(RES / "nb16b_combination_weights.csv", index=False)
json.dump({"scored_pairs": len(D), "dev_pairs": int(D.dev.sum()), "base_chosen": {str(L): ("sarima" if v == "hybs5" else "trend_clim") for L, v in CH.items()}},
          open(RES / "nb16b_summary.json", "w"), indent=2)
D.to_csv(PROC / "nb16b_scored.csv", index=False)
