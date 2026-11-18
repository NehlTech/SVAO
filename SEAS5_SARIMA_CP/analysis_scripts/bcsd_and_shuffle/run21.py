"""SEAS5-BCSD comparison for station mean temperature, same rules as the paper (S04 design, rolling origin, ridge tuned on earlier years)."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent; PROC, OUT = HERE / "proc", HERE / "out"
RAWS = HERE.parent / "t04" / "raw" / "seas5_station_forecasts.csv"
STATIONS = ["KIAMO-Accra", "Koforidua", "Kumasi", "Navrongo", "Tamale", "Wa"]; LEADS = list(range(1, 7)); N_MEM = 25
TEST_YEARS = list(range(2005, 2025)); TUNE_YEARS = [2005, 2010, 2015, 2020]; DEV_END = 2012
ALPHAS = list(np.logspace(-2, 4, 12)); FEATURE_START = pd.Timestamp("1997-01-01"); B_BOOT, BLOCK = 1000, 12
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}; SL = ["DJF", "MAM", "JJA", "SON"]
CELL = "near"

# ---------- observations and station-based features (as S04) ----------
IDX = pd.date_range("1993-01-01", "2026-06-01", freq="MS")
M = pd.read_csv(PROC / "gmet_monthly_clean.csv", parse_dates=["ym"])
TM = M.pivot(index="ym", columns="station", values="tmean").reindex(IDX)[STATIONS]
MONTH = IDX.month.values
DES = np.zeros((len(IDX), 13)); DES[np.arange(len(IDX)), MONTH - 1] = 1.0; DES[:, 12] = np.arange(len(IDX)) / 12.0
def trend_feats(y, k):
    h = y[:k + 1]; ok = ~np.isnan(h)
    if ok.sum() < 24: return None
    beta, *_ = np.linalg.lstsq(DES[:k + 1][ok], h[ok], rcond=None); res = h - DES[:k + 1] @ beta
    last3 = res[-3:]; last3 = last3[~np.isnan(last3)]
    return DES[k + 1:k + 7] @ beta, (0.0 if np.isnan(res[-1]) else float(res[-1])), (float(last3.mean()) if len(last3) else 0.0)
rows = []
for st in STATIONS:
    y = TM[st].values.astype(float)
    for start in pd.date_range(FEATURE_START, "2025-12-01", freq="MS"):
        k = IDX.get_loc(start) - 1; f = trend_feats(y, k)
        if f is None: continue
        for L in LEADS: rows.append(dict(station=st, year=start.year, month=start.month, lead=L, trend_clim=f[0][L - 1], e1=f[1], e3=f[2]))
FEAT = pd.DataFrame(rows)

# ---------- SEAS5 (ECMWF system 51, first 25 members) exactly as S04 ----------
S = pd.read_csv(RAWS); S = S[(S.system == 51) & (S.member < N_MEM)]
EN = S.groupby(["station", "year", "month", "lead"]).t2m_c.agg(ens_mean="mean", spread="std").reset_index().sort_values(["station", "month", "lead", "year"])
g = EN.groupby(["station", "month", "lead"])["ens_mean"]
EN["clim_s"] = g.transform(lambda s: s.shift(1).expanding().mean()); EN["n_clim"] = g.transform(lambda s: s.shift(1).expanding().count())
EN["a_s"] = np.where(EN.n_clim >= 4, EN.ens_mean - EN.clim_s, np.nan)
P = FEAT.merge(EN[["station", "year", "month", "lead", "ens_mean", "spread", "a_s"]], on=["station", "year", "month", "lead"], how="inner")

# ---------- BCSD ----------
B = pd.read_csv(HERE / "bcsd.csv.gz"); B = B[(B.cell == CELL) & B.lead.between(1, 6)][["station", "year", "month", "lead", "ens_mean", "ens_std", "n"]]
B = B.rename(columns={"ens_mean": "b_mean", "ens_std": "sp_b", "n": "n_b"}).sort_values(["station", "month", "lead", "year"])
gb = B.groupby(["station", "month", "lead"])["b_mean"]
B["clim_b"] = gb.transform(lambda s: s.shift(1).expanding().mean()); B["nb_clim"] = gb.transform(lambda s: s.shift(1).expanding().count())
B["a_b"] = np.where(B.nb_clim >= 4, B.b_mean - B.clim_b, np.nan)
P = P.merge(B[["station", "year", "month", "lead", "b_mean", "sp_b", "a_b", "n_b"]], on=["station", "year", "month", "lead"], how="left")
P["start"] = pd.to_datetime(dict(year=P.year, month=P.month, day=1)); P["valid_ts"] = (P.start.dt.to_period("M") + (P.lead - 1)).dt.to_timestamp()
ob = TM.rename_axis("valid_ts").reset_index().melt(id_vars="valid_ts", var_name="station", value_name="obs")
P = P.merge(ob, on=["station", "valid_ts"], how="left")
P["vm"] = P.valid_ts.dt.month; P["sinv"], P["cosv"] = np.sin(2 * np.pi * P.vm / 12), np.cos(2 * np.pi * P.vm / 12); P["season"] = P.vm.map(SEASON)
P["y_res"] = P.obs - P.trend_clim
for s_ in SL: P["seas_" + s_] = (P.season == s_).astype(float); P["as_" + s_] = P.a_s * P["seas_" + s_]; P["ab_" + s_] = P.a_b * P["seas_" + s_]
for st in STATIONS: P["st_" + st] = (P.station == st).astype(float)
P = P.sort_values(["station", "start", "lead"]).reset_index(drop=True)
R = pd.read_csv(PROC / "reference_forecasts.csv", parse_dates=["start"])
chk = P.merge(R[["station", "year", "month", "lead", "trend_clim"]].rename(columns={"trend_clim": "tc02"}), on=["station", "year", "month", "lead"])
assert np.allclose(chk.trend_clim, chk.tc02, atol=1e-6, equal_nan=True); print("trend climatology matches reference forecasts:", len(chk), "rows")

# ---------- independence check for the BCSD anomaly ----------
rng = np.random.default_rng(21); issues = []; n_chk = 0
Bfull = pd.read_csv(HERE / "bcsd.csv.gz"); Bfull = Bfull[Bfull.cell == CELL]
for st in STATIONS:
    for start in rng.choice(pd.date_range("2000-01-01", "2024-12-01", freq="MS"), size=4, replace=False):
        start = pd.Timestamp(start); sub = P[(P.station == st) & (P.start == start)].sort_values("lead")
        e = Bfull[(Bfull.station == st) & (Bfull.year < start.year) & (Bfull.month == start.month) & Bfull.lead.between(1, 6)].groupby("lead").ens_mean.agg(["mean", "count"])
        cur = Bfull[(Bfull.station == st) & (Bfull.year == start.year) & (Bfull.month == start.month)].set_index("lead").ens_mean.reindex(range(1, 7))
        a_b = (cur - e["mean"]).where(e["count"] >= 4).reindex(range(1, 7))
        if not np.allclose(a_b.values, sub.a_b.values, atol=1e-8, equal_nan=True): issues.append(f"{st} {start.date()}")
        n_chk += 1
assert not issues, issues; print("BCSD anomaly independence checks passed:", n_chk)

# ---------- ridge (S04 design) ----------
def cols_for(tag):
    if tag == "s": return ["a_s", "e1", "e3", "spread", "sinv", "cosv"] + ["st_" + s for s in STATIONS] + ["as_" + s for s in SL]
    if tag == "b": return ["a_b", "e1", "e3", "sp_b", "sinv", "cosv"] + ["st_" + s for s in STATIONS] + ["ab_" + s for s in SL]
    if tag == "sb": return cols_for("s") + ["a_b", "sp_b"] + ["ab_" + s for s in SL]
def fit(tr, cols, alpha):
    sc = StandardScaler().fit(tr[cols].values); m = Ridge(alpha=alpha).fit(sc.transform(tr[cols].values), tr.y_res.values); return m, sc
def predict(mod, te, cols): m, sc = mod; return m.predict(sc.transform(te[cols].values))
def run_ridge(tag, years):
    cols = cols_for(tag); need = ["obs", "trend_clim"] + [c for c in cols if c not in ("sinv", "cosv") and not c.startswith("st_") and not c.startswith("seas_")]
    ok = P[need].notna().all(axis=1); chosen = {}
    for Yt in TUNE_YEARS:
        cut, end = pd.Timestamp(f"{Yt - 4}-12-01"), pd.Timestamp(f"{Yt - 1}-12-01"); base = P[ok]
        tr, va = base[base.valid_ts <= cut], base[(base.valid_ts > cut) & (base.valid_ts <= end)]; sc_ = []
        for a in ALPHAS:
            s = 0.0
            for L in LEADS:
                t_, v_ = tr[tr.lead == L], va[va.lead == L]; s += float(np.sum((predict(fit(t_, cols, a), v_, cols) - v_.y_res.values) ** 2))
            sc_.append(s)
        chosen[Yt] = ALPHAS[int(np.argmin(sc_))]
    pred = pd.Series(np.nan, index=P.index)
    for Y in years:
        alpha = chosen[max(t for t in TUNE_YEARS if t <= Y)]; tr = P[ok & (P.valid_ts <= pd.Timestamp(f"{Y - 1}-12-01"))]
        for L in LEADS:
            t_ = tr[tr.lead == L]; te = P[ok & (P.year == Y) & (P.lead == L)]
            if len(te): pred.loc[te.index] = te.trend_clim.values + predict(fit(t_, cols, alpha), te, cols)
    return pred, chosen
ALPHA_LOG = {}
P["ridge_s"], ALPHA_LOG["s"] = run_ridge("s", TEST_YEARS)
PP = pd.read_csv(PROC / "postprocessed_forecasts.csv", parse_dates=["start"])[["station", "year", "month", "lead", "ridge"]]
c = P.merge(PP, on=["station", "year", "month", "lead"]); c = c[c.year <= 2024]
d = np.nanmax(np.abs(c.ridge_s - c.ridge)); print("Reproduction of the paper's ridge: rows", int(c.ridge_s.notna().sum()), "max abs diff", float(d)); assert d < 1e-8
P["ridge_b"], ALPHA_LOG["b"] = run_ridge("b", TEST_YEARS); P["ridge_sb"], ALPHA_LOG["sb"] = run_ridge("sb", TEST_YEARS); print("alphas", {k: {int(a): round(float(b), 3) for a, b in v.items()} for k, v in ALPHA_LOG.items()})

# ---------- scoring sample ----------
V = pd.read_csv(PROC / "seas5_verification_table.csv")[["station", "year", "month", "lead", "bc"]]
P = P.merge(V, on=["station", "year", "month", "lead"], how="left").merge(R[["station", "year", "month", "lead", "sarima"]], on=["station", "year", "month", "lead"], how="left")
P["tc"] = P.trend_clim; P["seas5_raw"] = P.ens_mean; P["bcsd_raw"] = P.b_mean; P["tc_a_s"] = P.trend_clim + P.a_s; P["tc_a_b"] = P.trend_clim + P.a_b; P["seas5_bc"] = P.bc
MODELS = ["tc", "sarima", "seas5_raw", "seas5_bc", "bcsd_raw", "tc_a_s", "tc_a_b", "ridge_s", "ridge_b", "ridge_sb"]
D = P[P.year.between(2005, 2024) & P.obs.notna()].dropna(subset=MODELS).copy(); D["start_key"] = D.year * 100 + D.month; D["dev"] = D.year <= DEV_END
print("Scored pairs:", len(D), "| development", int(D.dev.sum()), "| confirmation", int((~D.dev).sum()))
D[["station", "year", "month", "lead", "season", "start_key", "dev", "obs"] + MODELS].to_csv(OUT / "scored21.csv", index=False)

def sq(n, g): return (g[n] - g["obs"]) ** 2
def mse(g, n): return float(np.mean(sq(n, g)))
def rmse_net(g, n): return float(np.mean([np.sqrt(np.mean(sq(n, s))) for _, s in g.groupby("station")]))
def dm_test(d, h, bw=12):
    d = np.asarray(d, float); n = len(d); dbar = d.mean(); lr = np.mean((d - dbar) ** 2)
    for l in range(1, bw + 1): lr += 2 * (1 - l / (bw + 1)) * np.mean((d[l:] - dbar) * (d[:-l] - dbar))
    dm = dbar / np.sqrt(max(lr, 1e-12) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n); return dbar, dm, 2 * (1 - stats.t.cdf(abs(dm), n - 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty_like(p); m = 0
    for r, i in enumerate(o): m = max(m, (len(p) - r) * p[i]); out[i] = min(1.0, m)
    return out
def boot(g, a, b, rng):
    s = g.assign(ea=sq(a, g), eb=sq(b, g)).groupby("start_key")[["ea", "eb"]].sum(); n = len(s); nb = int(np.ceil(n / BLOCK)); ea, eb = s.ea.values, s.eb.values; o = []
    for _ in range(B_BOOT):
        st = rng.integers(0, n - BLOCK + 1, nb); ix = np.concatenate([np.arange(x, x + BLOCK) for x in st])[:n]; o.append(1 - ea[ix].sum() / eb[ix].sum())
    return np.percentile(o, [2.5, 97.5])
def compare(a, b, sel, fam):
    rng = np.random.default_rng(21); res = []
    for L in LEADS:
        g = D[sel & (D.lead == L)]; d = (sq(a, g) - sq(b, g)).groupby(g.start_key).mean().sort_index(); dbar, dm, p = dm_test(d.values, L); lo, hi = boot(g, a, b, rng)
        res.append(dict(family=fam, model=a, against=b, lead=L, skill=1 - mse(g, a) / mse(g, b), lo=lo, hi=hi, p=p))
    return res
RM = pd.DataFrame([dict(period=per, lead=L, model=n, rmse=rmse_net(D[sel & (D.lead == L)], n)) for per, sel in (("all", D.index == D.index), ("development", D.dev), ("confirmation", ~D.dev)) for L in LEADS for n in MODELS])
RM.to_csv(OUT / "rmse21.csv", index=False)
FAM = {"F1 ridge: BCSD vs SEAS5": [("ridge_b", "ridge_s"), ("ridge_b", "tc"), ("ridge_s", "tc")],
       "F2 raw forecasts": [("bcsd_raw", "seas5_bc"), ("bcsd_raw", "tc"), ("tc_a_b", "tc_a_s")],
       "F3 both together": [("ridge_sb", "ridge_s"), ("ridge_sb", "ridge_b")]}
T = []
for per, sel in (("all", D.index == D.index), ("confirmation", ~D.dev)):
    for fam, lst in FAM.items():
        res = sum((compare(a, b, sel, fam) for a, b in lst), [])
        for r, ph in zip(res, holm([r["p"] for r in res])): r.update(period=per, p_holm=ph)
        T += res
T = pd.DataFrame(T); T.to_csv(OUT / "tests21.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("\nRMSE by lead, all test years 2005-2024 (mean over stations)"); print(RM[RM.period == "all"].pivot(index="model", columns="lead", values="rmse").loc[MODELS].round(3).to_string())
print("\nRMSE by lead, confirmation 2013-2024"); print(RM[RM.period == "confirmation"].pivot(index="model", columns="lead", values="rmse").loc[MODELS].round(3).to_string())
for per in ("all", "confirmation"):
    print(f"\nTests ({per}); skill = 1 - MSE ratio of model vs 'against'"); print(T[T.period == per][["family", "model", "against", "lead", "skill", "lo", "hi", "p", "p_holm"]].round(3).to_string(index=False))
# by station: BCSD raw bias and ridge skill
rows = []
for st in STATIONS:
    g = D[D.station == st]; rows.append(dict(station=st, bias_bcsd_raw=float((g.bcsd_raw - g.obs).mean()), bias_seas5_raw=float((g.seas5_raw - g.obs).mean()),
        rmse_bcsd_raw=float(np.sqrt(mse(g, "bcsd_raw"))), rmse_seas5_bc=float(np.sqrt(mse(g, "seas5_bc"))), skill_ridge_s_vs_tc=1 - mse(g, "ridge_s") / mse(g, "tc"), skill_ridge_b_vs_tc=1 - mse(g, "ridge_b") / mse(g, "tc")))
BS = pd.DataFrame(rows); BS.to_csv(OUT / "by_station21.csv", index=False); print("\nBy station (all leads):"); print(BS.round(3).to_string(index=False))
