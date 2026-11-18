"""Notebook 16 (part G): learning curve. Does the hybrid close the gap to ridge as stations are added?
Same data, inputs, tuning grid, training loop and rolling-origin protocol as Notebook 05."""
import json, time, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch, torch.nn as nn
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
PROC, RES, RAW = HERE / "proc", HERE / "res", HERE / "raw"

SEED = 0; np.random.seed(SEED); torch.manual_seed(SEED)
STATIONS = ["KIAMO-Accra", "Koforidua", "Kumasi", "Navrongo", "Tamale", "Wa"]
LEADS = list(range(1, 7)); N_MEM = 25; SEEDS = [0, 1, 2, 3, 4]
TEST_YEARS = list(range(2005, 2026)); TUNE_YEARS = [2005, 2010, 2015, 2020]
SAR_START, LAST_START = pd.Timestamp("1999-01-01"), pd.Timestamp("2025-12-01")
NN_CFGS = [dict(hid=h, wd=w) for h, w in itertools.product([8, 16, 32], [1e-4, 1e-3, 1e-2, 1e-1])]
EPOCHS, BATCH, LR = 40, 128, 3e-3

IDX = pd.date_range("1993-01-01", "2026-06-01", freq="MS")
M = pd.read_csv(PROC / "gmet_monthly_clean.csv", parse_dates=["ym"])
TM = M.pivot(index="ym", columns="station", values="tmean").reindex(IDX)[STATIONS]
ORD = json.loads((RES / "nb02_summary.json").read_text())["sarima_orders"]

# ---- SARIMA forecasts (same as Notebook 05) ----
IDX95 = IDX[24:]
def fit_sarimax(y, order, sorder):
    trend = "c" if (order[1] == 0 and sorder[1] == 0) else None
    return SARIMAX(y, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
def guard_ok(sar, y, k):
    ref = np.nanmean(y[max(0, k - 35):k + 1]); return bool(np.all(np.isfinite(sar)) and np.all(np.abs(sar - ref) < 8))
def sarima_station(st):
    y = TM[st].values.astype(float)[24:]; order, sorder = tuple(ORD[st]["order"]), tuple(ORD[st]["seasonal_order"])
    rows, res_fit = [], None
    for start in pd.date_range(SAR_START, LAST_START, freq="MS"):
        k = IDX95.get_loc(start) - 1
        if IDX95[k].month == 12 or res_fit is None:
            new = fit_sarimax(y[:k + 1], order, sorder); sar = np.asarray(new.forecast(6))
            if guard_ok(sar, y, k) or res_fit is None: res_fit = new
            else: sar = np.asarray(res_fit.apply(y[:k + 1], refit=False).forecast(6))
        else:
            sar = np.asarray(res_fit.apply(y[:k + 1], refit=False).forecast(6))
        for L in LEADS: rows.append(dict(station=st, start=start, lead=L, sarima=sar[L - 1]))
    return pd.DataFrame(rows)
t0 = time.time()
SAR = pd.concat(Parallel(n_jobs=-1)(delayed(sarima_station)(st) for st in STATIONS), ignore_index=True)
R = pd.read_csv(PROC / "reference_forecasts.csv", parse_dates=["start"])
SAR = SAR.merge(R[["station", "start", "lead", "sarima"]].rename(columns={"sarima": "s02"}), on=["station", "start", "lead"], how="left")
SAR["sarima"] = SAR.s02.fillna(SAR.sarima); SAR = SAR.drop(columns="s02")
print(f"SARIMA rebuilt in {(time.time()-t0)/60:.1f} min")

# ---- inputs (same as Notebook 05) ----
MONTH = IDX.month.values
DES = np.zeros((len(IDX), 13)); DES[np.arange(len(IDX)), MONTH - 1] = 1.0; DES[:, 12] = np.arange(len(IDX)) / 12.0
def trend_window(y, k):
    h = y[:k + 1]; ok = ~np.isnan(h)
    if ok.sum() < 24: return None
    beta, *_ = np.linalg.lstsq(DES[:k + 1][ok], h[ok], rcond=None)
    res = (h - DES[:k + 1] @ beta)[-12:]; miss = np.isnan(res).astype(float)
    return DES[k + 1:k + 7] @ beta, np.column_stack([np.nan_to_num(res), miss])
Sx = pd.read_csv(RAW / "seas5_station_forecasts.csv"); Sx = Sx[(Sx.system == 51) & (Sx.member < N_MEM)]
EN = Sx.groupby(["station", "year", "month", "lead"]).t2m_c.agg(ens_mean="mean", spread="std").reset_index().sort_values(["station", "month", "lead", "year"])
g_ = EN.groupby(["station", "month", "lead"])["ens_mean"]
EN["clim_s"] = g_.transform(lambda s: s.shift(1).expanding().mean()); EN["n_clim"] = g_.transform(lambda s: s.shift(1).expanding().count())
EN["a_s"] = np.where(EN.n_clim >= 4, EN.ens_mean - EN.clim_s, np.nan)
ENi = EN.set_index(["station", "year", "month", "lead"])
meta, SEQ, TC, SARM, YOBS, VIDX, VM, AS, SP = [], [], [], [], [], [], [], [], []
SARi = SAR.set_index(["station", "start", "lead"]).sarima
for st in STATIONS:
    y = TM[st].values.astype(float)
    for start in pd.date_range(SAR_START, LAST_START, freq="MS"):
        k = IDX.get_loc(start) - 1; f = trend_window(y, k)
        if f is None: continue
        tc, win = f; sar = np.array([SARi[(st, start, L)] for L in LEADS])
        meta.append((st, start, start.year, start.month)); SEQ.append(win); TC.append(tc); SARM.append(sar)
        YOBS.append(np.array([y[k + L] for L in LEADS])); VIDX.append(np.arange(k + 1, k + 7)); VM.append(MONTH[k + 1:k + 7])
        AS.append(np.array([ENi.a_s.get((st, start.year, start.month, L), np.nan) for L in LEADS]))
        SP.append(np.array([ENi.spread.get((st, start.year, start.month, L), np.nan) for L in LEADS]))
META = pd.DataFrame(meta, columns=["station", "start", "year", "month"])
SEQ, TC, SARM, YOBS, VIDX, VM, AS, SP = (np.array(a) for a in (SEQ, TC, SARM, YOBS, VIDX, VM, AS, SP))
STI = np.eye(6)[META.station.map({s: i for i, s in enumerate(STATIONS)}).values]
SM = np.column_stack([np.sin(2 * np.pi * META.month / 12), np.cos(2 * np.pi * META.month / 12)])
S5 = np.column_stack([np.nan_to_num(AS), np.nan_to_num(SP)])

# ---- Ridge forecasts as in Notebook 04, rebuilt out of sample from 1999 so the hybrid has training targets ----
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEAS_LIST = ["DJF", "MAM", "JJA", "SON"]
def trend_feats(y, k):
    h = y[:k + 1]; ok = ~np.isnan(h)
    if ok.sum() < 24: return None
    beta, *_ = np.linalg.lstsq(DES[:k + 1][ok], h[ok], rcond=None)
    res = h - DES[:k + 1] @ beta; last3 = res[-3:]; last3 = last3[~np.isnan(last3)]
    return DES[k + 1:k + 7] @ beta, (0.0 if np.isnan(res[-1]) else float(res[-1])), (float(last3.mean()) if len(last3) else 0.0)
rows = []
for st in STATIONS:
    y = TM[st].values.astype(float)
    for start in pd.date_range("1997-01-01", "2025-12-01", freq="MS"):
        k = IDX.get_loc(start) - 1; f = trend_feats(y, k)
        if f is None: continue
        for L in LEADS: rows.append(dict(station=st, year=start.year, month=start.month, lead=L, trend_clim=f[0][L - 1], e1=f[1], e3=f[2]))
PP = pd.DataFrame(rows).merge(EN[["station", "year", "month", "lead", "spread", "a_s"]], on=["station", "year", "month", "lead"], how="inner")
PP["start"] = pd.to_datetime(dict(year=PP.year, month=PP.month, day=1))
PP["valid_ts"] = (PP.start.dt.to_period("M") + (PP.lead - 1)).dt.to_timestamp()
ob = TM.rename_axis("valid_ts").reset_index().melt(id_vars="valid_ts", var_name="station", value_name="obs")
PP = PP.merge(ob, on=["station", "valid_ts"], how="left"); PP["vm"] = PP.valid_ts.dt.month
PP["sinv"], PP["cosv"] = np.sin(2 * np.pi * PP.vm / 12), np.cos(2 * np.pi * PP.vm / 12); PP["season"] = PP.vm.map(SEASON); PP["y_res"] = PP.obs - PP.trend_clim
for s_ in SEAS_LIST: PP["seas_" + s_] = (PP.season == s_).astype(float); PP["as_" + s_] = PP.a_s * PP["seas_" + s_]
for st in STATIONS: PP["st_" + st] = (PP.station == st).astype(float)
FULL = ["a_s", "e1", "e3", "spread", "sinv", "cosv"] + ["st_" + s for s in STATIONS] + ["as_" + s for s in SEAS_LIST]
NB04 = json.loads((RES / "nb04_summary.json").read_text())["tuned_settings"]
ALPHA = {int(k): v["ridge_alpha"] for k, v in NB04.items()}
PP["ridge_r"] = np.nan
for Y in range(1999, 2026):
    a = ALPHA[max([t for t in ALPHA if t <= Y] or [min(ALPHA)])]
    tr = PP[(PP.valid_ts <= pd.Timestamp(f"{Y - 1}-12-01")) & PP.obs.notna() & PP.a_s.notna()]
    for L in LEADS:
        t_ = tr[tr.lead == L]; ix = PP.index[(PP.year == Y) & (PP.lead == L) & PP.a_s.notna()]
        if len(t_) < 30 or len(ix) == 0: continue
        sc = StandardScaler().fit(t_[FULL].values); m = Ridge(alpha=a).fit(sc.transform(t_[FULL].values), t_.y_res.values)
        PP.loc[ix, "ridge_r"] = PP.loc[ix, "trend_clim"].values + m.predict(sc.transform(PP.loc[ix, FULL].values))
Pc = pd.read_csv(PROC / "postprocessed_forecasts.csv")[["station", "year", "month", "lead", "ridge"]]
chk = PP.merge(Pc, on=["station", "year", "month", "lead"]).dropna(subset=["ridge", "ridge_r"])
print("Ridge rebuilt vs Notebook 04 (2005-2025):", len(chk), "rows, max abs diff", float(np.abs(chk.ridge - chk.ridge_r).max()))
PP = PP.merge(Pc, on=["station", "year", "month", "lead"], how="left"); PP["ridge_use"] = PP.ridge.fillna(PP.ridge_r)
RIi = PP.set_index(["station", "year", "month", "lead"]).ridge_use
RIDGE = np.array([[RIi.get((st, s.year, s.month, L), np.nan) for L in LEADS] for st, s in zip(META.station, META.start)])
print("Ridge base available for", int(np.isfinite(RIDGE).all(axis=1).sum()), "of", len(META), "samples")

import itertools as it
BASEM = np.column_stack([SARM[:, :2], TC[:, 2:]])          # lead-aware base: SARIMA at leads 1-2, trend climatology at 3-6
STAT = np.column_stack([STI, SARM - TC, SM, S5]); YT = YOBS - BASEM
HID, WD, LC_SEEDS = 16, 0.1, [0, 1, 2]
class Net(nn.Module):
    def __init__(self, n_stat, hid):
        super().__init__(); self.lstm = nn.LSTM(2, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + n_stat, 32), nn.ReLU(), nn.Linear(32, 6))
    def forward(self, seq, stat):
        _, (h, _) = self.lstm(seq); return self.head(torch.cat([h[-1], stat], 1))
def hyb_fit_pred(smask, cut_idx, pred_idx, seed):
    torch.set_num_threads(1)
    M_ = np.isfinite(YT) & (VIDX <= cut_idx) & smask[:, None]; rows = np.where(M_.any(axis=1))[0]
    sdY = float(np.std(YT[M_])); seq_sd = float(np.std(SEQ[rows][:, :, 0])); mu, sg = STAT[rows].mean(0), STAT[rows].std(0) + 1e-6
    def prep(ix):
        q = SEQ[ix].copy(); q[:, :, 0] /= seq_sd
        return torch.tensor(q, dtype=torch.float32), torch.tensor((STAT[ix] - mu) / sg, dtype=torch.float32)
    xs, xt = prep(rows); yt = torch.tensor(np.nan_to_num(YT[rows]) / sdY, dtype=torch.float32); mt = torch.tensor(M_[rows], dtype=torch.float32)
    torch.manual_seed(seed); net = Net(STAT.shape[1], HID); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(rows), generator=gen)
        for b in range(0, len(rows), BATCH):
            ix = perm[b:b + BATCH]; opt.zero_grad()
            ((net(xs[ix], xt[ix]) - yt[ix]) ** 2 * mt[ix]).sum().div(mt[ix].sum().clamp(min=1e-6)).backward(); opt.step()
    net.eval(); ps, pt = prep(pred_idx)
    with torch.no_grad(): return BASEM[pred_idx] + net(ps, pt).numpy() * sdY

def ridge_pred(sub, Y):
    a = ALPHA[max([t for t in ALPHA if t <= Y] or [min(ALPHA)])]
    tr = PP[PP.station.isin(sub) & (PP.valid_ts <= pd.Timestamp(f"{Y - 1}-12-01")) & PP.obs.notna() & PP.a_s.notna()]
    out = {}
    for L in LEADS:
        t_ = tr[tr.lead == L]; te = PP[PP.station.isin(sub) & (PP.year == Y) & (PP.lead == L) & PP.a_s.notna()]
        sc = StandardScaler().fit(t_[FULL].values); m = Ridge(alpha=a).fit(sc.transform(t_[FULL].values), t_.y_res.values)
        for (st, yy, mm), v in zip(zip(te.station, te.year, te.month), te.trend_clim.values + m.predict(sc.transform(te[FULL].values))): out[(st, yy, mm, L)] = v
    return out

rng = np.random.default_rng(2026)
SUBSETS = []
for k in (2, 3, 4):
    allc = list(it.combinations(STATIONS, k)); pick = rng.choice(len(allc), 6, replace=False); SUBSETS += [allc[i] for i in pick]
SUBSETS += list(it.combinations(STATIONS, 5)) + [tuple(STATIONS)]
print("Subsets:", len(SUBSETS))

def job(sub, Y):
    smask = META.station.isin(sub).values; cut = IDX.get_loc(pd.Timestamp(f"{Y - 1}-12-01"))
    te = np.where(smask & (META.year.values == Y))[0]
    H = np.mean([hyb_fit_pred(smask, cut, te, s) for s in LC_SEEDS], axis=0); RP = ridge_pred(sub, Y); rows = []
    for j, i in enumerate(te):
        for L in LEADS:
            rows.append(dict(subset="|".join(sub), k=len(sub), station=META.station[i], year=Y, month=int(META.month[i]), lead=L, obs=YOBS[i, L - 1],
                             hyb=H[j, L - 1], ridge=RP.get((META.station[i], Y, int(META.month[i]), L), np.nan), sarima=SARM[i, L - 1], trend=TC[i, L - 1]))
    return rows
t0 = time.time()
OUT = pd.DataFrame(sum(Parallel(n_jobs=-1)(delayed(job)(sub, Y) for sub in SUBSETS for Y in TEST_YEARS), []))
print(f"done in {(time.time()-t0)/60:.1f} min, rows {len(OUT)}")
OUT.to_csv(PROC / "learning_curve_forecasts.csv", index=False)
json.dump({"subsets": ["|".join(s) for s in SUBSETS], "hid": HID, "wd": WD, "seeds": LC_SEEDS, "subset_seed": 2026}, open(RES / "nb19_learning_curve_settings.json", "w"), indent=2)
print("saved")
