src=open(__file__.replace("run22b.py","run21.py")).read()
exec(compile(src[:src.index("ALPHA_LOG = {}")],"run21_head","exec"))
from sklearn.ensemble import GradientBoostingRegressor
YEARS=list(range(2005,2026)); cols=cols_for("s")
need=["obs","trend_clim"]+[c for c in cols if c not in ("sinv","cosv") and not c.startswith("st_") and not c.startswith("seas_")]
ok=P[need].notna().all(axis=1); D=P[ok&P.year.isin(YEARS)].copy(); D["sk"]=D.year*100+D.month
def gb(): return GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0)
D["roll"]=np.nan
for Y in YEARS:
    tr=P[ok&(P.valid_ts<=pd.Timestamp(f"{Y-1}-12-01"))]
    for L in LEADS:
        t_=tr[tr.lead==L]; te=P[ok&(P.year==Y)&(P.lead==L)]
        if len(te): D.loc[te.index,"roll"]=te.trend_clim.values+gb().fit(t_[cols].values,t_.y_res.values).predict(te[cols].values)
rng=np.random.default_rng(0); D["shuf"]=np.nan
fold=D.index.map(dict(zip(D.index,rng.permutation(len(D))%5))).values
for k in range(5):
    te=D[fold==k]; tr=D[fold!=k]
    for L in LEADS:
        t_=tr[tr.lead==L]; e_=te[te.lead==L]
        D.loc[e_.index,"shuf"]=e_.trend_clim.values+gb().fit(t_[cols].values,t_.y_res.values).predict(e_[cols].values)
rows=[]
for L in LEADS:
    g=D[D.lead==L]; rm=lambda c: np.mean([np.sqrt(np.mean((s[c]-s.obs)**2)) for _,s in g.groupby("station")])
    rows.append(dict(lead=L,rolling=rm("roll"),shuffled=rm("shuf"),lower_pct=100*(1-rm("shuf")/rm("roll"))))
R=pd.DataFrame(rows); R.to_csv(OUT/"leak22_gbm.csv",index=False); print(R.round(4).to_string())
