"""Leakage demonstration: ridge on the SEAS5 anomaly scored with the paper's rolling origin and with shuffled splits."""
src=open(__file__.replace("run22.py","run21.py")).read()
exec(compile(src[:src.index("ALPHA_LOG = {}")],"run21_head","exec"))
YEARS=list(range(2005,2026))
cols=cols_for("s"); need=["obs","trend_clim"]+[c for c in cols if c not in ("sinv","cosv") and not c.startswith("st_") and not c.startswith("seas_")]
ok=P[need].notna().all(axis=1); D=P[ok&P.year.isin(YEARS)].copy(); D["sk"]=D.year*100+D.month
A_FIXED=811.130831
# A: paper's rolling origin with tuned alpha
pa,_=run_ridge("s",YEARS); D["roll_tuned"]=pa.loc[D.index]
# A2: rolling origin, fixed alpha
D["roll_fixed"]=np.nan
for Y in YEARS:
    tr=P[ok&(P.valid_ts<=pd.Timestamp(f"{Y-1}-12-01"))]
    for L in LEADS:
        t_=tr[tr.lead==L]; te=P[ok&(P.year==Y)&(P.lead==L)]
        if len(te): D.loc[te.index,"roll_fixed"]=te.trend_clim.values+predict(fit(t_,cols,A_FIXED),te,cols)
def kfold(group_key,K=5,seed=0):
    rng=np.random.default_rng(seed); keys=D[group_key].unique() if group_key else D.index.values
    perm=rng.permutation(keys); fold={k:i%K for i,k in enumerate(perm)}
    f=D[group_key].map(fold).values if group_key else D.index.map(fold).values
    out=pd.Series(np.nan,index=D.index)
    for k in range(K):
        te=D[f==k]; tr=D[f!=k]
        for L in LEADS:
            t_=tr[tr.lead==L]; e_=te[te.lead==L]
            if len(e_): out.loc[e_.index]=e_.trend_clim.values+predict(fit(t_,cols,A_FIXED),e_,cols)
    return out
res={}
for nm,key in [("shuf_start","sk"),("shuf_pair",None)]:
    runs=[kfold(key,seed=s) for s in range(5)]; D[nm]=pd.concat(runs,axis=1).mean(axis=1)  # mean of RMSE below uses per-seed
    res[nm]=runs
rows=[]
for L in LEADS:
    g=D[D.lead==L]
    def rm(col): return np.mean([np.sqrt(np.mean((s[col]-s.obs)**2)) for _,s in g.groupby("station")])
    r={"lead":L,"n":len(g),"rolling_tuned":rm("roll_tuned"),"rolling_fixed":rm("roll_fixed")}
    for nm in res:
        v=[]
        for run in res[nm]:
            gg=g.assign(p=run.loc[g.index]); v.append(np.mean([np.sqrt(np.mean((s.p-s.obs)**2)) for _,s in gg.groupby("station")]))
        r[nm]=float(np.mean(v)); r[nm+"_sd"]=float(np.std(v))
    rows.append(r)
R=pd.DataFrame(rows); R["lower_start_pct"]=100*(1-R.shuf_start/R.rolling_fixed); R["lower_pair_pct"]=100*(1-R.shuf_pair/R.rolling_fixed)
R.to_csv(OUT/"leak22.csv",index=False); print(R.round(4).to_string())
