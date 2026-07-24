"""Intraday (hourly) GROUP causality: does a pair {A,B} jointly drive a distinct
asset C beyond each alone -- in volatility and in tail-stress? ANTE, market-conditioned."""
import sys, json; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from itertools import combinations
from synergistic_granger import ante_sg
nm={"TSLA":"Tesla","NVDA":"Nvidia","AAPL":"Apple","MSFT":"Microsoft","AMZN":"Amazon","META":"Meta",
    "JPM":"JPMorgan","XOM":"ExxonMobil","SPY":"S&P500","GLD":"Gold","USO":"Oil","TLT":"Treasuries"}
rets=pd.read_csv("data/famous_hourly_returns.csv",index_col=0,parse_dates=True)
def scan(panel, cond, label):
    P=(panel-panel.mean())/panel.std(); c=(cond-cond.mean())/cond.std()
    assets=[x for x in P.columns if x!="SPY"]
    rows=[]
    for C in assets:
        for A,B in combinations([a for a in assets if a!=C],2):
            r=ante_sg(P[C].values,P[A].values,P[B].values,cond=[c.values],p=1,alpha=0.05,contemp=True)
            rows.append((A,B,C,float(np.log10(max(r['final_E'],1e-9))),r['reject_time']))
    R=pd.DataFrame(rows,columns=["A","B","C","logE","t"]).sort_values("logE",ascending=False)
    m=len(R); fw=float(np.log10(m/0.05))
    top=[{"A":nm[x.A],"B":nm[x.B],"C":nm[x.C],"logE":x.logE,"t":(None if pd.isna(x.t) else int(x.t)),
          "fw":bool(x.logE>=fw)} for x in R.head(12).itertuples()]
    print(f"[{label}] {m} triples, fw_thr={fw:.2f}, fw_sig={(R['logE']>=fw).sum()}, any-reject={R['t'].notna().sum()}")
    for t in top[:8]: print(f"   {t['A']}+{t['B']}->{t['C']}: log10E={t['logE']:+.2f} t*={t['t']} fw={t['fw']}")
    return {"n":int(m),"fw_thr":fw,"n_fw":int((R['logE']>=fw).sum()),"n_reject":int(R['t'].notna().sum()),"top":top}
vol=rets.abs()
def tail(s):
    q=s.abs().expanding(min_periods=200).quantile(0.90); return (s.abs()>q).astype(float)
E=rets.apply(tail).dropna()
out={"volatility":scan(vol, vol["SPY"], "hourly volatility"),
     "tail":scan(E.drop(columns=[]), E["SPY"], "hourly tail-stress")}
json.dump(out,open("experiments/results/intraday_group_scan.json","w"),indent=2)
print("done")
