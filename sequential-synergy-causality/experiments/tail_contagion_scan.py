"""Group TAIL-CONTAGION causality: does the JOINT stress (worst-10% |return|) of a
pair {A,B} explain a distinct asset C's stress beyond each alone? (AND-gate synergy
that mean/variance methods miss).  Contemporaneous, market-stress-conditioned, ANTE."""
import sys, os, json; sys.path.insert(0,"src")
import numpy as np, pandas as pd
from itertools import combinations
from synergistic_granger import ante_sg
nm={"TSLA":"Tesla","NVDA":"Nvidia","AAPL":"Apple","MSFT":"Microsoft","AMZN":"Amazon","META":"Meta",
    "JPM":"JPMorgan","XOM":"ExxonMobil","SPY":"S&P500","GLD":"Gold","USO":"Oil","TLT":"Treasuries","^VIX":"VIX"}
rets=pd.read_csv("data/famous_returns.csv",index_col=0,parse_dates=True)
def stress(s):
    q=s.abs().expanding(min_periods=100).quantile(0.90); return (s.abs()>q).astype(float)
E=rets.apply(stress).dropna(); mkt=E["SPY"].values
assets=[c for c in E.columns if c!="SPY"]
rows=[]
for C in assets:
    for A,B in combinations([a for a in assets if a!=C],2):
        r=ante_sg(E[C].values,E[A].values,E[B].values,cond=[mkt],p=1,alpha=0.05,contemp=True)
        rows.append((A,B,C,float(np.log10(max(r['final_E'],1e-9))),r['reject_time']))
R=pd.DataFrame(rows,columns=["A","B","C","logE","t"]).sort_values("logE",ascending=False)
m=len(R); fw=float(np.log10(m/0.05))
R.to_csv("experiments/results/tail_contagion_scan.csv",index=False)
top=[{"A":nm[x.A],"B":nm[x.B],"C":nm[x.C],"logE":x.logE,"t":(None if pd.isna(x.t) else int(x.t)),
      "fw_sig":bool(x.logE>=fw),"reject":bool(x.t==x.t and x.t is not None)} for x in R.head(20).itertuples()]
json.dump({"n":int(m),"fw_thr":fw,"n_fw_sig":int((R['logE']>=fw).sum()),
           "n_reject":int(R['t'].notna().sum()),"top":top},
          open("experiments/results/tail_contagion_scan.json","w"),indent=2)
print("done",m,"triples; fw_sig",int((R['logE']>=fw).sum()),"; any-reject",int(R['t'].notna().sum()))
