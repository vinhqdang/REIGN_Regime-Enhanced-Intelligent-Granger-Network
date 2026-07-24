import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, json
from itertools import combinations
from synergistic_granger import ante_sg
rets=pd.read_csv("data/famous_returns.csv",index_col=0,parse_dates=True)
nm={"TSLA":"Tesla","NVDA":"Nvidia","AAPL":"Apple","MSFT":"Microsoft","AMZN":"Amazon","META":"Meta",
    "JPM":"JPMorgan","XOM":"ExxonMobil","SPY":"S&P500","GLD":"Gold","USO":"Oil","TLT":"Treasuries","^VIX":"VIX"}
vol=rets.abs(); V=(vol-vol.mean())/vol.std()
assets=[c for c in V.columns if c!="SPY"]
rows=[]
for C in assets:
    others=[a for a in assets if a!=C]
    for A,B in combinations(others,2):
        r=ante_sg(V[C].values,V[A].values,V[B].values,cond=[V["SPY"].values],p=1,alpha=0.05,contemp=True)
        rows.append((A,B,C,float(np.log10(max(r['final_E'],1e-9))),r['reject_time']))
R=pd.DataFrame(rows,columns=["A","B","C","logE","t"]).sort_values("logE",ascending=False)
m=len(R); fw=float(np.log10(m/0.05))
R.to_csv("experiments/results/group_causal_scan.csv",index=False)
out={"n":int(m),"fw_thr_log10":fw,"n_sig":int((R['logE']>=fw).sum()),
     "top":[{"A":nm[x.A],"B":nm[x.B],"C":nm[x.C],"logE":x.logE,"t":(None if pd.isna(x.t) else int(x.t)),
             "sig":bool(x.logE>=fw)} for x in R.head(15).itertuples()]}
json.dump(out,open("experiments/results/group_causal_scan.json","w"),indent=2)
print("done", m, "triples; sig", out["n_sig"])
