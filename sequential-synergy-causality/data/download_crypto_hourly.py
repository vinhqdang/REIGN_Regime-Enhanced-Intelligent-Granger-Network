"""Download hourly crypto prices (Yahoo chart API, period-based) -> return panel.

Hourly data (~700 days, ~16.8k points/asset) gives the large sample and the
market inefficiency where genuine synergistic structure is most likely to appear,
unlike near-efficient daily equity return means.
"""
import urllib.request, json, time, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
           "ADA-USD", "DOGE-USD", "LTC-USD", "AVAX-USD", "LINK-USD"]


def fetch_hourly(tkr, days=700, retries=4):
    now = int(time.time()); p1 = now - days * 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}"
           f"?period1={p1}&period2={now}&interval=1h")
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(urllib.request.urlopen(req, timeout=60).read())
            r = j["chart"]["result"][0]
            s = pd.Series(r["indicators"]["quote"][0]["close"],
                          index=pd.to_datetime(r["timestamp"], unit="s"), name=tkr).dropna()
            return s[~s.index.duplicated()]
        except Exception as e:  # noqa
            last = e; time.sleep(2 ** k)
    raise RuntimeError(f"{tkr}: {last}")


def build(days=700):
    px = {}
    for t in TICKERS:
        px[t] = fetch_hourly(t, days=days); time.sleep(0.3)
        print(f"  {t}: {len(px[t])}")
    df = pd.DataFrame(px).sort_index().dropna(how="any")
    rets = np.log(df).diff().dropna(how="any")
    rets.to_csv(os.path.join(HERE, "crypto_hourly_returns.csv"))
    json.dump(dict(tickers=TICKERS, n_obs=int(rets.shape[0]), n_assets=rets.shape[1],
                   start=str(rets.index[0]), end=str(rets.index[-1]), freq="1h"),
              open(os.path.join(HERE, "crypto_manifest.json"), "w"), indent=2)
    print(f"panel {rets.shape} {rets.index[0]} .. {rets.index[-1]}")
    return rets


if __name__ == "__main__":
    build()
