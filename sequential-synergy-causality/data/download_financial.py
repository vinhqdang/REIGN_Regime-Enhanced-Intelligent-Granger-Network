"""
Download real financial time series (daily) from the Yahoo Finance chart API and
build an aligned log-return panel for the ANTE synergy experiments.

No API key required. Uses urllib (honours the environment HTTPS proxy). Output:
  data/returns_panel.csv   -- date-indexed daily log returns, one column per asset
  data/prices_panel.csv    -- aligned adjusted-close prices
  data/manifest.json       -- tickers, roles, date range, counts

The basket is chosen as a synergy testbed: sector ETFs + macro/vol/commodity
drivers, where joint (super-additive) causal structure is economically plausible
(e.g. oil + financials jointly driving the broad market or energy sector; VIX
jointly with a sector driving another).
"""
import urllib.request, json, time, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ticker -> (human name, role)
BASKET = {
    "SPY":  ("S&P 500 ETF", "broad_market"),
    "XLF":  ("Financials sector", "sector"),
    "XLK":  ("Technology sector", "sector"),
    "XLE":  ("Energy sector", "sector"),
    "XLV":  ("Health-care sector", "sector"),
    "XLI":  ("Industrials sector", "sector"),
    "XLY":  ("Consumer-discretionary sector", "sector"),
    "XLP":  ("Consumer-staples sector", "sector"),
    "XLU":  ("Utilities sector", "sector"),
    "XLB":  ("Materials sector", "sector"),
    "TLT":  ("20y+ Treasury bonds", "rates"),
    "GLD":  ("Gold", "commodity"),
    "USO":  ("Crude oil", "commodity"),
    "HYG":  ("High-yield credit", "credit"),
    "UUP":  ("US dollar index", "fx"),
    "^VIX": ("CBOE volatility index", "volatility"),
}


def fetch(ticker, rng="10y", interval="1d", retries=4):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={rng}&interval={interval}")
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(urllib.request.urlopen(req, timeout=60).read())
            r = j["chart"]["result"][0]
            ts = r["timestamp"]
            ind = r["indicators"]
            adj = ind.get("adjclose", [{}])[0].get("adjclose")
            if adj is None:
                adj = ind["quote"][0]["close"]
            s = pd.Series(adj, index=pd.to_datetime(ts, unit="s").normalize(),
                          name=ticker).dropna()
            s = s[~s.index.duplicated(keep="last")]
            return s
        except Exception as e:  # noqa
            last = e
            time.sleep(2 ** k)
    raise RuntimeError(f"failed to fetch {ticker}: {last}")


def build_panel(rng="10y"):
    prices = {}
    for t in BASKET:
        prices[t] = fetch(t, rng=rng)
        time.sleep(0.4)
        print(f"  fetched {t:5s} n={len(prices[t])}")
    px = pd.DataFrame(prices).sort_index()
    px = px.dropna(how="any")                       # align on common trading days
    rets = np.log(px).diff().dropna(how="any")
    px.to_csv(os.path.join(HERE, "prices_panel.csv"))
    rets.to_csv(os.path.join(HERE, "returns_panel.csv"))
    manifest = dict(
        tickers={t: dict(name=BASKET[t][0], role=BASKET[t][1]) for t in BASKET},
        n_assets=px.shape[1], n_days=int(rets.shape[0]),
        start=str(rets.index[0].date()), end=str(rets.index[-1].date()),
        range=rng,
    )
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPanel: {rets.shape[0]} days x {px.shape[1]} assets "
          f"({manifest['start']} .. {manifest['end']})")
    return rets, manifest


if __name__ == "__main__":
    build_panel()
