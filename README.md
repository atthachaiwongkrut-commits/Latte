# Portfolio Lab — Live Website

Web-based dashboard with add/remove tickers and automatic historical metrics.

## Deploy a public link (Render)
1. Create a new GitHub repository and upload **all files in this directory to the repository root**.
2. Go to https://dashboard.render.com/ and select **New → Blueprint**.
3. Connect your GitHub repository and approve the service described by `render.yaml`.
4. When deployment is complete, open the `https://portfolio-lab-live-....onrender.com` URL assigned by Render. (Your URL will differ.)
5. Bookmark that URL. The page runs in a normal browser, including on a phone.

Render may change available free plans, suspend services when idle, or charge for services if you pick a paid plan. Check service settings before confirming.

## Run locally (optional)
`pip install -r requirements.txt` then `python server.py` and visit `http://127.0.0.1:5000`.

## How calculations work
The server downloads monthly adjusted closing prices using Yahoo Finance through `yfinance`, computes CAGR, annualized volatility, drawdown, Sharpe, Sortino, Calmar and rolling returns. Pairwise correlation is from aligned monthly returns. ETF overlap uses **available top holdings only**, a partial estimate; missing holdings remain blank. External requests may fail due to rate-limits/availability. Returns are historically observed and cannot predict future returns. A risk-free rate is selectable in the interface. It is not an investment recommendation.

**This ZIP is deploy-ready, not pre-published:** you must deploy it through your own Render/GitHub accounts to obtain a publicly accessible URL.
