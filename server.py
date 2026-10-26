
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.exceptions import HTTPException
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import yfinance as yf
import math
import time
import re
import os

app = Flask(__name__, static_folder=".")

CACHE = {}
VALID = re.compile(r"^[A-Z0-9^.=\-]{1,22}$")

# Annual expense ratios, stored as fractions.
# Reference values; update when fund fees change.
ETF_TER = {
    "VOO": 0.0003,
    "QQQM": 0.0015,
    "QQQ": 0.0018,
    "SCHG": 0.0004,
}

def clean(value):
    try:
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fetch(symbol):
    now = time.time()

    if symbol in CACHE:
        timestamp, result = CACHE[symbol]
        if now - timestamp < 3600:
            return result

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period="max",
        interval="1mo",
        auto_adjust=True,
        actions=False,
        timeout=20,
    )

    if df.empty:
        raise ValueError("ไม่มีข้อมูลราคาสำหรับ " + symbol)

    prices = df["Close"].dropna().astype(float)

    dates = pd.to_datetime(prices.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)

    prices.index = dates.to_period("M").to_timestamp("M")
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()

    if len(prices) < 13:
        raise ValueError("ข้อมูลย้อนหลังน้อยกว่า 13 เดือน")

    info = {}
    try:
        info = ticker.get_info() or {}
    except Exception:
        pass

    quote_type = str(info.get("quoteType") or "Unknown")
    is_fund = quote_type.upper() in ("ETF", "MUTUALFUND")
    if symbol in ETF_TER:
        is_fund = True
        quote_type = "ETF"

    holdings = {}
    if is_fund:
        try:
            h = ticker.funds_data.top_holdings
            if isinstance(h, pd.DataFrame):
                for name, row in h.iterrows():
                    weight = clean(row.get("Holding Percent"))
                    if weight is not None and weight > 0:
                        holdings[str(name).upper()] = weight
        except Exception:
            pass

    expense = ETF_TER.get(symbol)

    # For other ETFs, accept only a valid fractional fee.
    if expense is None and is_fund:
        candidate = clean(info.get("annualReportExpenseRatio"))
        if candidate is not None and 0 <= candidate < 0.10:
            expense = candidate

    profile = {
        "name": info.get("longName")
                or info.get("shortName")
                or symbol,
        "type": quote_type,
        "sector": info.get("sector"),
        "category": info.get("category"),
        "currency": info.get("currency"),
        "description": (
            info.get("longBusinessSummary")
            or info.get("description")
            or ""
        )[:800],
        "expense": expense if is_fund else 0.0,
        "pe": clean(info.get("trailingPE")),
        "forward_pe": clean(info.get("forwardPE")),
        "pb": clean(info.get("priceToBook")),
        "beta": clean(info.get("beta")),
        "market_cap": clean(info.get("marketCap")),
    }

    result = {
        "s": prices,
        "info": profile,
        "holdings": holdings,
    }

    CACHE[symbol] = (now, result)
    return result


def trim(series, years):
    if years == "max":
        return series

    cutoff = series.index.max() - pd.DateOffset(
        years=int(years)
    )
    return series[series.index >= cutoff]


def calculate(series, years, rolling_window, rf):
    s = trim(series.dropna().sort_index(), years)

    if len(s) < 13:
        return None

    returns = s.pct_change().dropna()

    if returns.empty:
        return None

    n = len(returns)

    if s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return None

    cagr = (s.iloc[-1] / s.iloc[0]) ** (12 / n) - 1

    volatility = returns.std(ddof=1) * np.sqrt(12)

    drawdown = s / s.cummax() - 1
    mdd = drawdown.min()

    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess = returns - monthly_rf

    std = returns.std(ddof=1)

    sharpe = (
        excess.mean() / std * np.sqrt(12)
        if std > 0 else None
    )

    downside = np.minimum(excess, 0)
    downside_dev = np.sqrt(np.mean(downside ** 2))

    sortino = (
        excess.mean() / downside_dev * np.sqrt(12)
        if downside_dev > 0 else None
    )

    calmar = (
        cagr / abs(mdd) if mdd < 0 else None
    )

    rolling = (
        s / s.shift(rolling_window) - 1
    ).dropna()

    return {
        "cagr": clean(cagr),
        "volatility": clean(volatility),
        "max_drawdown": clean(mdd),
        "sharpe": clean(sharpe),
        "sortino": clean(sortino),
        "calmar": clean(calmar),
        "rolling_latest": clean(rolling.iloc[-1])
            if len(rolling) else None,
        "rolling_min": clean(rolling.min())
            if len(rolling) else None,
        "rolling_max": clean(rolling.max())
            if len(rolling) else None,
        "observations": n,
        "start": s.index[0].strftime("%Y-%m-%d"),
        "end": s.index[-1].strftime("%Y-%m-%d"),
    }


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "Latte Portfolio API v3",
    })


@app.post("/api/analyze")
def analyze():
    args = request.get_json(silent=True) or {}

    symbols = [
        str(x).strip().upper()
        for x in args.get("tickers", [])
    ]

    symbols = list(dict.fromkeys(symbols))[:20]

    if not symbols or any(
        not VALID.fullmatch(x) for x in symbols
    ):
        return jsonify(error="Ticker ไม่ถูกต้อง"), 400

    years = str(args.get("years", "5"))
    if years not in ("3", "5", "10", "max"):
        years = "5"

    rolling = int(args.get("rolling", 12))
    if rolling not in (12, 36, 60):
        rolling = 12

    rf = float(args.get("rf", 0.03))
    rf = max(-0.05, min(0.30, rf))

    records = {}
    errors = {}

    with ThreadPoolExecutor(
        max_workers=min(6, len(symbols))
    ) as pool:
        futures = {
            symbol: pool.submit(fetch, symbol)
            for symbol in symbols
        }

        for symbol, future in futures.items():
            try:
                records[symbol] = future.result()
            except Exception as e:
                errors[symbol] = str(e)[:200]

    data = {}

    for symbol, record in records.items():
        data[symbol] = {
            "metrics": calculate(
                record["s"], years, rolling, rf
            ),
            "profile": record["info"],
            "holdings_count": len(record["holdings"]),
        }

    # Correlation and holdings overlap
    correlation = {}
    overlap = {}

    for a in symbols:
        correlation[a] = {}
        overlap[a] = {}

        for b in symbols:
            if a not in records or b not in records:
                correlation[a][b] = None
                overlap[a][b] = None
                continue

            if a == b:
                correlation[a][b] = 1.0
            else:
                pair = pd.concat(
                    [
                        records[a]["s"].pct_change(),
                        records[b]["s"].pct_change()
                    ],
                    axis=1,
                    join="inner"
                ).dropna()

                pair = trim(pair, years)

                correlation[a][b] = (
                    clean(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                    if len(pair) >= 12 else None
                )

            h1 = records[a]["holdings"]
            h2 = records[b]["holdings"]

            overlap[a][b] = (
                clean(
                    sum(
                        min(value, h2.get(name, 0))
                        for name, value in h1.items()
                    )
                )
                if h1 and h2 else None
            )

    result = {
        "data": data,
        "errors": errors,
        "correlation": correlation,
        "overlap": overlap,
        "asof": pd.Timestamp.now(
            tz="Asia/Bangkok"
        ).strftime("%Y-%m-%d %H:%M %Z"),
        "years": years,
        "rolling": rolling,
        "method": (
            "Monthly adjusted close; monthly rebalancing; "
            "annualized historical statistics. "
            "ETF overlap uses reported holdings only."
        ),
        "portfolio": {},
        "dca": {},
    }

    # Calculate the portfolio only when weights supplied.
    raw_weights = args.get("weights")

    if raw_weights is not None:
        try:
            weights = {
                symbol: float(raw_weights.get(symbol, 0))
                for symbol in symbols
            }

            if any(
                not math.isfinite(v) or v < 0
                for v in weights.values()
            ):
                raise ValueError("น้ำหนักหุ้นไม่ถูกต้อง")

            total = sum(weights.values())

            if abs(total - 100) > 0.01:
                raise ValueError(
                    "น้ำหนักพอร์ตต้องรวมเป็น 100%"
                )

            active = {
                s: w / 100
                for s, w in weights.items()
                if w > 0
            }

            missing = [
                s for s in active if s not in records
            ]

            if missing:
                raise ValueError(
                    "โหลดราคาหุ้นไม่สำเร็จ: " +
                    ", ".join(missing)
                )

            # Align all active assets on shared months.
            prices = pd.concat(
                {
                    s: records[s]["s"]
                    for s in active
                },
                axis=1,
                join="inner"
            ).dropna().sort_index()

            prices = trim(prices, years)

            if len(prices) < 13:
                raise ValueError(
                    "ช่วงข้อมูลร่วมกันน้อยกว่า 13 เดือน"
                )

            monthly_returns = prices.pct_change().dropna()
            allocation = pd.Series(active)

            # Rebalance to target weights every month.
            portfolio_returns = (
                monthly_returns[allocation.index]
                * allocation
            ).sum(axis=1)

            nav = pd.Series(
                [1.0] + list(
                    (1 + portfolio_returns).cumprod()
                ),
                index=prices.index,
                dtype=float
            )

            portfolio_metrics = calculate(
                nav, "max", rolling, rf
            )

            if portfolio_metrics is None:
                raise ValueError(
                    "ข้อมูลไม่เพียงพอสำหรับคำนวณพอร์ต"
                )

            portfolio_metrics["method"] = (
                "Monthly rebalanced portfolio. "
                "Common adjusted-price history. "
                "Before trading fees, taxes and FX."
            )

            result["portfolio"] = portfolio_metrics

            initial = max(
                0.0,
                float(args.get("initial_capital", 0))
            )
            monthly = max(
                0.0,
                float(args.get("monthly_contribution", 0))
            )

            if not all(
                math.isfinite(x) for x in [initial, monthly]
            ):
                raise ValueError("จำนวนเงินลงทุนไม่ถูกต้อง")

            value = initial
            contributed = initial

            points = [{
                "date": prices.index[0].strftime("%Y-%m-%d"),
                "value": clean(value),
                "contributed": clean(contributed),
            }]

            for date, monthly_return in portfolio_returns.items():
                value *= 1 + float(monthly_return)
                value += monthly
                contributed += monthly

                points.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "value": clean(value),
                    "contributed": clean(contributed),
                })

            result["dca"] = {
                "ending_value": clean(value),
                "total_contributed": clean(contributed),
                "profit": clean(value - contributed),
                "points": points,
            }

            result["portfolio_period"] = {
                "start": prices.index[0].strftime("%Y-%m-%d"),
                "end": prices.index[-1].strftime("%Y-%m-%d"),
                "months": len(portfolio_returns),
            }

        except Exception as e:
            # Keep individual stock data visible.
            result["portfolio_error"] = str(e)
            result["portfolio"] = {
                "method": "Portfolio error: " + str(e)
            }

    return jsonify(result)


@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, HTTPException):
        return jsonify(
            error=error.description
        ), error.code

    app.logger.exception("API error")

    return jsonify({
        "error": "Server error: " + str(error)
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
