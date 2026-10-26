from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, numpy as np, yfinance as yf
import re, math, time
from concurrent.futures import ThreadPoolExecutor

app=Flask(__name__,static_folder='.')
cache={}
VALID=re.compile(r'^[A-Z0-9^.=\-]{1,22}$')

def clean(x):
    if x is None: return None
    try:
        y=float(x)
        return round(y,5) if np.isfinite(y) else None
    except (TypeError,ValueError):return None

def fetch(symbol):
    now=time.time()
    if symbol in cache and now-cache[symbol][0]<3600: return cache[symbol][1]
    t=yf.Ticker(symbol)
    df=t.history(period='max',interval='1mo',auto_adjust=True,actions=False,timeout=18)
    if df.empty or len(df)<3:raise ValueError('ไม่พบข้อมูลราคาย้อนหลังเพียงพอ')
    s=df['Close'].dropna().astype(float)
    s.index=pd.to_datetime(s.index).tz_localize(None).to_period('M').to_timestamp('M')
    s=s[~s.index.duplicated(keep='last')]
    if len(s)<3:raise ValueError('ข้อมูลย้อนหลังไม่เพียงพอ')
    info={}
    try:info=t.get_info() or {}
    except Exception:pass
    holdings={}
    fund=info.get('quoteType','').upper() in ('ETF','MUTUALFUND')
    if fund:
        try:
            h=t.funds_data.top_holdings
            if isinstance(h,pd.DataFrame):
                for ticker,row in h.iterrows():
                    weight=row.get('Holding Percent')
                    if pd.notna(weight): holdings[str(ticker).upper()]=float(weight)
        except Exception:pass
    result={'s':s,'info':{'name':info.get('longName') or info.get('shortName') or symbol,
        'type':info.get('quoteType') or 'Unknown','sector':info.get('sector'),
        'category':info.get('category'),'expense':clean(info.get('annualReportExpenseRatio') or info.get('totalAssetsExpenseRatio')),
        'description':(info.get('longBusinessSummary') or info.get('description') or '')[:850],
        'currency':info.get('currency'), 'source':'Yahoo Finance via yfinance'}, 'holdings':holdings}
    cache[symbol]=(now,result)
    return result

def calculate(s,years,window,rf):
    if years and years!='max':
        cutoff=s.index.max()-pd.DateOffset(years=int(years))
        s=s[s.index>=cutoff]
    if len(s)<13:return None
    returns=s.pct_change().dropna()
    n=len(returns)
    cagr=(s.iloc[-1]/s.iloc[0])**(12/n)-1
    vol=returns.std(ddof=1)*np.sqrt(12)
    dd=s/s.cummax()-1
    mdd=dd.min()
    monthly_rf=(1+rf)**(1/12)-1
    excess=returns-monthly_rf
    sharpe=excess.mean()/returns.std(ddof=1)*np.sqrt(12) if returns.std(ddof=1)>0 else None
    downside=np.minimum(excess,0)
    downside_dev=np.sqrt((downside**2).mean())
    sortino=excess.mean()/downside_dev*np.sqrt(12) if downside_dev>0 else None
    rolling=(s/s.shift(window)-1).dropna()
    return {'cagr':clean(cagr),'volatility':clean(vol),'max_drawdown':clean(mdd),'sharpe':clean(sharpe),
        'sortino':clean(sortino),'calmar':clean(cagr/abs(mdd) if mdd else None),
        'rolling_latest':clean(rolling.iloc[-1]) if len(rolling) else None,
        'rolling_min':clean(rolling.min()) if len(rolling) else None,'rolling_max':clean(rolling.max()) if len(rolling) else None,
        'observations':n+1,'start':s.index[0].strftime('%Y-%m-%d'),'end':s.index[-1].strftime('%Y-%m-%d')}

@app.get('/')
def index():return send_from_directory('.','index.html')

@app.post('/api/analyze')
def analyze():
    args=request.get_json(silent=True) or {}
    syms=[str(x).strip().upper() for x in args.get('tickers',[])][:20]
    syms=list(dict.fromkeys(syms))
    if not syms or any(not VALID.fullmatch(s) for s in syms):return jsonify(error='Ticker ไม่ถูกต้อง'),400
    years=str(args.get('years','5'))
    if years not in ('3','5','10','max'): years='5'
    window=int(args.get('rolling',12))
    if window not in (12,36,60):window=12
    rf=float(args.get('rf',0.03))
    rf=max(-0.05,min(0.3,rf))
    with ThreadPoolExecutor(max_workers=min(6,len(syms))) as pool:
        futures={s:pool.submit(fetch,s) for s in syms}
        records={};errors={}
        for sym,f in futures.items():
            try:records[sym]=f.result()
            except Exception as e:errors[sym]=str(e)[:150]
    output={}
    for sym,x in records.items():
        output[sym]={'metrics':calculate(x['s'],years,window,rf),'profile':x['info'],
                     'holdings_count':len(x['holdings'])}
    corr={};overlap={}
    for a in syms:
        corr[a]={};overlap[a]={}
        for b in syms:
            if a==b: corr[a][b]=1.0 if a in records else None;overlap[a][b]=1.0 if a in records and records[a]['holdings'] else None;continue
            if a not in records or b not in records:corr[a][b]=None;overlap[a][b]=None;continue
            sa=records[a]['s'];sb=records[b]['s']
            common=pd.concat([sa.pct_change(),sb.pct_change()],axis=1,join='inner').dropna()
            if years!='max' and len(common): common=common[common.index>=common.index.max()-pd.DateOffset(years=int(years))]
            corr[a][b]=clean(common.iloc[:,0].corr(common.iloc[:,1])) if len(common)>=12 else None
            h1=records[a]['holdings'];h2=records[b]['holdings']
            overlap[a][b]=clean(sum(min(v,h2.get(k,0)) for k,v in h1.items())) if h1 and h2 else None
    return jsonify({'data':output,'errors':errors,'correlation':corr,'overlap':overlap,
        'asof':pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),'years':years,'rolling':window,
        'method':'Monthly adjusted close; annualized statistics; percent overlap from reported top holdings only (partial/lower bound).'})

# ===== LATTE PORTFOLIO ANALYTICS FIX =====
# Add this BEFORE: if __name__ == '__main__':

_original_analyze = app.view_functions["analyze"]

def analyze_with_portfolio():
    response = _original_analyze()

    # Preserve original error responses
    if isinstance(response, tuple):
        return response

    result = response.get_json()
    args = request.get_json(silent=True) or {}

    raw_weights = args.get("weights") or {}
    if not raw_weights:
        return response

    try:
        weights = {
            str(k).upper(): float(v)
            for k, v in raw_weights.items()
            if float(v) > 0
        }

        if not weights:
            raise ValueError("ไม่มีหุ้นที่มีน้ำหนักมากกว่า 0%")

        total = sum(weights.values())

        if abs(total - 100) > 0.01:
            raise ValueError("น้ำหนักพอร์ตต้องรวม 100%")

        missing = [
            ticker for ticker in weights
            if ticker not in result.get("data", {})
        ]

        if missing:
            raise ValueError(
                "โหลดข้อมูลหุ้นไม่สำเร็จ: " +
                ", ".join(missing)
            )

        years = str(args.get("years", "5"))
        rolling = int(args.get("rolling", 12))
        rf = float(args.get("rf", 0.03))

        # Fetch cached monthly adjusted prices
        price_series = {}

        for ticker in weights:
            record = fetch(ticker)
            prices = record["s"].copy()

            if years != "max":
                cutoff = (
                    prices.index.max() -
                    pd.DateOffset(years=int(years))
                )
                prices = prices[prices.index >= cutoff]

            price_series[ticker] = prices

        # Use only dates shared by all assets
        prices = pd.concat(
            price_series,
            axis=1,
            join="inner"
        ).dropna()

        if len(prices) < 13:
            raise ValueError(
                "ข้อมูลที่ซ้อนทับกันไม่ถึง 13 เดือน"
            )

        # Monthly asset returns
        asset_returns = prices.pct_change().dropna()

        w = pd.Series({
            ticker: weight / 100
            for ticker, weight in weights.items()
        })

        # Monthly rebalanced portfolio
        portfolio_returns = (
            asset_returns[w.index] * w
        ).sum(axis=1)

        # Build synthetic portfolio NAV
        nav = pd.Series(
            [1.0] + list(
                (1 + portfolio_returns).cumprod()
            ),
            index=prices.index,
            dtype=float
        )

        metrics = calculate(
            nav, "max", rolling, rf
        )

        if metrics is None:
            raise ValueError(
                "คำนวณสถิติพอร์ตไม่ได้"
            )

        metrics["method"] = (
            "Monthly rebalanced portfolio; "
            "common adjusted-price history; "
            "gross historical returns before "
            "transaction costs and taxes."
        )

        result["portfolio"] = metrics

        # Historical DCA simulation
        initial = max(
            0, float(args.get("initial_capital", 0))
        )

        monthly = max(
            0, float(args.get("monthly_contribution", 0))
        )

        value = initial
        contributed = initial

        points = [{
            "date": prices.index[0].strftime("%Y-%m-%d"),
            "value": round(value, 2),
            "contributed": round(contributed, 2)
        }]

        for date, monthly_return in portfolio_returns.items():

            # Existing money grows first
            value *= (1 + float(monthly_return))

            # New contribution at month end
            value += monthly
            contributed += monthly

            points.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(value, 2),
                "contributed": round(contributed, 2)
            })

        result["dca"] = {
            "ending_value": round(value, 2),
            "total_contributed": round(contributed, 2),
            "profit": round(value - contributed, 2),
            "points": points
        }

        result["portfolio_period"] = {
            "start": prices.index[0].strftime("%Y-%m-%d"),
            "end": prices.index[-1].strftime("%Y-%m-%d"),
            "months": len(portfolio_returns)
        }

    except Exception as exc:
        result["portfolio"] = {
            "method": "Portfolio error: " + str(exc)
        }
        result["dca"] = {}
        result["portfolio_error"] = str(exc)

    response.set_json(result)
    return response

app.view_functions["analyze"] = analyze_with_portfolio

# ===== END LATTE PORTFOLIO FIX =====

if __name__=='__main__':
    print('OPEN: http://127.0.0.1:5000')
    app.run(host='0.0.0.0',port=int(__import__('os').environ.get('PORT','5000')),debug=False)
