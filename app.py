# app.py
# Python 3.10 compatible
# Streamlit app: Single-stock analytics + multi-asset portfolio analysis for Indian tickers

import io
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Optional deps (guarded): News & simple sentiment (single-stock tab)
try:
    from newsapi import NewsApiClient
    from textblob import TextBlob
    HAS_NEWS = True
except Exception:
    HAS_NEWS = False

# ML (single-stock simple baseline model)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="📈 Indian Stock & Portfolio Analyzer", layout="wide")
st.title("📈 Indian Stock & Portfolio Analyzer")

# ------------ Helpers ------------ #
@st.cache_data(show_spinner=False)
def load_universe(csv_path: str = "indian_stocks.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        # Expecting a column named 'symbol'; fallback to first column
        if "symbol" not in df.columns:
            df = df.rename(columns={df.columns[0]: "symbol"})
        df = df.dropna(subset=["symbol"]).drop_duplicates("symbol")
        return df
    except Exception as e:
        st.warning(f"Could not read {csv_path}: {e}. Falling back to a minimal list.")
        return pd.DataFrame({"symbol": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]})

@st.cache_data(show_spinner=False)
def fetch_price_data(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        # yfinance returns multiindex columns when multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"].copy()
        else:
            prices = data[["Close"]].copy()
            prices.columns = tickers
        prices = prices.dropna(how="all")
        return prices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def annualize_return(ret: pd.Series, periods_per_year: int = 252) -> float:
    return float(ret.mean() * periods_per_year)

def annualize_vol(ret: pd.Series, periods_per_year: int = 252) -> float:
    return float(ret.std(ddof=0) * math.sqrt(periods_per_year))

def sharpe_ratio(ret: pd.Series, rf_annual: float = 0.0658, periods_per_year: int = 252) -> float:
    rf_daily = rf_annual / periods_per_year
    excess = ret - rf_daily
    if excess.std(ddof=0) == 0:
        return float("nan")
    return float(math.sqrt(periods_per_year) * (excess.mean() / excess.std(ddof=0)))

def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return float(dd.min())

def portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray, rf_annual: float = 0.0658) -> dict:
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    port_ret = (returns @ weights)
    cum = (1 + port_ret).cumprod()
    met = {
        "Annual Return": annualize_return(port_ret),
        "Annual Vol": annualize_vol(port_ret),
        "Sharpe": sharpe_ratio(port_ret, rf_annual=rf_annual),
        "Max Drawdown": max_drawdown(cum),
        "CAGR": (cum.iloc[-1]) ** (252 / len(port_ret)) - 1 if len(port_ret) > 0 else float("nan"),
        "Final CumReturn": cum.iloc[-1] - 1 if len(cum) > 0 else float("nan"),
    }
    return met, port_ret, cum

def inverse_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    var = returns.var(ddof=0)
    inv_var = 1 / var.replace(0, np.nan)
    inv_var = inv_var.fillna(0.0)
    w = inv_var.values
    if w.sum() == 0:
        w = np.ones(len(w))
    return w / w.sum()

def efficient_frontier(returns: pd.DataFrame, n: int = 3000, rf_annual: float = 0.0658):
    mean = returns.mean().values
    cov = returns.cov().values
    n_assets = len(mean)
    if n_assets == 0:
        return pd.DataFrame()
    rf_daily = rf_annual / 252
    out = []
    for _ in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()
        mu = float(w @ mean * 252)
        vol = float(math.sqrt(w @ cov @ w) * math.sqrt(252))
        sharpe = (mu - rf_annual) / vol if vol > 0 else np.nan
        out.append((mu, vol, sharpe, w))
    ef = pd.DataFrame(out, columns=["ann_return", "ann_vol", "sharpe", "weights"]) 
    return ef

def beta_vs_benchmark(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return float("nan")
    cov = aligned.cov().iloc[0, 1]
    var_b = aligned.var().iloc[1]
    return float(cov / var_b) if var_b != 0 else float("nan")

# ------------ Sidebar: global inputs ------------ #
universe_df = load_universe("indian_stocks.csv")
all_tickers = universe_df["symbol"].tolist()

with st.sidebar:
    st.header("⚙️ Settings")
    default_start = date(2020, 1, 1)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=date.today())
    rf = st.number_input("Risk-free rate (annual, e.g., 0.0658 = 6.58%)", value=0.0658, format="%.5f")

# ------------ Tabs ------------ #
stock_tab, portfolio_tab = st.tabs(["🔍 Single Stock", "🧺 Portfolio Analysis"])

# ------------ Single Stock Tab ------------ #
with stock_tab:
    ticker = st.selectbox("Select a stock ticker", options=all_tickers, index=0 if all_tickers else None, key="single_sel")
    if st.button("Fetch Data", key="single_btn") and ticker:
        data = fetch_price_data(ticker, start_date, end_date)
        if not data.empty:
            price_col = data.columns[0]
            df = data.rename(columns={price_col: "Close"}).copy()
            st.subheader("Closing Price Over Time")
            st.line_chart(df["Close"])

            # basic table
            st.subheader(f"Raw Data for {ticker}")
            st.dataframe(df.head())

            df["Daily Return"] = df["Close"].pct_change()
            st.subheader("Descriptive Statistics")
            st.write(df.describe())

            # Tech indicators
            df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()
            for w in (20, 50, 200):
                df[f"{w} Day MA"] = df["Close"].rolling(window=w).mean()
            delta = df["Close"].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
            df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
            df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = df["EMA_12"] - df["EMA_26"]
            df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["Volatility20"] = df["Daily Return"].rolling(window=20).std()

            st.subheader("Cumulative Return")
            st.line_chart(df["Cumulative Return"]) 

            st.subheader("RSI (14)")
            st.line_chart(df["RSI"])

            st.subheader("Volatility (20D)")
            st.line_chart(df["Volatility20"])

            st.subheader("Moving Averages")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df["Close"], label="Close")
            ax.plot(df.index, df["20 Day MA"], label="20D MA")
            ax.plot(df.index, df["50 Day MA"], label="50D MA")
            ax.plot(df.index, df["200 Day MA"], label="200D MA")
            ax.legend()
            st.pyplot(fig)

            st.subheader("MACD")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(df.index, df["MACD"], label="MACD", linestyle="--")
            ax2.plot(df.index, df["Signal"], label="Signal")
            ax2.legend()
            st.pyplot(fig2)

            # Fundamental snapshot (best-effort; yfinance may deprecate .info occasionally)
            st.subheader("Fundamental Snapshot")
            try:
                tkr = yf.Ticker(ticker)
                info = tkr.info or {}
            except Exception:
                info = {}
            risk_free_daily = rf / 252
            sharpe = sharpe_ratio(df["Daily Return"].dropna(), rf_annual=rf)
            fundamentals = {
                "Market Cap": info.get("marketCap", "N/A"),
                "PE (TTM)": info.get("trailingPE", "N/A"),
                "PB": info.get("priceToBook", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
                "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
                "52W Low": info.get("fiftyTwoWeekLow", "N/A"),
                "Beta": info.get("beta", "N/A"),
                "EPS (TTM)": info.get("trailingEps", "N/A"),
                "Revenue (TTM)": info.get("totalRevenue", "N/A"),
                "Profit Margins": info.get("profitMargins", "N/A"),
                "Sharpe (daily→annualized)": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
            }
            st.table(pd.DataFrame(list(fundamentals.items()), columns=["Metric", "Value"]))

            # Monthly & weekly heatmaps
            st.subheader("Monthly Return Heatmap")
            tmp = df.copy()
            tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "Date"})
            tmp["Year"] = tmp["Date"].dt.year
            tmp["Month"] = tmp["Date"].dt.month
            monthly_return = tmp["Close"].pct_change().groupby([tmp["Year"], tmp["Month"]]).mean().unstack()
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.heatmap(monthly_return, cmap="YlGnBu", annot=True, fmt=".2%", cbar=True, ax=ax3)
            ax3.set_title("Monthly Average Return")
            st.pyplot(fig3)

            st.subheader("Day-of-Week Return Heatmap")
            tmp["DOW"] = tmp["Date"].dt.dayofweek
            dow = tmp["Close"].pct_change().groupby([tmp["Year"], tmp["DOW"]]).mean().unstack()
            fig4, ax4 = plt.subplots(figsize=(10, 3))
            sns.heatmap(dow, cmap="YlOrRd", annot=True, fmt=".2%", cbar=True, ax=ax4)
            ax4.set_title("Avg Return by Day-of-Week")
            st.pyplot(fig4)

            # News + sentiment (best-effort and optional)
            st.subheader("News Sentiment (last 30 days)")
            if HAS_NEWS:
                # API key priority: st.secrets, then text input in sidebar, then None
                api_key = st.secrets.get("NEWS_API_KEY", None) if hasattr(st, "secrets") else None
                if not api_key:
                    api_key = st.text_input("Enter NewsAPI key (optional)", type="password")
                if api_key:
                    try:
                        stock_info = yf.Ticker(ticker).info
                        stock_name = stock_info.get('longName', ticker.split('.')[0])
                        newsapi = NewsApiClient(api_key=api_key)
                        from_date = (pd.Timestamp.today(tz=None) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
                        to_date = pd.Timestamp.today(tz=None).strftime('%Y-%m-%d')
                        articles = newsapi.get_everything(q=stock_name, language='en', sort_by='publishedAt',
                                                           from_param=from_date, to=to_date, page_size=5)
                        news_list = []
                        sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
                        if articles.get('status') == 'ok' and articles.get('totalResults', 0) > 0:
                            for art in articles['articles']:
                                title = art.get('title') or ''
                                desc = art.get('description') or ''
                                url = art.get('url')
                                text = f"{title} {desc}".strip()
                                polarity = TextBlob(text).sentiment.polarity if text else 0
                                s_type = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
                                sentiments[s_type] += 1
                                news_list.append((title, s_type, url))
                            st.write("### Sentiment Summary")
                            st.write(f"Positive: {sentiments['Positive']}, Negative: {sentiments['Negative']}, Neutral: {sentiments['Neutral']}")
                            for i, (title, s_type, url) in enumerate(news_list, 1):
                                st.write(f"{i}. {title[:90]}… - {s_type}  ")
                                if url:
                                    st.markdown(f"[Read more]({url})")
                        else:
                            st.info("No recent news found.")
                    except Exception as e:
                        st.warning(f"News fetch failed: {e}")
                else:
                    st.caption("Provide a NewsAPI key to enable sentiment.")
            else:
                st.caption("Install optional packages 'newsapi-python' and 'textblob' to enable news sentiment.")

            # Simple predictive baseline (date → price linear reg; illustrative only)
            st.subheader("Predictive Modeling (Baseline Linear Regression)")
            dfx = df.reset_index().rename(columns={df.index.name or "index": "Date"}).copy()
            dfx['Date'] = pd.to_datetime(dfx['Date'])
            dfx['DateOrdinal'] = dfx['Date'].map(pd.Timestamp.toordinal)
            X = dfx[['DateOrdinal']]
            y = dfx['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression().fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            st.write(f"MAE: {mae:.4f}")
            st.write(f"MSE: {mse:.4f}")
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            ax5.scatter(X_test['DateOrdinal'], y_test, label='Actual')
            ax5.scatter(X_test['DateOrdinal'], preds, label='Predicted')
            ax5.legend()
            st.pyplot(fig5)
            st.success("Predictive Modeling Completed Successfully!")

# ------------ Portfolio Tab ------------ #
with portfolio_tab:
    st.markdown("### Build a Portfolio")
    cols = st.columns([2, 1])
    with cols[0]:
        sel = st.multiselect("Select tickers", options=all_tickers, default=all_tickers[:5] if len(all_tickers) >= 5 else all_tickers)
    with cols[1]:
        bench_default = "^NSEI"  # NIFTY 50 index on Yahoo Finance
        benchmark = st.text_input("Benchmark (Yahoo ticker)", value=bench_default)

    if st.button("Run Portfolio Analysis", key="pf_btn") and sel:
        prices = fetch_price_data(sel, start_date, end_date)
        if prices.empty:
            st.warning("No price data retrieved.")
        else:
            rets = daily_returns(prices)

            st.subheader("Price History (Close)")
            st.line_chart(prices)

            st.subheader("Correlation Heatmap")
            fig_hm, ax_hm = plt.subplots(figsize=(6 + 0.2 * len(sel), 4 + 0.2 * len(sel)))
            sns.heatmap(rets.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_hm)
            st.pyplot(fig_hm)

            # Default weights: equal
            init_w = np.repeat(1 / len(sel), len(sel))
            w_df = pd.DataFrame({"Ticker": sel, "Weight": init_w})
            st.caption("Edit weights so they sum to 1. We'll normalize if they don't.")
            w_df = st.data_editor(w_df, num_rows="fixed")
            w = w_df["Weight"].to_numpy(dtype=float)
            if w.sum() == 0:
                w = init_w.copy()
            w = w / w.sum()

            # Metrics current weights
            met, port_ret, cum = portfolio_metrics(rets[sel], w, rf_annual=rf)

            # Display key metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Annual Return", f"{met['Annual Return']*100:,.2f}%")
            kpi2.metric("Annual Vol", f"{met['Annual Vol']*100:,.2f}%")
            kpi3.metric("Sharpe", f"{met['Sharpe']:.2f}")
            kpi4.metric("Max Drawdown", f"{met['Max Drawdown']*100:,.2f}%")

            st.subheader("Portfolio Cumulative Return vs Benchmark")
            bench_prices = fetch_price_data(benchmark, start_date, end_date)
            if not bench_prices.empty:
                bench_ret = daily_returns(bench_prices).iloc[:, 0]
                bench_cum = (1 + bench_ret).cumprod()
                comp = pd.DataFrame({"Portfolio": cum, "Benchmark": bench_cum}).dropna()
                st.line_chart(comp)

                # Beta of portfolio vs benchmark
                beta = beta_vs_benchmark(port_ret, bench_ret)
                st.caption(f"Portfolio beta vs {benchmark}: {beta:.2f}" if not np.isnan(beta) else "Beta unavailable")
            else:
                st.line_chart(cum.rename("Portfolio"))
                st.caption("Benchmark data unavailable; showed portfolio only.")

            # Risk contributions
            st.subheader("Risk Contributions (approx.)")
            cov = rets[sel].cov()
            port_vol = math.sqrt(w @ cov.values @ w)
            mrc = cov.values @ w / port_vol if port_vol > 0 else np.zeros_like(w)
            rc = w * mrc
            rc_df = pd.DataFrame({"Ticker": sel, "Weight": w, "Risk Contribution": rc})
            rc_df["% Risk"] = rc_df["Risk Contribution"] / rc_df["Risk Contribution"].sum()
            st.dataframe(rc_df.set_index("Ticker"))

            # Suggested alt weights: inverse-variance and equal-weight
            st.subheader("Suggested Weights")
            iv_w = inverse_variance_weights(rets[sel])
            alt = pd.DataFrame({"Ticker": sel,
                                "Equal Weight": np.repeat(1/len(sel), len(sel)),
                                "Inv-Variance": iv_w})
            st.dataframe(alt.set_index("Ticker"))

            # Value-at-Risk & CVaR (parametric, 95%)
            st.subheader("Risk Measures (95% One-day)")
            z = 1.65
            mu = port_ret.mean()
            sigma = port_ret.std(ddof=0)
            var95 = -(mu - z * sigma)
            cvar95 = -(mu - sigma * (np.exp(-z**2/2) / (math.sqrt(2*math.pi) * (1-0.95)))) if sigma > 0 else float("nan")
            rm1, rm2 = st.columns(2)
            rm1.metric("VaR 95% (1D)", f"{var95*100:,.2f}%")
            rm2.metric("CVaR 95% (1D)", f"{cvar95*100:,.2f}%")

            # Rolling volatility chart (60D)
            st.subheader("Rolling 60D Volatility")
            roll_vol = port_ret.rolling(60).std() * math.sqrt(252)
            st.line_chart(roll_vol.rename("Ann. Vol (60D)"))

            # Efficient frontier (Monte Carlo) & highlight current
            st.subheader("Efficient Frontier (Monte Carlo)")
            ef = efficient_frontier(rets[sel], n=3000, rf_annual=rf)
            if not ef.empty:
                figef, axef = plt.subplots(figsize=(7, 5))
                axef.scatter(ef["ann_vol"], ef["ann_return"], s=8, alpha=0.35)
                cur_vol = annualize_vol(port_ret)
                cur_ret = annualize_return(port_ret)
                axef.scatter([cur_vol], [cur_ret], s=80, marker="*", label="Current", zorder=5)
                axef.set_xlabel("Annual Volatility")
                axef.set_ylabel("Annual Return")
                axef.legend()
                st.pyplot(figef)

                # Show top Sharpe candidates
                top = ef.nlargest(5, "sharpe").copy()
                top["weights"] = top["weights"].apply(lambda x: dict(zip(sel, np.round(x, 3))))
                st.write("Top Sharpe Portfolios (sampled)")
                st.dataframe(top[["ann_return", "ann_vol", "sharpe", "weights"]])

            # Download report
            st.subheader("Export")
            report = {
                "Metric": list(met.keys()),
                "Value": [met[k] for k in met.keys()],
            }
            rep_df = pd.DataFrame(report)
            weights_df = pd.DataFrame({"Ticker": sel, "Weight": w})
            with io.BytesIO() as buffer:
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    rep_df.to_excel(writer, index=False, sheet_name="Metrics")
                    weights_df.to_excel(writer, index=False, sheet_name="Weights")
                    prices.to_excel(writer, sheet_name="Prices")
                    rets.to_excel(writer, sheet_name="Returns")
                st.download_button(
                    label="Download Portfolio Report (Excel)",
                    data=buffer.getvalue(),
                    file_name="portfolio_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.caption("Note: Results are educational, not investment advice.")
