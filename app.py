import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import pipeline
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

st.title("Indian Stock Analyzer")

# List of Top 100 Indian Stocks
top_100_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "LT.NS", "SBIN.NS", "ULTRACEMCO.NS",
    "M&M.NS", "SUNPHARMA.NS", "WIPRO.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS",
    "INDUSINDBK.NS", "BAJAJFINSV.NS", "TITAN.NS", "NESTLEIND.NS", "JSWSTEEL.NS",
    "HDFC.NS", "COALINDIA.NS", "DRREDDY.NS", "TATAMOTORS.NS", "BPCL.NS"
]

ticker = st.selectbox("Select a stock ticker:", top_100_stocks)
start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('today'))

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

if st.button("Fetch Data") and ticker:
    def fetch_data(ticker):
        try:
            return yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    data = fetch_data(ticker)
    price_col = 'Close'

    if data is not None:
        data.dropna(inplace=True)
        st.subheader("Closing Price Over Time")
        plt.figure(figsize=(14, 7))
        plt.plot(data[price_col], label=price_col, alpha=0.8, color='black')
        plt.legend()
        st.pyplot(plt)
        st.subheader(f"Raw Data for {ticker}")
        st.dataframe(data.head())
        
        data['Daily Return'] = data[price_col].pct_change()

        st.subheader("Descriptive Statistics")
        st.write(data.describe())

        risk_free_rate = 0.0658 / 252  
        excess_return = data['Daily Return'] - risk_free_rate
        sharpe_ratio = np.sqrt(252) * (excess_return.mean() / excess_return.std())

        st.subheader("Fundamental Analysis")
        ticker_info = yf.Ticker(ticker).info
        fundamentals = {
                "Market Cap": ticker_info.get("marketCap", "N/A"),
                "PE Ratio (TTM)": ticker_info.get("trailingPE", "N/A"),
                "PB Ratio": ticker_info.get("priceToBook", "N/A"),
                "Sharpe Ratio": f"{sharpe_ratio:.2f}",
                "Dividend Yield": ticker_info.get("dividendYield", "N/A"),
                "52-Week High": ticker_info.get("fiftyTwoWeekHigh", "N/A"),
                "52-Week Low": ticker_info.get("fiftyTwoWeekLow", "N/A"),
                "Beta": ticker_info.get("beta", "N/A"),
                "EPS (TTM)": ticker_info.get("trailingEps", "N/A"),
                "Revenue (TTM)": ticker_info.get("totalRevenue", "N/A"),
                "Profit Margins": ticker_info.get("profitMargins", "N/A")
            }
        fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"])
        st.table(fundamentals_df)

        data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()
        data['20 Day MA'] = data[price_col].rolling(window=20).mean()
        data['50 Day MA'] = data[price_col].rolling(window=50).mean()
        data['200 Day MA'] = data[price_col].rolling(window=200).mean()
        data['Volatility'] = data['Daily Return'].rolling(window=20).std()
        delta = data[price_col].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['EMA_20'] = data[price_col].ewm(span=20, adjust=False).mean()
        data['EMA_12'] = data[price_col].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data[price_col].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        risk_free_rate = 0.0658 / 252  
        excess_return = data['Daily Return'] - risk_free_rate
        sharpe_ratio = np.sqrt(252) * (excess_return.mean() / excess_return.std())

        st.subheader("Cumulative Return Over Time")
        st.line_chart(data['Cumulative Return'])

        st.subheader("Relative Strength Index (RSI)")
        st.line_chart(data['RSI'])

        st.subheader("Volatility (Rolling Std Dev)")
        if not data['Volatility'].empty:
            st.line_chart(data['Volatility'])
        else:
            st.warning("No volatility data available for the selected date range.")

        st.subheader("Moving Averages")
        plt.figure(figsize=(14, 7))
        plt.plot(data[price_col], label=price_col, alpha=0.8, color='blue')
        plt.plot(data['20 Day MA'], label='20 Day MA', alpha=0.75, color='orange')
        plt.plot(data['50 Day MA'], label='50 Day MA', alpha=0.75, color='green')
        plt.plot(data['200 Day MA'], label='200 Day MA', alpha=0.75, color='black')
        plt.title(f'Moving Averages for {ticker}')
        plt.legend()
        st.pyplot(plt)

        st.subheader("Moving Average Convergence Divergence (MACD)")
        plt.figure(figsize=(14, 7))
        plt.plot(data[price_col], label=price_col, alpha=0.8, color='black')
        plt.plot(data['MACD'], label='MACD', color='blue', alpha=0.9, linestyle='--')
        plt.plot(data['Signal'], label='Signal Line', color='red', alpha=0.9, linestyle='-.')
        plt.plot(data['EMA_12'], label='EMA 12', alpha=0.75, color='orange')
        plt.plot(data['EMA_26'], label='EMA 26', alpha=0.75, color='green')
        plt.legend()
        st.pyplot(plt)

        # Heatmap: Monthly Returns
        st.subheader("Monthly Return Heatmap")
        data.reset_index(inplace=True)
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        monthly_return = data[price_col].pct_change().groupby([data['Year'], data['Month']]).mean().unstack()

        plt.figure(figsize=(12, 6))
        sns.heatmap(monthly_return, cmap='YlGnBu', annot=True, fmt='.2%', cbar=True)
        plt.title('Monthly Return Heatmap')
        st.pyplot(plt)

        st.subheader("Day of the Week Return Heatmap")
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        weekly_return = data[price_col].pct_change().groupby([data['Year'], data['DayOfWeek']]).mean().unstack()

        plt.figure(figsize=(12, 4))
        sns.heatmap(weekly_return, cmap='YlOrRd', annot=True, fmt='.2%', cbar=True)
        plt.title('Day of the Week Return Heatmap')
        st.pyplot(plt)

        st.subheader("Sentiment Analysis based on the recent News")
        API_KEY = st.secrets["API_KEY"]
        newsapi = NewsApiClient(api_key=API_KEY)
        stock_info = yf.Ticker(ticker).info
        stock_name = stock_info.get('longName', ticker.split('.')[0])
        from_date = (pd.to_datetime('today') - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        to_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        articles = newsapi.get_everything(
            q=stock_name, 
            language='en',
            sort_by='publishedAt',
            from_param=from_date, 
            to=to_date,
            page_size=20
        )

        news_list = []
        sentiments = {'Positive': 0, 'Negative': 0}
        sentiment_trend = []

        if articles['status'] == 'ok' and articles['totalResults'] > 0:
            for article in articles['articles']:
                title = article['title']
                description = article['description']
                url = article['url']
                published_at = article['publishedAt']

                result = sentiment_model(title + " " + description)[0]
                label = result['label']
                sentiment_type = 'Positive' if label == 'POSITIVE' else 'Negative'

                sentiments[sentiment_type] += 1
                news_list.append((title, sentiment_type, url))

                sentiment_trend.append({
                    'date': pd.to_datetime(published_at),
                    'sentiment': sentiment_type
                })

            st.write("### Sentiment Summary")
            st.write(f"Positive: {sentiments['Positive']}")
            st.write(f"Negative: {sentiments['Negative']}")

            sentiment_df = pd.DataFrame(sentiment_trend)
            sentiment_df['date'] = sentiment_df['date'].dt.date
            sentiment_counts = sentiment_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            fig = px.bar(sentiment_counts, x='date', y='count', color='sentiment', barmode='group',
                         title='Sentiment Trend Over Time')
            st.plotly_chart(fig)

            for i, (title, sentiment, url) in enumerate(news_list, 1):
                st.write(f"{i}. {title[:75]}... - {sentiment}")
                st.write(f"[Read more]({url})")
        else:
            st.write("No recent news available.")

         # Prepare Data for Prediction
        st.subheader("Predictive Modeling - Linear Regression")
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data['DateOrdinal'] = data['Date'].map(pd.Timestamp.toordinal)

        # Features and Target
        X = data[['DateOrdinal']]
        y = data[price_col]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make Predictions
        predictions = model.predict(X_test)

        # Evaluation
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Mean Squared Error (MSE): {mse}")

        # Plot Predictions
        plt.figure(figsize=(14, 7))
        plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
        plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Linear Regression Predictions for {ticker}')
        plt.legend()
        st.pyplot(plt)

        st.success("Predictive Modeling Completed Successfully!")

        st.write("Thank you for your time :)")
