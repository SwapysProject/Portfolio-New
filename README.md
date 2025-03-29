# 📈 Stock Analyzer

## 📖 Overview
The **Stock Analyzer** project is a **Streamlit** application designed to analyze stock market data. It provides insights through visualizations, fundamental analysis, sentiment analysis, and predictive modeling using **Linear Regression**.

## 📂 File Structure
- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of dependencies.
- `README.md`: Documentation of the project.
- `.gitignore`: Specifies intentionally untracked files to ignore.


## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/stock-analyzer.git
cd stock-analyzer
```

### 2. Create a Virtual Environment (Optional)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Libraries:**
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `newsapi-python`
- `textblob`
- `scikit-learn`

## 🔍 Key Features

### 1. Stock Data Analysis
- Fetches data for **Top 100 Indian Stocks** using **Yahoo Finance (yfinance)**.
- Visualizes closing prices, moving averages, and volatility.

### 2. Technical Indicators
- **Moving Averages (20, 50, 200-day)**
- **Relative Strength Index (RSI)**
- **Moving Average Convergence Divergence (MACD)**
- **Volatility (Rolling Standard Deviation)**

### 3. Fundamental Analysis
- **Market Cap**, **PE Ratio**, **PB Ratio**, **Sharpe Ratio**, and more.

### 4. Sentiment Analysis
- Fetches recent news using **NewsAPI**.
- Analyzes sentiment polarity using **TextBlob**.

### 5. Predictive Modeling
- Implements **Linear Regression** for stock price prediction.
- Evaluates model performance with **MAE** and **MSE**.

## 🚀 How to Run the Application
```bash
streamlit run app.py
```

## 📊 Model Performance
- **Mean Absolute Error (MAE):** Varies per stock.
- **Mean Squared Error (MSE):** Varies per stock.

## 💡 Insights
- Useful for identifying trends, volatility, and sentiment impact on stock prices.
- Helps in making informed investment decisions.

## 💬 Suggestions & Interaction
Suggestions, contributions, and advice are highly encouraged! I am willing to interact and collaborate with anyone interested in improving this project.



