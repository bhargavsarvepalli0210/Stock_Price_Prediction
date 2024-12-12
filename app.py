import matplotlib.pyplot as plt
from urllib.parse import quote as url_quote
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import io
from flask import Flask, render_template, request, send_file, url_for
from xgboost import XGBRegressor
from textblob import TextBlob
from flask import send_from_directory
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import concurrent.futures
import os
import sys
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, static_folder='static')

def load_ticker_codes(file_path):
    ticker_codes = []
    with open(file_path, "r") as file:
        inside_list = False
        for line in file:
            line = line.strip()
            if "TICKER_CODES" in line and "[" in line:
                inside_list = True
            elif inside_list:
                if "]" in line:
                    inside_list = False
                else:
                    ticker = line.strip('",[] ')
                    if ticker:
                        ticker_codes.append(ticker)
    return ticker_codes

def get_stock_price(ticker, retries=3, delay=5):
    sys.stderr = open(os.devnull, 'w')
    attempt = 0
    while attempt < retries:
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            if not history.empty and history["Close"].notna().any():
                price = history["Close"].iloc[-1]
                return ticker, price
            else:
                return ticker, None
        except Exception:
            attempt += 1
            time.sleep(delay)
    return ticker, None

def filter_top_100_companies(stock_prices, max_price):
    filtered_companies = [
        (company, price) for company, price in stock_prices.items() if price is not None and price <= max_price
    ]
    filtered_companies.sort(key=lambda x: x[1], reverse=True)
    return filtered_companies[:100]

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, start=pd.Timestamp.today() - pd.DateOffset(months=6), end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    return stock_data

def fetch_news_articles(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        xml_data = response.text
        df = pd.read_xml(io.StringIO(xml_data))
        for index, row in df.iterrows():
            articles.append({'title': row.get('title', 'No Title'), 'content': row.get('description', 'No Content'), 'link': row.get('link', '')})
    return articles

def analyze_sentiment(articles):
    sentiment_scores = []
    for article in articles:
        text = article['content']
        sentiment = TextBlob(text).sentiment.polarity if text else 0
        sentiment_scores.append(sentiment)
    return sentiment_scores

def add_technical_indicators(stock_data, sentiment_scores):
    stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
    stock_data['Sentiment'] = pd.Series(sentiment_scores).reindex(stock_data.index, method='ffill')
    for lag in [1, 2, 3, 5]:
        stock_data[f'Lag_{lag}'] = stock_data['Close'].shift(lag)
    stock_data.dropna(inplace=True)
    return stock_data

def train_model(train_data):
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'EMA_20', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Sentiment']
    X_train = train_data[features]
    y_train = train_data['Close']
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model, features

def predict_future_prices(model, test_data, features):
    predictions = []
    current_data = test_data.copy()
    for index in range(len(current_data)):
        if index > 0:
            current_data.iloc[index, current_data.columns.get_loc('Lag_1')] = current_data.iloc[index - 1, current_data.columns.get_loc('Close')]
            current_data.iloc[index, current_data.columns.get_loc('Lag_2')] = current_data.iloc[index - 2, current_data.columns.get_loc('Close')] if index > 1 else current_data.iloc[index, current_data.columns.get_loc('Lag_1')]
            current_data.iloc[index, current_data.columns.get_loc('Lag_3')] = current_data.iloc[index - 3, current_data.columns.get_loc('Close')] if index > 2 else current_data.iloc[index, current_data.columns.get_loc('Lag_2')]
            current_data.iloc[index, current_data.columns.get_loc('Lag_5')] = current_data.iloc[index - 5, current_data.columns.get_loc('Close')] if index > 4 else current_data.iloc[index, current_data.columns.get_loc('Lag_3')]
        current_data.iloc[index, current_data.columns.get_loc('SMA_5')] = current_data['Close'].rolling(window=5).mean().iloc[index]
        current_data.iloc[index, current_data.columns.get_loc('SMA_20')] = current_data['Close'].rolling(window=20).mean().iloc[index]
        current_data.iloc[index, current_data.columns.get_loc('EMA_20')] = current_data['Close'].ewm(span=20, adjust=False).mean().iloc[index]
        if index >= 5:
            pred = model.predict(current_data[features].iloc[[index]])
            predictions.append(pred[0])
        else:
            predictions.append(np.nan)
    return predictions

def create_matplotlib_plot(stock_data, predictions):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Prices', color='green')
    ax.plot(pd.date_range(start=pd.Timestamp.today(), periods=len(predictions), freq='B'), predictions, label='Predicted Prices', color='blue')
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.route('/static/images/<path:filename>')
def static_images(filename):
    return send_from_directory('static/images', filename)

@app.route('/index')
def Get_Started():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('aboutpage.html')

@app.route('/contact')
def contact():
    return render_template('contactpage.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/stocks')
def stocks():
    return render_template('stock_prices.html')

@app.route('/', methods=['GET', 'POST'])
def get_started():
    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()
        stock_data = fetch_stock_data(ticker)
        news_articles = fetch_news_articles(ticker)
        sentiment_scores = analyze_sentiment(news_articles)
        stock_data = add_technical_indicators(stock_data, sentiment_scores)
        test_period = 15
        train_data = stock_data[:-test_period]
        test_data = stock_data[-test_period:]
        model, features = train_model(train_data)
        predictions = predict_future_prices(model, test_data, features)
        img_str = create_matplotlib_plot(stock_data, predictions)
        return render_template('index.html', ticker=ticker, articles=news_articles, img_str=img_str)
    return render_template('homepage.html')

@app.route('/stock_price', methods=['POST'])
def stock_price():
    try:
        max_price = float(request.form['max_price'])
    except ValueError:
        return redirect(url_for('homepage'))
    config = {
        'file_path': os.path.join(os.getcwd(), 'result.py')
    }
    ticker_codes = load_ticker_codes(config['file_path'])
    stock_prices = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ticker = {executor.submit(get_stock_price, ticker): ticker for ticker in ticker_codes}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                ticker, price = future.result()
                stock_prices[ticker] = price
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
    filtered_companies = filter_top_100_companies(stock_prices, max_price)
    return render_template('stock_price.html', max_price=max_price, filtered_companies=filtered_companies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
