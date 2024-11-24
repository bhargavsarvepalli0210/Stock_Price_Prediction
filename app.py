import concurrent.futures
import sys
import io
import json

# from concurrent.futures import ThreadPoolExecutor
import concurrent

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import requests
import yfinance as yf

# from flask import Flask, render_template, redirect, url_for,
from flask import Flask, render_template, request, send_from_directory, flash
from plotly.subplots import make_subplots
from result import TICKER_CODES
from textblob import TextBlob
from xgboost import XGBRegressor
import redis

redis_conn = redis.StrictRedis(host="localhost",port=6379)

app = Flask(__name__, static_folder="static")
app.secret_key = "your_secret_key"

CURRENCY_RATES = {
    "USD": 82.74,
    "EUR": 88.45,
    "GBP": 2.35,
    "AUD": 52.48,
    "CAD": 60.56,
    "JPY": 0.57,
    "CNY": 11.20,
    "CHF": 92.10,
    "NZD": 51.25,
    "SGD": 60.09,
    "INR": 1,
}


# Fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    stock_data = yf.download(
        ticker,
        start=pd.Timestamp.today() - pd.DateOffset(months=6),
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    return stock_data


# Fetch news articles from Google RSS Feed
def fetch_news_articles(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    articles = []

    if response.status_code == 200:
        xml_data = response.text
        df = pd.read_xml(io.StringIO(xml_data))
        for index, row in df.iterrows():
            articles.append(
                {
                    "title": row.get("title", "No Title"),
                    "content": row.get("description", "No Content"),
                    "link": row.get("link", ""),
                }
            )

    return articles


# Analyze sentiment of news articles
def analyze_sentiment(articles):
    sentiment_scores = []
    for article in articles:
        text = article["content"]
        if text:
            sentiment = TextBlob(text).sentiment.polarity
        else:
            sentiment = 0
        sentiment_scores.append(sentiment)
    return sentiment_scores


# Add technical indicators and lag features
def add_technical_indicators(stock_data, sentiment_scores):
    stock_data["SMA_5"] = stock_data["Close"].rolling(window=5).mean()
    stock_data["SMA_20"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["EMA_20"] = stock_data["Close"].ewm(span=20, adjust=False).mean()
    stock_data["Sentiment"] = pd.Series(sentiment_scores).reindex(
        stock_data.index, method="ffill"
    )

    for lag in [1, 2, 3, 5]:
        stock_data[f"Lag_{lag}"] = stock_data["Close"].shift(lag)

    stock_data.dropna(inplace=True)
    return stock_data


# Train the XGBoost model
def train_model(train_data):
    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "SMA_5",
        "SMA_20",
        "EMA_20",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_5",
        "Sentiment",
    ]
    X_train = train_data[features]
    y_train = train_data["Close"]

    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model, features


# Predict future prices
def predict_future_prices(model, test_data, features):
    predictions = []
    current_data = test_data.copy()

    for index in range(len(current_data)):
        if index > 0:
            current_data.iloc[index, current_data.columns.get_loc("Lag_1")] = (
                current_data.iloc[index - 1, current_data.columns.get_loc("Close")]
            )
            current_data.iloc[index, current_data.columns.get_loc("Lag_2")] = (
                current_data.iloc[index - 2, current_data.columns.get_loc("Close")]
                if index > 1
                else current_data.iloc[index, current_data.columns.get_loc("Lag_1")]
            )
            current_data.iloc[index, current_data.columns.get_loc("Lag_3")] = (
                current_data.iloc[index - 3, current_data.columns.get_loc("Close")]
                if index > 2
                else current_data.iloc[index, current_data.columns.get_loc("Lag_2")]
            )
            current_data.iloc[index, current_data.columns.get_loc("Lag_5")] = (
                current_data.iloc[index - 5, current_data.columns.get_loc("Close")]
                if index > 4
                else current_data.iloc[index, current_data.columns.get_loc("Lag_3")]
            )

        current_data.iloc[index, current_data.columns.get_loc("SMA_5")] = (
            current_data["Close"].rolling(window=5).mean().iloc[index]
        )
        current_data.iloc[index, current_data.columns.get_loc("SMA_20")] = (
            current_data["Close"].rolling(window=20).mean().iloc[index]
        )
        current_data.iloc[index, current_data.columns.get_loc("EMA_20")] = (
            current_data["Close"].ewm(span=20, adjust=False).mean().iloc[index]
        )

        if index >= 5:
            pred = model.predict(current_data[features].iloc[[index]])
            predictions.append(pred[0])
        else:
            predictions.append(np.nan)

    return predictions


# Prepare interactive plot using Plotly
def create_plot(stock_data, predictions, ticker):
    fig = make_subplots()

    # Add historical prices
    fig.add_trace(
        go.Scatter(
            x=stock_data["Date"],
            y=stock_data["Close"],
            mode="lines",
            name="Historical Prices",
            line=dict(color="green", width=2),
            hovertemplate="<b>Date</b>: %{x}<br><b>Price</b>: %{y}<extra></extra>",
        )
    )

    # Add predicted prices (15 future days)
    fig.add_trace(
        go.Scatter(
            x=pd.date_range(
                start=pd.Timestamp.today(), periods=len(predictions), freq="B"
            ),
            y=predictions,
            mode="lines",
            name="Predicted Prices",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Date</b>: %{x}<br><b>Predicted Price</b>: %{y}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Historical and Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        font=dict(size=12),
        legend=dict(font=dict(size=10)),
    )

    # Convert the plot to a div that can be embedded in HTML
    graph_div = fig.to_html(full_html=False)
    return graph_div


@app.route("/static/images/<path:filename>")
def static_images(filename):
    return send_from_directory("static/images", filename)


@app.route("/index")
def Get_Started():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("aboutpage.html")


@app.route("/contact")
def contact():
    return render_template("contactpage.html")


@app.route("/homepage")
def homepage():
    return render_template("homepage.html")


@app.route("/", methods=["GET", "POST"])
def get_started():
    if request.method == "POST":
        ticker = request.form["ticker"].strip().upper()

        stock_data = fetch_stock_data(ticker)
        news_articles = fetch_news_articles(ticker)
        sentiment_scores = analyze_sentiment(news_articles)

        stock_data = add_technical_indicators(stock_data, sentiment_scores)

        test_period = 15
        train_data = stock_data[:-test_period]
        test_data = stock_data[-test_period:]

        model, features = train_model(train_data)
        predictions = predict_future_prices(model, test_data, features)

        # Use Plotly to create interactive plot
        plot_div = create_plot(stock_data, predictions, ticker)

        return render_template(
            "index.html", ticker=ticker, articles=news_articles, plot_div=plot_div
        )

    return render_template("homepage.html")  # Redirect to homepage if GET request


@app.route("/select_form")
def select_form(value=False):
    try:
        print("the value is ", value)
        return render_template(
            "stocksDisplay.html", value=value, response=[]
        )
    except Exception as error:
        print(
            "the error select_form is ",
            error,
            " line no is ",
            sys.exc_info()[-1].tb_lineno,
        )
        return render_template(
            "stocksDisplay.html", value=value, response=[]
        )


def fetch_stock_record(response: dict):
    try:
        # response.append()
        len_response = len(response) - 1
        while len_response > 0:
            prev_record = response[len_response]
            curr_record = response[len_response - 1]

            prev_Value = response[len_response]["indian_price"]
            curr_value = response[len_response - 1]["indian_price"]

            if prev_Value < curr_value:
                break

            response[len_response], response[len_response - 1] = (
                curr_record,
                prev_record,
            )
            len_response -= 1

            if len_response < 0:
                break

        return response
    except Exception as error:
        print(
            "The fetch stock record errror is ",
            error,
            " line no is ",
            sys.exc_info()[-1].tb_lineno,
        )
        return response


def get_status(ticker, response, company_dict, price):
    try:
        # with ThreadPoolExecutor() as executor:
        # executor.map(get_status, TICKER_CODES)
        # for ticker in TICKER_CODES:
        # print("jjjj",ticker, response, company_dict, price)
        ticker_info = yf.Ticker(ticker)
        stocks = ticker_info.info
        company_name = stocks.get("longName", "N/A")
        company_currency = stocks.get("currency", "USD")

        stock_price = ticker_info.fast_info["open"]
        # print("stoclk price ", stock_price)
        if stock_price:
            indian_price = CURRENCY_RATES[company_currency] * stock_price
            company_dict[ticker] = {
                "name": company_name,
                "price": stock_price,
                "indian_price": indian_price,
                "currency": company_currency,
            }

            # print("ddd ", indian_price, price)
            if indian_price <= price:
                response.append(
                    {
                        "indian_price": indian_price,
                        "name": company_name,
                        "currency": company_currency,
                        "ticker": ticker
                    }
                )
                # stock_price, ticker, company_dict
                fetch_stock_record(response)
        return response
    except Exception as error:
        print("The get status errro is ", error)
        return response

# @app.route('/add-price',methods=["POST"], defaults={'response': [], 'page': 1})
# @app.route("/add-price/<response>/<page>", methods=["POST"])
@app.route("/add-price", methods=["POST","GET"])
def add_price():
    try:
        #price = None,response = [], page = 1
        print("The redis_conn is ", redis_conn.ping())
        page = int(request.args.get("page", 1))
        response = request.args.get("response", None)
        price = float(request.args.get("price", 0))

        # d_fixed = response.replace("'", '"')
        # print("d_fixed ", d_fixed)
        company_dict = {}
        if request.method == "POST":
            price = float(request.form["price"])
            response = []
            # breakpoint()
            if not redis_conn.get(str(price)):
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(
                            get_status, ticker, response, company_dict, float(price)
                        ): ticker
                        for ticker in TICKER_CODES
                        if len(response) < 100
                    }

                concurrent.futures.as_completed(futures)
                print("the length is ", len(response))
                res = redis_conn.set(str(price),json.dumps(response), ex = 86400)
                print("res is ", res)

            response = json.loads(redis_conn.get(str(price)))

            return render_template(
                "stocksDisplay.html",
                value=price,
                response=response,
                page = page,
                limit = 7
            )
        if response in ['None',None]:
            if not redis_conn.get(str(float(price))):
                response = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(
                            get_status, ticker, response, company_dict, float(price)
                        ): ticker
                        for ticker in TICKER_CODES
                        if len(response) < 100
                    }

                concurrent.futures.as_completed(futures)
                res = redis_conn.set(str(float(price)),json.dumps(response), ex = 86400)
                print("resdis res ", res)
            
            response = json.loads(redis_conn.get(str(float(price))))
        print("response ", response , price)
        if response:
            print("nddd ", len(response))
            return render_template(
                "stocksDisplay.html",
                value=price,
                response=response,
                page = page,
                limit = 7
            )
        return render_template("homepage.html")
    except Exception as error:
        print(
            "The add prices error is ",
            error,
            " limne no ",
            sys.exc_info()[-1].tb_lineno,
        )
        return render_template("homepage.html")


if __name__ == "__main__":
    app.run(debug=True)
