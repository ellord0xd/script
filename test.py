import ccxt
import telegram
import pandas as pd
import yfinance as yf
import configparser
import argparse
import os
import backtrader as bt
import sqlite3
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, IchimokuIndicator
from strategy import StochasticStrategy
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
import requests
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle
import time
from bs4 import BeautifulSoup

#Define configuration variables
config = configparser.ConfigParser()
config.read('config.ini')
# Define the URL for the EUR/USD pair
url = "https://www.investing.com/currencies/eur-usd"

# Define the headers for the HTTP request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

# Download page and parse    
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Find EUR/USD price     
price_element = soup.find('span', {'class': 'instrument-price_last__KQzyA'}) 
if price_element:
    price = float(price_element.text)
else: 
    price = 0
    
# Download the page content and parse it using BeautifulSoup
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the current price of the EUR/USD pair
price = float(soup.find('span', {'class': 'instrument-price_last__KQzyA'}).text)

# Set up risk management parameters
stop_loss = 0.01  # 1% stop loss per trade
position_size = 0.02  # 2% position size per trade
trailing_stop = 0.005  # 0.5% trailing stop

# Set up initial balance
balance = 10000.0

# Loop to continuously check the price and send recommendations
while True:
    # Download the page content and parse it using BeautifulSoup
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the current price of the EUR/USD pair
    new_price = float(soup.find('span', {'class': 'instrument-price_last__KQzyA'}).text)
    
# Download historical data for EUR/USD from Yahoo Finance
symbol = "EURUSD=X"
data = yf.download(symbol, start="2022-01-01", end="2022-05-13")

# Calculate technical indicators using talib
macd, macdsignal, macdhist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
rsi = talib.RSI(data['Close'], timeperiod=14)
bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(data['Close'], timeperiod=20)
slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
adx = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

# Load the machine learning models from saved files
with open('model_1.pkl', 'rb') as f:
    model_1 = pickle.load(f)
with open('model_2.pkl', 'rb') as f:
    model_2 = pickle.load(f)
with open('model_3.pkl', 'rb') as f:
    model_3 = pickle.load(f)
    
#Define command-line arguments
parser = argparse.ArgumentParser(description='Trading Bot')
parser.add_argument('--symbol', type=str, required=True, help='The trading symbol to use')
parser.add_argument('--timeframes', type=str, required=True, help='The timeframes to use, separated by commas')
parser.add_argument('--strategy', type=str, required=True, help='The trading strategy to use')
args = parser.parse_args()

# Load config file
config_file_path = os.path.join('/home/ellord/Desktop/scripts/forex', 'config.ini')
if not os.path.isfile(config_file_path):
    print("Config file does not exist")
    exit() 
config = configparser.ConfigParser()
config.read(config_file_path)

# Check if program has read access to config.ini
if not os.access(config_file_path, os.R_OK):
    print("Program does not have read access to config.ini")
    exit()

# Define Telegram bot
bot_token = config.get('TELEGRAM', 'bot_token')
bot_chat_id = config.get('TELEGRAM', 'bot_chat_id')

if not bot_token or not bot_chat_id:
    print('Error: Please provide bot_token and bot_chat_id in config.ini')
    sys.exit(1)

# Convert chat_id to integer
chat_id = int(bot_chat_id)

# Define CCXT exchange
if 'BINANCE' not in config:
    print('Error: Please provide BINANCE section in config.ini')
    sys.exit(1)
    
exchange = ccxt.binance({
    'apiKey': config['BINANCE'].get('apiKey'),
    'secret': config['BINANCE'].get('secret'),
    'enableRateLimit': True
})
#Define SQLite database
conn = sqlite3.connect('tradingbot.db')
c = conn.cursor()

#Define trading symbol and timeframes
symbol = args.symbol
timeframes = args.timeframes.split(',')

#Define trading strategy
if args.strategy == 'rsi':
    class RSIStrategy(bt.Strategy):
        params = (('rsi_period', 14), ('rsi_upper', 70), ('rsi_lower', 30), ('sma_period', 20), ('tp_sl_ratio', 3))
def __init__(self):
    self.rsi = RSIIndicator(self.data.close, self.params.rsi_period)
    self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)
    self.tp_sl_ratio = self.params.tp_sl_ratio

def next(self):
    if self.position.size == 0:
        if self.rsi < self.params.rsi_lower:
            self.buy(size=self.broker.getcash() // self.data.close * self.tp_sl_ratio)
    elif self.position.size > 0:
        if self.rsi > self.params.rsi_upper:
            self.close()

cerebro = bt.Cerebro()
if args.strategy == 'moving_average':
    cerebro.addstrategy(MovingAverageStrategy)
elif args.strategy == 'rsi':
    class RSIStrategy(bt.Strategy):
        params = (('rsi_period', 14), ('rsi_upper', 70), ('rsi_lower', 30), ('sma_period', 20), ('tp_sl_ratio', 3))
def init(self):
    self.stoch = StochasticOscillator(self.data.high, self.data.low, self.data.close, self.params.stoch_period)
    self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)
    self.tp_sl_ratio = self.params.tp_sl_ratio

def next(self):
    if self.position.size == 0:
        if self.stoch.oscillator_k < self.params.stoch_lower:
            self.buy(size=self.broker.getcash() // self.data.close * self.tp_sl_ratio)
    elif self.position.size > 0:
        if self.stoch.oscillator_k > self.params.stoch_upper:
            self.close()

cerebro = bt.Cerebro()
cerebro.addstrategy(StochasticStrategy)

# Define data feed
data = {}
for timeframe in timeframes:
    data[timeframe] = bt.feeds.PandasData(dataname=exchange.fetch_ohlcv(symbol, timeframe))

# Add data feed to cerebro
for timeframe in timeframes:
    cerebro.adddata(data[timeframe])

# Set broker settings
cerebro.broker.setcash(1000)
cerebro.broker.setcommission(commission=0.001)

# Run cerebro engine
cerebro.run()

# Get trading results
strat = cerebro.runstrats[0][0]
pnl = cerebro.broker.getvalue() - cerebro.broker.getcash()
percent_pnl = pnl / cerebro.broker.getcash() * 100

# Print trading results
print('PnL: ${:.2f}'.format(pnl))
print('Percent PnL: {:.2f}%'.format(percent_pnl))
bot.send_message(chat_id=chat_id, text='PnL: ${:.2f}\nPercent PnL: {:.2f}%'.format(pnl, percent_pnl))

# Save trading results to SQLite database
c.execute('CREATE TABLE IF NOT EXISTS trading_results (symbol TEXT, timeframes TEXT, strategy TEXT, pnl REAL, percent_pnl REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
c.execute('INSERT INTO trading_results (symbol, timeframes, strategy, pnl, percent_pnl) VALUES (?, ?, ?, ?, ?)', (symbol, args.timeframes, args.strategy, pnl, percent_pnl))
conn.commit()

# Close database connection
conn.close()

# Get Reddit sentiment data
titles = get_reddit_sentiment_data()

# Load SVM model and vectorizer
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict sentiment
sentiment = predict_sentiment(svm_model, titles)

# Send Telegram message with sentiment
bot.send_message(chat_id=chat_id, text='Reddit sentiment for {}: {}'.format(symbol, sentiment[0]))

# Define function to plot stock data
def plot_stock_data(symbol, start_date, end_date):
    data = get_stock_data(symbol, start_date, end_date)
    plt.plot(data)
    plt.title(symbol)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Define function to plot technical indicators
def plot_technical_indicators(data):
    rsi, stoch, bb, atr, adx, ichimoku, obv, cmf = calculate_indicators(data)
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    axs[0, 0].plot(data)
    axs[0, 0].set_title('Price')
    axs[0, 1].plot(rsi)
    axs[0, 1].axhline(y=30, color='r', linestyle='-')
    axs[0, 1].axhline(y=70, color='r', linestyle='-')
    axs[0, 1].set_title('RSI')
    axs[1, 0].plot(stoch.oscillator_k)
    axs[1, 0].axhline(y=20, color='r', linestyle='-')
    axs[1, 0].axhline(y=80, color='r', linestyle='-')
    axs[1, 0].set_title('Stochastic Oscillator %K')
    axs[1, 1].plot(bb.bollinger_mavg)
    axs[1, 1].fill_between(data.index, bb.bollinger_hband, bb.bollinger_lband, alpha=0.1)
    axs[1, 1].set_title('Bollinger Bands')
    axs[2, 0].plot(atr.average_true_range)
    axs[2, 0].set_title('Average True Range')
    axs[2, 1].plot(adx.adx)
    axs[2, 1].axhline(y=25, color='r', linestyle='-')
    axs[2, 1].set_title('ADX')
    axs[3, 0].plot(ichimoku.ichimoku_a)
    axs[3, 0].plot(ichimoku.ichimoku_b)
    axs[3, 0].fill_between(data.index, ichimoku.ichimoku_a, ichimoku.ichimoku_b, alpha=0.1)
    axs[3, 0].set_title('Ichimoku Cloud')
    axs[3, 1].plot(obv.on_balance_volume)
    axs[3, 1].set_title('On Balance Volume')
    plt.tight_layout()
    plt.show()

# Define function to run the trading bot
def run_trading_bot():
    # Get command-line arguments
    symbol = args.symbol
    timeframes = args.timeframes.split(',')
    strategy = args.strategy
    # Run trading bot
    run_backtesting(symbol, timeframes, strategy)   
    plot_stock_data(symbol, '2020-01-01', '2022-05-01') 
    plot_technical_indicators(data[symbol]['2020-01-01':'2022-05-01']) 
    make_stock_predictions(symbol, '2020-01-01', '2022-05-01') 
    run_sentiment_analysis()

# Execute trading bot
run_trading_bot()


#--strategy --strategy SMAStrategy --strategy MACDStrategy  SMAStrategy
