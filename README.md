# forex trading bot
This script is a forex trading bot that does the following:

It parses data from the investing.com EUR/USD page to get the current live price.

It downloads historical data for the EUR/USD pair from Yahoo Finance.

It calculates technical indicators on that data like MACD, RSI, Bollinger Bands, etc.

It has defined different trading strategies like RSIStrategy, StochasticStrategy, MovingAverageStrategy.

It uses an SQLite database to store trading results.

It loads machine learning models to predict sentiment and price movements.

It uses the CCXT library to interact with a Binance API.

It can send notifications and recommendations to a Telegram bot.

It defines functions to plot stock data and technical indicators.

It has a run_trading_bot() function that:

Runs the backtesting
Plots the stock data
Plots indicators
Makes stock predictions
Runs sentiment analysis
You can specify the trading strategy using the --strategy flag, for example:
--strategy RSIStrategy
--strategy MovingAverageStrategy

So in summary, this script aims to act as an end-to-end automated trading bot by:

Monitoring the latest price
Analyzing historical data
Using technical indicators and machine learning to generate signals
Placing trades through an exchange API
Reporting results and sending notifications
Plotting data for analysis purposes
