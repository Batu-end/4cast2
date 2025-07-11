# 4Cast: Stock Prediction Project

## Description

4Cast is designed to explore and evaluate different algorithms for stock price prediction. The primary objective is to implement and compare the performance of various machine learning models on financial time-series data. This project is not finalized and serves as a proof-of-concept.

## Features

* **Multi-Model Implementation**: Integrates Long Short-Term Memory (LSTM), Support Vector Machine (SVM), and Linear Regression models.
* **Comparative Analysis**: Allows for the direct comparison of prediction results from the different implemented models.
* **Modular Structure**: Designed to be extensible for the future inclusion of other predictive algorithms.

## Models Used

The core of this project is the application and comparison of the following models to forecast stock prices.

### Long Short-Term Memory (LSTM)

A specialized type of Recurrent Neural Network (RNN), LSTM networks are adept at learning from time-series data. They can capture long-term dependencies in historical price data, making them a powerful candidate for financial forecasting.

### Support Vector Machine (SVM)

For this project, Support Vector Machine is used as a regression model (Support Vector Regression). It works by identifying an optimal hyperplane that best fits the historical stock data points, enabling it to predict future price values.

### Linear Regression

A fundamental statistical model, Linear Regression serves as a baseline for performance comparison. It models the relationship between independent variables (e.g., time, past prices) and a dependent variable (the future stock price) by fitting a linear equation to the observed data.

## Dynamic Training Methodology

To enhance prediction accuracy, 4Cast employs a dynamic training strategy that adapts to the current market sentiment. Instead of using a single, monolithic dataset, we use two distinct training approaches based on whether the stock is currently in a bullish or bearish phase.

### Bullish Trend Training

When the current chart for a stock is identified as **bullish** (experiencing a sustained uptrend), the prediction models (LSTM, SVM, etc.) are trained primarily using historical data from previous periods where the stock also exhibited bullish behavior. This allows the algorithm to learn from patterns, volatility, and momentum specific to upward trends.

### Bearish Trend Training

Conversely, if the stock is currently in a **bearish** phase (a sustained downtrend), the models are trained on historical data collected from prior bearish periods. The logic is that the characteristics of a falling market are fundamentally different from those of a rising one, and training on like-for-like historical data should yield more relevant predictions.

The goal of this dual-method approach is to create a more context-aware prediction engine that is better tuned to the immediate, prevailing market conditions. The trend is typically identified using technical indicators, such as moving average crossovers, before the appropriate training data is selected.
The exact value of this method is up for debate, however I wanted to approach this creatively, as there are computers which can do what we are aiming for at a 1000x faster rate.
