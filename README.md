# stock-price-predictor

A suite of machine learning algorithms trained on historical data corresponding to the Apple stock (ticker: AAPL). Each model "analyzes" the previous 50 closing prices of a stock and predicts the next closing price.

## Implemented Machine Learning Algorithms

1. [Linear regression (LR)](/linear_regression.py)
   - implemented from scratch
2. [Long short-term memory (LSTM)](/lstm.py)
   - implemented using [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
   - recurrent artificial neural network
   - one lstm layer
3. [Multi-layer perceptron (MLP)](/multi_layer_perceptron.py)
   - implemented from scratch
   - fully connected feed-forward artificial neural network
   - one input layer
   - one hidden layer
   - one output layer

## Results

### LR

The results of the trained LR model on the test data can be seen in the image below.
![](lr/stock_price_predictions.jpg)

### LSTM

The results of the trained LSTM model on the test data can be seen in the image below.
![](lstm/stock_price_predictions.jpg)

### MLP

The results of the trained MLP model on the test data can be seen in the image below.
![](mlp/stock_price_predictions.jpg)

## Dataset

The dataset used for training, validation, and testing was retreived from [Yahoo Finance](https://ca.finance.yahoo.com) via the [Yahoo Finance Python Package](https://pypi.org/project/yfinance/).

## Loss Function

The loss function used in the implementation of all the models was [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error).

## Evaluation Metric

The metric used to evaluate all the models was [mean absolute difference](https://en.wikipedia.org/wiki/Mean_absolute_difference).

## Running Instructions

Ensure the following dependencies are installed.

- pandas
- numpy
- tensorflow
- matplotlib

Train and test the models with:

```
python3 main.py
```
