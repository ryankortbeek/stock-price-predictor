from matplotlib import pyplot as plt
from matplotlib import dates as pld
# https://pypi.org/project/yfinance/
import yfinance as yf
import numpy as np


class HyperParameters:
    def __init__(self, alpha, batch_size, decay, max_epochs, k):
        # Learning rate
        self.alpha = alpha
        # Batch size
        self.batch_size = batch_size
        # Weight decay for l2-regularization
        self.decay = decay
        # Max number of iterations
        self.max_epochs = max_epochs
        # Number of partitions for k-fold validation
        self.k = k
        

class Config:
    TICKER = 'AAPL'
    N = 50
    S = 1
    DATE_KEY = 'Date'
    CLOSE_KEY = 'Close'


def get_data():
    # Fetch historical data from yfinance (format: date, open, high, low, close, volume, dividends, stock splits)
    data = yf.Ticker(Config.TICKER).history(period='max', auto_adjust=True)
    # Save data
    data.to_csv('data/AAPL_daily.csv')
    # Get necessary fields
    close_data = data.loc[:, Config.CLOSE_KEY].to_numpy()
    date_data = data.reset_index().loc[:, Config.DATE_KEY].tolist()
    return date_data, close_data


def preprocess_data(data_arr):
    # Preprocess data - iterate chronologically
    # Get closing price of stock for previous 50 days as features
    X = np.array([data_arr[i - Config.N:i] for i in range(Config.N, len(data_arr))], dtype='float64')
    # Get actual closing price for the day as target
    t = np.array([data_arr[i] for i in range(Config.N, len(data_arr))], dtype='float64')
    return X, t


def normalize_data(X, t):
    # Normalize data
    mean, std_dev = np.mean(t), np.std(t)
    X = (X - mean) / std_dev
    t = (t - mean) / std_dev
    return X, t, (mean, std_dev)


def denormalize_data(t_hat, t, norm_params):
    mean, std_dev = norm_params
    t_hat = t_hat * std_dev + mean
    t = t * std_dev + mean
    return t_hat, t


def plot_graph(xdata, xlabel, ylabel, filename, ydata1, ydata2=None, ydata1label=None, ydata2label=None, xdates=False):
    def conv_dates(d):
        return pld.date2num(d)

    plt.figure()

    # Use dates to label the x-axis if necessary
    if xdates:
        conv_dates_vec = np.vectorize(conv_dates)
        xdata = conv_dates_vec(xdata)

        xtick_locator = pld.AutoDateLocator()
        xtick_formatter = pld.AutoDateFormatter(xtick_locator)
        ax = plt.axes()
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)
    
    plt.scatter(xdata, ydata1, color='blue', s=Config.S, label=ydata1label) if ydata1label else plt.scatter(xdata, ydata1, color='blue', s=Config.S)
    if ydata2 is not None:
        plt.scatter(xdata, ydata2, color='red', s=Config.S, label=ydata2label) if ydata2label else plt.scatter(xdata, ydata2, color='red', s=Config.S)
    
    if ydata1label or ydata2label:
        plt.legend()

    # Rotate date labels
    if xdates:
        [l.set_rotation(30) for l in ax.get_xticklabels()]

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
