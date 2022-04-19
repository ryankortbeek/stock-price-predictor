import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as pld


class HyperParameters:
    '''
    Simple class for storing hyperparameters.
    '''

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
    '''
    Simple class for defining dataset-related parameters.
    '''
    TICKER = 'AAPL'
    N = 50
    S = 1
    DATE_KEY = 'Date'
    CLOSE_KEY = 'Close'


def get_data():
    '''
    Fetches historical stock data from yfinance. Data comes in the format:
    (date, open, high, low, close, volume, dividends, stock splits)

    Source: https://pypi.org/project/yfinance/
    '''
    data = yf.Ticker(Config.TICKER).history(period='max', auto_adjust=True)
    # Save data for future reference
    data.to_csv('data/AAPL_daily.csv')
    # Get necessary fields
    close_data = data.loc[:, Config.CLOSE_KEY].to_numpy()
    date_data = data.reset_index().loc[:, Config.DATE_KEY].tolist()
    return date_data, close_data


def preprocess_data(data_arr):
    '''
    Preprocesses data by gathering the closing price of the stock over the
    previous 50 days for each day - these are the features, and then the
    actual closing price for the corresponding day - this is the target.
    '''
    # Get closing price of stock for previous 50 days as features
    X = np.array([data_arr[i - Config.N:i]
                 for i in range(Config.N, len(data_arr))], dtype='float64')
    # Get actual closing price for the day as target
    t = np.array([data_arr[i]
                 for i in range(Config.N, len(data_arr))], dtype='float64')
    return X, t


def normalize_data(X, t):
    '''
    Normalizes data to the range [0, 1].
    '''
    min_val, max_val = np.min(t), np.max(t)
    X = (X - min_val) / max_val
    t = (t - min_val) / max_val
    return X, t, (min_val, max_val)


def denormalize_data(t_hat, t, norm_params):
    '''
    Denormalizes data to convert normalized stock prices to actual stock
    prices.
    '''
    min_val, max_val = norm_params
    t_hat = t_hat * max_val + min_val
    t = t * max_val + min_val
    return t_hat, t


def plot_graph(
        xlabel,
        ylabel,
        filename,
        xdata,
        xdata_is_dates=False,
        **kwargs):
    '''
    Plots a graph.
    '''
    plt.figure()

    ax = plt.axes()
    if xdata_is_dates:
        # Use dates to label values on the x-axis
        def conv_dates(d):
            return pld.date2num(d)
        conv_dates_vec = np.vectorize(conv_dates)

        xdata = conv_dates_vec(xdata)
        xtick_locator = pld.AutoDateLocator()
        xtick_formatter = pld.AutoDateFormatter(xtick_locator)
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)

    # Plot datasets
    show_legend = False
    for _, value in kwargs.items():
        label, ydata, color = value['label'], value['data'], value['color']
        if label:
            show_legend = True
            plt.scatter(xdata, ydata, color=color, s=Config.S, label=label)
        else:
            plt.scatter(xdata, ydata, color=color, s=Config.S)

    if show_legend:
        plt.legend()

    # Rotate date labels
    if xdata_is_dates:
        for xticklabel in ax.get_xticklabels():
            xticklabel.set_rotation(30)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
