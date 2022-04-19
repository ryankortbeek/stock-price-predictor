import sys
import numpy as np
from sklearn.utils import shuffle
from util import HyperParameters, get_data, normalize_data, denormalize_data, preprocess_data, plot_graph
from linear_regression import lr_train, lr_predict
from multi_layer_perceptron import mlp_train, do_forward_pass, calc_loss_and_risk
from lstm import lstm_train, lstm_predict


class Algorithm:
    '''
    Supported algorithms.
    '''
    LR = 0
    MLP = 1
    LSTM = 2


def k_fold_validation(algorithm, X_kfold, t_kfold, hyperparams):
    '''
    Performs k-fold validation.
    '''
    risk_best = 10000
    training_data_local, training_data_best, decay_best = None, None, None

    for i in range(len(decay_values)):
        hyperparams.decay = decay_values[i]
        total_risk = 0
        partition_size = X_kfold.shape[0] // hyperparams.k

        for i in range(hyperparams.k):
            X_train = np.concatenate(
                (X_kfold[0:partition_size * i], X_kfold[partition_size * (i + 1):]))
            t_train = np.concatenate(
                (t_kfold[0:partition_size * i], t_kfold[partition_size * (i + 1):]))
            X_val = X_kfold[partition_size * i:partition_size * (i + 1)]
            t_val = t_kfold[partition_size * i:partition_size * (i + 1)]

            if algorithm == Algorithm.LR:
                training_data = lr_train(
                    X_train, t_train, X_val, t_val, hyperparams)
            elif algorithm == Algorithm.MLP:
                training_data = mlp_train(
                    X_train, t_train, X_val, t_val, hyperparams)
            elif algorithm == Algorithm.LSTM:
                training_data = lstm_train(
                    X_train, t_train, X_val, t_val, hyperparams)
            else:
                print('Invalid algorithm!', file=sys.stderr)
                sys.exit(1)

            total_risk += training_data[1]
            if not training_data_local or training_data[1] < training_data_local[1]:
                training_data_local = training_data

        avg_risk = total_risk / hyperparams.k
        if avg_risk < risk_best:
            training_data_best, risk_best, decay_best = training_data_local, avg_risk, hyperparams.decay

    return training_data_best, decay_best, risk_best


# MAIN CODE---------------------------------------------------------------
num_test_samples = 1500
w_best, epoch_best, risk_best = None, None, None

# Get dataset
date_data, raw_data = get_data()
X, t = preprocess_data(raw_data)
X, t, data_bounds = normalize_data(X, t)

# Augment input data to include bias term
X = np.hstack((np.ones([X.shape[0], 1]), X))

# Split train and test data
X_train, X_test = X[:-num_test_samples], X[-num_test_samples:]
t_train, t_test = t[:-num_test_samples], t[-num_test_samples:]

# Shuffle training data
X_train, t_train = shuffle(X_train, t_train)
print(X_train.shape, t_train.shape, X_test.shape, t_test.shape)

# Calculate 50-day moving avg for test data as baseline (ignore augmented
# bias term)
moving_avg = np.average(X_test[:, 1:], axis=1)

# LSTM--------------------------------------------------------------------
# Hyperparameters
lstm_hyperparams = HyperParameters(
    alpha=0.03,
    batch_size=500,
    max_epochs=60,
    k=5,
    decay=0.0)
decay_values = [0.9, 0.85, 0.8]

# Remove bias term from augmented data (bias is handled by tf.keras LSTM layer)
X_train, X_test = X_train[:, 1:], X_test[:, 1:]

# Perform training
training_data_best, decay_best, risk_best = k_fold_validation(
    Algorithm.LSTM, X_train, t_train, lstm_hyperparams)
lstm_hyperparams.decay = decay_best

# Perform testing by the lstm model yielding the best validation performance
t_hat_test, test_risk = lstm_predict(training_data_best[0], X_test, t_test)

# Denormalize data to see actual stock prices instead of normalized values ranging from 0 to 1
# t_hat_test, t_test = denormalize_data(t_hat_test, t_test, data_bounds)

plot_graph('date',
           'AAPL stock price (normalized)',
           'lstm/stock_price_predictions.jpg',
           date_data[-num_test_samples:],
           xdata_is_dates=True,
           average={'label': '50-day moving average',
                    'data': moving_avg,
                    'color': 'green'},
           actual={'label': 'actual values',
                   'data': t_test,
                   'color': 'red'},
           predicted={'label': 'predicted values',
                      'data': t_hat_test,
                      'color': 'blue'})
plot_graph('epoch',
           'training loss',
           'lstm/learning_curve_training_loss.jpg',
           [i for i in range(len(training_data_best[3]))],
           loss={'label': None,
                 'data': training_data_best[3],
                 'color': 'blue'})
plot_graph('epoch',
           'validation risk',
           'lstm/learning_curve_validation_risk.jpg',
           [i for i in range(len(training_data_best[4]))],
           risk={'label': None,
                 'data': training_data_best[4],
                 'color': 'blue'})

print('K-FOLD VALIDATION LSTM******************************')
print(
    'The value of hyperparameter decay that yielded the best performance = {0}'.format(
        lstm_hyperparams.decay))
print(
    'The associated average validation performance (risk) = {0}'.format(risk_best))
print('The associated test performance (risk) = {0}'.format(test_risk))

# LINEAR REGRESSION-------------------------------------------------------
# Hyperparameters
lr_hyperparams = HyperParameters(
    alpha=0.05,
    batch_size=500,
    max_epochs=60,
    k=5,
    decay=0.0)
decay_values = [0.15, 0.1, 0.05, 0.01]

# Perform training
training_data_best, decay_best, risk_best = k_fold_validation(
    Algorithm.LR, X_train, t_train, lr_hyperparams)
lr_hyperparams.decay = decay_best

# Perform testing by the weights yielding the best validation performance
t_hat_test, _, test_risk = lr_predict(X_test, training_data_best[0], t_test)

# Denormalize data to see actual stock prices instead of normalized values ranging from 0 to 1
# t_hat_test, t_test = denormalize_data(t_hat_test, t_test, data_bounds)

plot_graph('date',
           'AAPL stock price (normalized)',
           'lr/stock_price_predictions.jpg',
           date_data[-num_test_samples:],
           xdata_is_dates=True,
           average={'label': '50-day moving average',
                    'data': moving_avg,
                    'color': 'green'},
           actual={'label': 'actual values',
                   'data': t_test,
                   'color': 'red'},
           predicted={'label': 'predicted values',
                      'data': t_hat_test,
                      'color': 'blue'})
plot_graph('epoch',
           'training loss',
           'lr/learning_curve_training_loss.jpg',
           [i for i in range(len(training_data_best[3]))],
           loss={'label': None,
                 'data': training_data_best[3],
                 'color': 'blue'})
plot_graph('epoch',
           'validation risk',
           'lr/learning_curve_validation_risk.jpg',
           [i for i in range(len(training_data_best[4]))],
           risk={'label': None,
                 'data': training_data_best[4],
                 'color': 'blue'})

print('K-FOLD VALIDATION LINEAR REGRESSION******************************')
print(
    'The value of hyperparameter decay that yielded the best performance = {0}'.format(
        lr_hyperparams.decay))
print(
    'The associated average validation performance (risk) = {0}'.format(risk_best))
print('The associated test performance (risk) = {0}'.format(test_risk))

# MULTI-LAYER PERCEPTRON--------------------------------------------------
# Hyperparameters
mlp_hyperparams = HyperParameters(
    alpha=0.0001,
    batch_size=25,
    max_epochs=60,
    k=5,
    decay=0.0)
decay_values = [0.001, 0.0005]

# Perform training
training_data_best, decay_best, risk_best = k_fold_validation(
    Algorithm.MLP, X_train, t_train, mlp_hyperparams)
mlp_hyperparams.decay = decay_best

# Perform testing by the weights yielding the best validation performance
Ys = do_forward_pass(X_test, training_data_best[0])
_, test_risk = calc_loss_and_risk(Ys, t_test)

# Denormalize data to see actual stock prices instead of normalized values ranging from 0 to 1
# t_hat_test, t_test = denormalize_data(Ys[-1], t_test, data_bounds)

plot_graph('date',
           'AAPL stock price (normalized)',
           'mlp/stock_price_predictions.jpg',
           date_data[-num_test_samples:],
           xdata_is_dates=True,
           average={'label': '50-day moving average',
                    'data': moving_avg,
                    'color': 'green'},
           actual={'label': 'actual values',
                   'data': t_test,
                   'color': 'red'},
           predicted={'label': 'predicted values',
                      'data': Ys[-1],
                      'color': 'blue'})
plot_graph('epoch',
           'training loss',
           'mlp/learning_curve_training_loss.jpg',
           [i for i in range(len(training_data_best[3]))],
           loss={'label': None,
                 'data': training_data_best[3],
                 'color': 'blue'})
plot_graph('epoch',
           'validation risk',
           'mlp/learning_curve_validation_risk.jpg',
           [i for i in range(len(training_data_best[4]))],
           risk={'label': None,
                 'data': training_data_best[4],
                 'color': 'blue'})

print('K-FOLD VALIDATION MULTI-LAYER PERCEPTRON******************************')
print(
    'The value of hyperparameter decay that yielded the best performance = {0}'.format(
        mlp_hyperparams.decay))
print(
    'The associated average validation performance (risk) = {0}'.format(risk_best))
print('The associated test performance (risk) = {0}'.format(test_risk))
