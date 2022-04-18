from util import HyperParameters, get_data, normalize_data, denormalize_data, preprocess_data, plot_graph
from linear_regression import lr_train, lr_predict
from multi_layer_perceptron import mlp_train, do_forward_propagation, calc_loss_and_risk
import numpy as np


def k_fold_validation(X_kfold, t_kfold, hyperparams):
    '''
    Performs k-fold validaiton on the passed in datasets.
    '''
    risk_best = 10000
    training_data_local, training_data_best, alpha_best = None, None, None

    for i in range(len(alpha_values)):
        hyperparams.alpha = alpha_values[i]
        total_risk = 0
        partition_size = X_kfold.shape[0] // hyperparams.k

        for i in range(hyperparams.k):
            X_train = np.concatenate((X_kfold[0:partition_size * i], X_kfold[partition_size * (i + 1):]))
            t_train = np.concatenate((t_kfold[0:partition_size * i], t_kfold[partition_size * (i + 1):]))
            X_val = X_kfold[partition_size * i:partition_size * (i + 1)]
            y_val = t_kfold[partition_size * i:partition_size * (i + 1)]

            # training_data = lr_train(X_train, t_train, X_val, y_val, hyperparams)
            training_data = mlp_train(X_train, t_train, X_val, y_val, hyperparams)

            total_risk += training_data[1]
            if not training_data_local or training_data[1] < training_data_local[1]:
                training_data_local = training_data

        avg_risk = total_risk / hyperparams.k
        if avg_risk < risk_best:
            risk_best = avg_risk
            alpha_best = hyperparams.alpha
            training_data_best = training_data_local

    return training_data_best, alpha_best, risk_best


# MAIN CODE
# Hyperparameters
hyperparams = HyperParameters(alpha=0.0, batch_size=250, max_epochs=100, k=8, decay=0.05)
alpha_values = [0.1, 0.05, 0.01, 0.005, 0.001]
# Other parameters
num_test_samples = 1374
# Results
w_best, epoch_best, risk_best = None, None, None

# Get dataset
date_data, raw_data = get_data()
X, t = preprocess_data(raw_data)
X, t, data_bounds = normalize_data(X, t)

# Augment input data to include bias term
X = np.hstack((np.ones([X.shape[0], 1]), X))

# For the AAPL stock we have 10374 samples - will use 1374 for testing
X_train, X_test = X[:-num_test_samples], X[-num_test_samples:]
t_train, t_test = t[:-num_test_samples], t[-num_test_samples:]

print(X_train.shape, t_train.shape, X_test.shape, t_test.shape)

# Perform training
training_data_best, alpha_best, risk_best = k_fold_validation(X_train, t_train, hyperparams)
hyperparams.alpha = alpha_best

# Perform testing by the weights yielding the best validation performance
# t_hat_test, _, test_risk = lr_predict(X_test, training_data_best[0], t_test)
Ys = do_forward_propagation(X_test, training_data_best[0])
_, test_risk = calc_loss_and_risk(Ys, t)

t_hat_test, t_test = denormalize_data(Ys, t_test, data_bounds)

# plot_graph(date_data[-num_test_samples:], 'date', 'AAPL stock price', 'linear_regression/stock_price_predictions.jpg', t_hat_test, ydata2=t_test, xdates=True, ydata1label='predicted values', ydata2label='actual values')
# plot_graph([i for i in range(len(training_data_best[3]))], 'number of epochs', 'training loss', 'linear_regression/learning_curve_training_loss.jpg', training_data_best[3])
# plot_graph([i for i in range(len(training_data_best[4]))], 'number of epochs', 'validation risk', 'linear_regression/learning_curve_validation_risk.jpg', training_data_best[4])

plot_graph(date_data[-num_test_samples:], 'date', 'AAPL stock price', 'mlp/stock_price_predictions.jpg', t_hat_test, ydata2=t_test, xdates=True, ydata1label='predicted values', ydata2label='actual values')
plot_graph([i for i in range(len(training_data_best[3]))], 'number of epochs', 'training loss', 'mlp/learning_curve_training_loss.jpg', training_data_best[3])
plot_graph([i for i in range(len(training_data_best[4]))], 'number of epochs', 'validation risk', 'mlp/learning_curve_validation_risk.jpg', training_data_best[4])

print('K-FOLD VALIDATION LINEAR REGRESSION******************************')
print('The value of hyperparameter alpha that yielded the best performance = {0}'.format(hyperparams.alpha))
print('The associated average validation performance (risk) = {0}'.format(risk_best))
print('The associated test performance (risk) = {0}'.format(test_risk))
