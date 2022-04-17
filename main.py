from util import get_data, normalize_data, denormalize_data, preprocess_data, plot_graph
import numpy as np


def lr_predict(X, w, t):
    '''
    LINEAR REGRESSION
    Given X, w, and t, predicts t_hat and calculates the corresponding loss (using mean square error) 
    and risk (using mean absolute difference).

    X_new: N x (d + 1)
    w: (d + 1) x 1
    t: N x 1 
    '''
    t_hat = np.matmul(X, w)
    # Mean square error
    loss = (1 / (2 * batch_size)) * np.linalg.norm(t_hat - t, 2) ** 2
    # Mean absolute difference
    risk = (1 / batch_size) * np.linalg.norm(np.absolute(t_hat - t), 1)

    return t_hat, loss, risk


def lr_train(X_train, t_train, X_val, t_val):
    '''
    LINEAR REGRESSION
    Performs training and validation on the respective datasets passed in using mini-batch gradient 
    descent with l2-regularization.

    X_train: N_train x (d + 1)
    t_train: N_train x 1
    X_val: N_val x (d + 1)
    t_val: N_val x 1
    '''

    # Initialize weights randomly, w: (d + 1) x 1
    w = np.random.rand(X_train.shape[1])

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(max_epochs):
        loss_this_epoch = 0
        for b in range(num_batches):
            # X_batch: batch_size x (d + 1)
            X_batch = X_train[b * batch_size:(b + 1) * batch_size]
            # t_batch: batch_size x 1
            t_batch = t_train[b * batch_size:(b + 1) * batch_size]

            # lr_predict t_hat
            _, loss_batch, _ = lr_predict(X_batch, w, t_batch)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent
            X_batch_T = np.matrix.transpose(X_batch)
            # gradient = (1 / batch_size) * (X^(T)Xw - X^(T)t)
            gradient = (1 / batch_size) * (np.matmul(np.matmul(X_batch_T, X_batch), w) - np.matmul(X_batch_T, t_batch))
            # Use l2 regularization
            w = w - alpha * (gradient + decay * w)

        # Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch / num_batches
        losses_train.append(training_loss)
        # Perform validation on the validation set by the risk
        _, _, risk_val = lr_predict(X_val, w, t_val)
        risks_val.append(risk_val)
        # Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            w_best, risk_best, epoch_best = w, risk_val, epoch

    return w_best, risk_best, epoch_best, losses_train, risks_val


def k_fold_validation(X_kfold, t_kfold):
    '''
    Performs k-fold validaiton on the passed in datasets.
    '''
    global alpha

    risk_best = 10000
    training_data_local, training_data_best, alpha_best = None, None, None

    for i in range(len(alpha_values)):
        alpha = alpha_values[i]
        total_risk = 0
        partition_size = X_kfold.shape[0] // k

        for i in range(k):
            X_train = np.concatenate((X_kfold[0:partition_size * i], X_kfold[partition_size * (i + 1):]))
            t_train = np.concatenate((t_kfold[0:partition_size * i], t_kfold[partition_size * (i + 1):]))
            X_val = X_kfold[partition_size * i:partition_size * (i + 1)]
            y_val = t_kfold[partition_size * i:partition_size * (i + 1)]

            training_data = lr_train(X_train, t_train, X_val, y_val)

            total_risk += training_data[1]
            if not training_data_local or training_data[1] < training_data_local[1]:
                training_data_local = training_data

        avg_risk = total_risk / k
        if avg_risk < risk_best:
            risk_best = avg_risk
            alpha_best = alpha
            training_data_best = training_data_local

    return training_data_best, alpha_best, risk_best


# MAIN CODE
# Hyperparameters
alpha = 0.0                 # Learning rate
batch_size = 200            # Batch size
max_epochs = 100            # Max number of iterations
k = 8                       # Number of partitions for k-fold validation
decay = 0.05                # Weight decay for l2-regularization
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
training_data_best, alpha_best, risk_best = k_fold_validation(X_train, t_train)

# Perform testing by the weights yielding the best validation performance
t_hat_test, _, test_risk = lr_predict(X_test, training_data_best[0], t_test)

t_hat_test, t_test = denormalize_data(t_hat_test, t_test, data_bounds)

plot_graph(date_data[-num_test_samples:], 'date', 'stock price', 'stock price predictions', t_hat_test, ydata2=t_test, xdates=True, ydata1label='predicted values', ydata2label='actual values')
plot_graph([i for i in range(len(training_data_best[3]))], 'number of epochs', 'training loss', 'learning_curve_training_loss.jpg', training_data_best[3])
plot_graph([i for i in range(len(training_data_best[4]))], 'number of epochs', 'validation risk', 'learning_curve_validation_risk.jpg', training_data_best[4])

print('K-FOLD VALIDATION******************************')
print('The value of hyperparameter alpha that yielded the best performance = {0}'.format(alpha_best))
print('The associated average validation performance (risk) = {0}'.format(risk_best))
print('The associated test performance (risk) = {0}'.format(test_risk))
