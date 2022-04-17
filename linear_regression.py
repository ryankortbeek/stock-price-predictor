import numpy as np

def lr_predict(X, w, t, hyperparams):
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
    loss = (1 / (2 * hyperparams.batch_size)) * np.linalg.norm(t_hat - t, 2) ** 2
    # Mean absolute difference
    risk = (1 / hyperparams.batch_size) * np.linalg.norm(np.absolute(t_hat - t), 1)

    return t_hat, loss, risk


def lr_train(X_train, t_train, X_val, t_val, hyperparams):
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

    num_batches = int(np.ceil(X_train.shape[0] / hyperparams.batch_size))

    for epoch in range(hyperparams.max_epochs):
        loss_this_epoch = 0
        for b in range(num_batches):
            # X_batch: batch_size x (d + 1)
            X_batch = X_train[b * hyperparams.batch_size:(b + 1) * hyperparams.batch_size]
            # t_batch: batch_size x 1
            t_batch = t_train[b * hyperparams.batch_size:(b + 1) * hyperparams.batch_size]

            # lr_predict t_hat
            _, loss_batch, _ = lr_predict(X_batch, w, t_batch, hyperparams)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent
            X_batch_T = np.matrix.transpose(X_batch)
            # gradient = (1 / batch_size) * (X^(T)Xw - X^(T)t)
            gradient = (1 / hyperparams.batch_size) * (np.matmul(np.matmul(X_batch_T, X_batch), w) - np.matmul(X_batch_T, t_batch))
            # Use l2 regularization
            w = w - hyperparams.alpha * (gradient + hyperparams.decay * w)

        # Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch / num_batches
        losses_train.append(training_loss)
        # Perform validation on the validation set by the risk
        _, _, risk_val = lr_predict(X_val, w, t_val, hyperparams)
        risks_val.append(risk_val)
        # Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            w_best, risk_best, epoch_best = w, risk_val, epoch

    return w_best, risk_best, epoch_best, losses_train, risks_val
