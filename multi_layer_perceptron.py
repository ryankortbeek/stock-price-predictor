import numpy as np


def _init_neural_network(dims):
    '''
    Randomly initializes the weights for a fully connected neural network.
    Augments weights to include bias term.

    w_l1: l1_dim x l2_dim
    w_l2: l2_dim x 1
    '''
    w = [np.random.rand(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
    return w


def do_forward_pass(X, w):
    '''
    Performs a forward pass through the neural network. Iterates over the
    layers in the neural network and computes the prediction. Uses sigmoid
    activation.
    '''
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    sigmoid_v = np.vectorize(sigmoid)

    Ys = [X]
    for i in range(len(w)):
        Ys.append(sigmoid_v(np.matmul(Ys[i], w[i])))
    return Ys


def calc_loss_and_risk(Ys, t):
    '''
    Calculates the loss (using mean squared error) and risk (using mean
    absolute difference) associated with Ys.
    '''
    t = np.reshape(t, [t.shape[0], 1])
    # mean squared error
    loss = (1 / Ys[-1].shape[0]) * np.linalg.norm(Ys[-1] - t, 2) ** 2
    # Mean absolute difference
    risk = (1 / Ys[-1].shape[0]) * np.linalg.norm(np.absolute(Ys[-1] - t), 1)
    return loss, risk


def do_backward_propagation(w, Ys, t):
    '''
    Calculates dws to use in gradient descent using back propagation.
    '''
    def signmoid_derivative(y):
        y_sigmoid = 1 / (1 + np.exp(-y))
        return y_sigmoid * (1 - y_sigmoid)
    sigmoid_derivative_v = np.vectorize(signmoid_derivative)
    loss, _ = calc_loss_and_risk(Ys, t)

    # Derivative of loss function - mean squared error
    loss_derivative = (2 / Ys[-1].shape[0]) * np.linalg.norm(Ys[-1] - t, 2)
    dzs = [loss_derivative * sigmoid_derivative_v(Ys[-1])]
    dws = []
    for i in range(len(w) - 1, -1, -1):
        dws.append(np.matmul(np.transpose(Ys[i]), dzs[-1]))
        # Derivative of loss function - mean squared error
        loss_derivative = (2 / Ys[-1].shape[0]) * np.linalg.norm(Ys[-1] - t, 2)
        dzs.append(loss_derivative * sigmoid_derivative_v(dzs[-1]))
    return dws, loss


def mlp_train(X_train, t_train, X_val, t_val, hyperparams):
    '''
    Train a multi-layer perceptron using mini-batch gradient descent via
    back propagation with l2-regularization.
    '''
    # Dimensions in the format [input layer, hidden layer, output layer]
    # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    neural_network_dims = [
        X_train.shape[1],
        X_train.shape[1] // 2,
        1]
    w = _init_neural_network(neural_network_dims)

    losses_train = []
    risks_val = []

    risk_best = 10000
    epoch_best = 0
    w_best = None

    num_batches = int(np.ceil(X_train.shape[0] / hyperparams.batch_size))

    for epoch in range(hyperparams.max_epochs):
        loss_this_epoch = 0
        for b in range(num_batches):
            # X_batch: batch_size x (d + 1)
            X_batch = X_train[b *
                              hyperparams.batch_size:(b +
                                                      1) *
                              hyperparams.batch_size]
            # t_batch: batch_size x 1
            t_batch = t_train[b *
                              hyperparams.batch_size:(b +
                                                      1) *
                              hyperparams.batch_size]

            Ys = do_forward_pass(X_batch, w)
            dws, loss = do_backward_propagation(w, Ys, t_batch)
            dws = dws[::-1]
            loss_this_epoch += loss
            # Mini-batch gradient descent (dws is in reverse order)
            for i in range(len(w)):
                # Use l2 regularization
                w[i] -= hyperparams.alpha * dws[i] + \
                    (hyperparams.decay * w[i])

        # Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch / num_batches
        losses_train.append(training_loss)

        # Perform validation on the validation set by the risk
        Ys = do_forward_pass(X_val, w)
        _, risk = calc_loss_and_risk(Ys, t_val)
        risks_val.append(risk)

        # Keep track of the best validation epoch, risk, and the weights
        if risk < risk_best:
            w_best, risk_best, epoch_best = w, risk, epoch

    return w_best, risk_best, epoch_best, losses_train, risks_val
