import numpy as np
from tensorflow import keras


def _init_lstm(num_features, hyperparams):
    '''
    Initialize an lstm (artificial) neural network with a single lstm
    layer with one output. Uses mean squared error as the loss, mean
    absolute difference as the risk, and adam for optimization.
    '''
    model = keras.Sequential()
    model.add(
        keras.layers.LSTM(
            1,
            input_shape=(num_features, 1)))
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(
            learning_rate=hyperparams.alpha,
            beta_1=hyperparams.decay),
        metrics=[
            keras.metrics.MeanAbsoluteError()])
    return model


def lstm_train(X_train, t_train, X_val, t_val, hyperparams):
    '''
    Train an lstm recurrent neural network. Uses mean squared error as the
    loss, mean absolute difference as the risk, and adam for optimization.
    '''
    # Initialize model
    model = _init_lstm(X_train.shape[1], hyperparams)

    # Train model
    history = model.fit(
        X_train,
        t_train,
        batch_size=hyperparams.batch_size,
        epochs=hyperparams.max_epochs,
        validation_data=(
            X_val,
            t_val))

    losses_train = history.history['loss']
    risks_val = history.history['val_mean_absolute_error']
    epoch_best = np.argmin(risks_val)
    risks_best = risks_val[epoch_best]
    return model, risks_best, epoch_best, losses_train, risks_val


def lstm_predict(model, X_test, t_test):
    '''
    Given a trained lstm model and some input data, predicts outputs and
    calculates the resulting risk using mean absolute difference.
    '''
    t_hat = model(X_test)
    # Mean absolute difference
    risk = (1 / t_hat.shape[0]) * \
        np.linalg.norm(np.absolute(t_hat - t_test), 1)
    return t_hat, risk
