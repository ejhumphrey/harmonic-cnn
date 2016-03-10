"""Model definitions. Each model definition takes an
    * n_in : int
    * n_out : int
  There are different models for cqt and wcqt ('cause you need to treat
  them differently).
  Each returns the training function, and the prediction function.
"""
import lasagne
import theano
import theano.tensor as T


def cqt_iX_c1f1_oY(n_in, n_out, verbose=False):
    """Variable length input cqt learning model.

    Parameters
    ----------
    n_in : int
        Length of the input window.

    n_out : int
        Number of output dimensions.

    Returns
    -------
    trainer, predictor : theano Functions
    """
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = lasagne.layers.InputLayer((None, 1, n_in, 252),
                                        input_var=input_var)

    # A convolution layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=16, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Max Pooling
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Fully connected with 256 units & 50% dropout on inputs
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # Output unit with parameterized output dimension.
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)

    # Create the loss expression for training.
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Collect params and update expressions.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)
    # TODO: Make these hyperparameters accessible outside.

    # Loss expression for evaluation.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    return train_fn, val_fn
