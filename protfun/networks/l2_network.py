import lasagne
import lasagne.layers.dnn

from lasagne.regularization import regularize_layer_params_weighted, l2


def l2_network(input, n_outputs, last_nonlinearity):
    """
    l2_network is a shallow network with L2-norm regularization on all ConvLayers
    and DenseLayers (corresponds to a gaussian prior assumption on all weights).

    Usage::
        >>> import theano.tensor as T
        >>> from lasagne.layers import InputLayer
        >>> from lasagne.nonlinearities import sigmoid
        >>>
        >>> inputs = T.tensor4("inputs")
        >>> input_layer = InputLayer(input_var=inputs, shape=(None, 2, 32, 32, 32))
        >>> n_classes = 2
        >>> # apply the network
        >>> output_layer, l2_terms = dense_network(input_layer, n_classes, sigmoid)

    :param input: a lasagne layer, on top of which the network is applied
    :param n_outputs: number of output units in the last layer
    :param last_nonlinearity: what the non-linearity in the last layer should be
    :return: the last lasagne layer of the network, and L2 regularization terms
            if there are any (otherwise 0).
    """
    regularization = 0
    network = input

    # add deep convolutional structure
    network, penalty = add_shallow_conv_maxpool(network)
    regularization += penalty

    # add deep dense fully connected layers
    network, penalty = add_dense_layers(network, n_layers=1, n_units=256)
    regularization += penalty

    # add the output layer non-linearity
    network = lasagne.layers.DenseLayer(incoming=network, num_units=n_outputs,
                                        nonlinearity=last_nonlinearity)
    l2_penalty = regularize_layer_params_weighted({network: 0.2}, l2)
    regularization += l2_penalty
    return network, regularization


def add_shallow_conv_maxpool(network):
    regularization = 0
    filter_size = (3, 3, 3)

    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                num_filters=32,
                                                filter_size=filter_size,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l2_penalty = regularize_layer_params_weighted({network: 0.2}, l2)
    regularization += l2_penalty
    network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                   pool_size=(2, 2, 2),
                                                   stride=2)

    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                num_filters=64,
                                                filter_size=filter_size,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l2_penalty = regularize_layer_params_weighted({network: 0.2}, l2)
    regularization += l2_penalty
    network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                   pool_size=(2, 2, 2),
                                                   stride=2)

    return network, regularization


def add_dense_layers(network, n_layers, n_units):
    regularization = 0
    for i in range(0, n_layers):
        network = lasagne.layers.DenseLayer(incoming=network, num_units=n_units,
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l2_penalty = regularize_layer_params_weighted({network: 0.2}, l2)
        regularization += l2_penalty
    return network, regularization
