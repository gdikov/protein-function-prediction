import lasagne
import lasagne.layers.dnn


def standard_network(input, n_outputs, last_nonlinearity):
    """
    standard_network is a straightforward ConvNet with a total
    of 7 ConvLayers and 4 MaxPool layers. The network ends with
    fully connected layers before the last_nonlinearity is applied.

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
    network = input
    # add deep convolutional structure
    network = add_deep_conv_maxpool(network)
    # add deep dense fully connected layers
    network = add_dense_layers(network, n_layers=2, n_units=256)
    # add the output layer non-linearity
    output = lasagne.layers.DenseLayer(incoming=network, num_units=n_outputs,
                                       nonlinearity=last_nonlinearity)
    return output, 0


def add_deep_conv_maxpool(network):
    filter_size = (3, 3, 3)

    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                num_filters=32,
                                                filter_size=filter_size,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                   pool_size=(2, 2, 2),
                                                   stride=2)

    for i in range(0, 6):
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                    num_filters=2 ** (5 + i // 2),
                                                    filter_size=filter_size,
                                                    nonlinearity=lasagne.nonlinearities.leaky_rectify)
        if i % 2 == 1:
            network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                           pool_size=(2, 2, 2),
                                                           stride=2)
    return network


def add_dense_layers(network, n_layers, n_units):
    for i in range(0, n_layers):
        network = lasagne.layers.DenseLayer(incoming=network, num_units=n_units,
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
    return network
