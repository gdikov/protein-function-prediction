import lasagne
import lasagne.layers.dnn


def small_dense_network(input, n_outputs, last_nonlinearity):
    """
    small_dense_network is a shallower variant of the dense_net, with fewer blocks with
    residual connections.

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
    network = transition_layers(network, num_filters=8)

    # add deep convolutional structure
    network = dense_net_block(network, depth=3, num_filters=8)
    network = transition_layers(network, num_filters=8)
    network = dense_net_block(network, depth=6, num_filters=8)
    network = transition_layers(network, num_filters=16)
    network = lasagne.layers.NonlinearityLayer(incoming=network, nonlinearity=lasagne.nonlinearities.leaky_rectify)

    # add the output layer non-linearity
    output = lasagne.layers.DenseLayer(incoming=network, num_units=n_outputs,
                                       nonlinearity=last_nonlinearity)
    return output, 0


def dense_net_block(network, depth=6, num_filters=16):
    previous = list()
    for i in range(0, depth):
        previous += [network]
        input = lasagne.layers.ConcatLayer(incomings=previous, axis=1)

        network = lasagne.layers.BatchNormLayer(incoming=input)
        network = lasagne.layers.NonlinearityLayer(incoming=network, nonlinearity=lasagne.nonlinearities.leaky_rectify)

        # add  bottleneck convolution
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                    num_filters=4 * num_filters,
                                                    filter_size=(1, 1, 1),
                                                    nonlinearity=lasagne.nonlinearities.identity)

        network = lasagne.layers.BatchNormLayer(incoming=network)
        network = lasagne.layers.NonlinearityLayer(incoming=network, nonlinearity=lasagne.nonlinearities.leaky_rectify)

        # add the proper convolution
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                    num_filters=num_filters,
                                                    filter_size=(3, 3, 3),
                                                    nonlinearity=lasagne.nonlinearities.identity)
    return network


def transition_layers(network, num_filters=16):
    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                num_filters=num_filters,
                                                filter_size=(1, 1, 1),
                                                nonlinearity=lasagne.nonlinearities.identity)
    network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network, pool_size=(2, 2, 2), stride=2)
    return network
