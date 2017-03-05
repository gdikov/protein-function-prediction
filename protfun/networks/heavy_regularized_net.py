import lasagne
import lasagne.layers.dnn

"""
    Note: this network architecture was motivated by the following speculations:
    If we want the network to disregard the overall shape of the protein and look instead into the
    specific binding sites, let us naively restrict its receptive field by limiting the number of
    layers in depth. A crude estimate of the size of these sites was 10-15k A^3 which amounts to
    about 12x12x12 voxel receptive field size.
"""


def heavy_regularized_net(input, n_outputs, last_nonlinearity):
    """
    heavy_regularized_net is a shallow network with 0.7 dropout for regularization
    on all ConvLayers and DenseLayers.

    Usage::
        >>> import theano.tensor as T
        >>> from lasagne.nonlinearities import sigmoid
        >>>
        >>> inputs = T.tensor4("inputs")
        >>> n_classes = 2
        >>> # apply the network
        >>> output_layer, l2_terms = heavy_regularized_net(inputs, n_classes, sigmoid)

    :param input: a theano variable for the overall network input
    :param n_outputs: number of output units in the last layer
    :param last_nonlinearity: what the non-linearity in the last layer should be
    :return: the last lasagne layer of the network, and L2 regularization terms
            if there are any (otherwise 0).
    """
    network = input
    # add deep convolutional structure
    network = add_shallow_conv_maxpool(network)
    # add deep dense fully connected layers
    network = add_dense_layers(network, n_layers=1, n_units=256)
    # add the output layer non-linearity
    output = lasagne.layers.DenseLayer(incoming=network, num_units=n_outputs,
                                       nonlinearity=last_nonlinearity)
    return output, 0


def add_shallow_conv_maxpool(network):
    filter_size = (3, 3, 3)

    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                num_filters=32,
                                                filter_size=filter_size,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    network = lasagne.layers.DropoutLayer(incoming=network, p=0.8)
    network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                   pool_size=(2, 2, 2),
                                                   stride=2)

    network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                num_filters=64,
                                                filter_size=filter_size,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    network = lasagne.layers.DropoutLayer(incoming=network, p=0.8)
    network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                   pool_size=(2, 2, 2),
                                                   stride=2)

    return network


def add_dense_layers(network, n_layers, n_units):
    for i in range(0, n_layers):
        network = lasagne.layers.DenseLayer(incoming=network, num_units=n_units,
                                            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(incoming=network, p=0.8)
    return network
