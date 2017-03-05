import lasagne
import lasagne.layers.dnn

from lasagne.regularization import regularize_layer_params_weighted, l2


def l2_network(input, n_outputs, last_nonlinearity):
    regularization = 0
    network = input
    # add deep convolutional structure
    network, penalty = add_shallow_conv_maxpool(network)
    regularization += penalty
    # add deep dense fully connected layers
    network, penalty = add_dense_layers(network, n_layers=1, n_units=256)
    regularization += penalty
    # end each branch with a softmax
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
