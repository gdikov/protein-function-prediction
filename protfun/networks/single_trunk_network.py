import theano.tensor as T
import lasagne
import lasagne.layers.dnn


def single_trunk_network(input, n_outputs):
    network = input
    # add deep convolutional structure
    network = add_deep_conv_maxpool(network)
    # add deep dense fully connected layers
    network = add_dense_layers(network, n_layers=2, n_units=256)
    # end each branch with a softmax
    output = lasagne.layers.DenseLayer(incoming=network, num_units=n_outputs,
                                       nonlinearity=lasagne.nonlinearities.sigmoid)
    return output


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