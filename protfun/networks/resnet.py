import lasagne
import lasagne.layers.dnn


def resnet(input, n_outputs, last_nonlinearity):
    network = input
    network = transition_layers(network, num_filters=8)
    # add deep convolutional structure
    network = resnet_block(network, depth=3, num_filters=8)
    network = transition_layers(network, num_filters=8)

    network = resnet_block(network, depth=5, num_filters=8)
    network = transition_layers(network, num_filters=8)

    network = resnet_block(network, depth=10, num_filters=8)
    network = transition_layers(network, num_filters=8)

    network = resnet_block(network, depth=15, num_filters=8)
    network = transition_layers(network, num_filters=8)

    # add the sigmoid outputs
    output = lasagne.layers.DenseLayer(incoming=network, num_units=n_outputs,
                                       nonlinearity=last_nonlinearity)
    return output


def resnet_block(network, depth=6, num_filters=16):
    previous = list()
    for i in range(0, depth):
        previous += [network]
        input = lasagne.layers.ElemwiseSumLayer(incomings=previous)

        # network = lasagne.layers.BatchNormLayer(incoming=network)
        network = lasagne.layers.NonlinearityLayer(incoming=input, nonlinearity=lasagne.nonlinearities.leaky_rectify)

        # add a bottleneck
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                    num_filters=4 * num_filters,
                                                    filter_size=(1, 1, 1),
                                                    nonlinearity=lasagne.nonlinearities.identity)

        # network = lasagne.layers.BatchNormLayer(incoming=network)
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
