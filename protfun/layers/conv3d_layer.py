import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import theano.tensor.nlinalg

from os import path, listdir

class Conv3DLayer(lasagne.layers.Layer):
    """
    TODO: This lasagne layer should apply a 3-dimensional convolution
    """

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1), pad=0,
                 untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
                 convolution=theano.tensor.nnet.conv3d, **kwargs):
        pass

    def get_output_shape_for(self, input_shape):
        pass
        return None

    def get_output_for(self, molecule_ids, **kwargs):
        pass
        return None




