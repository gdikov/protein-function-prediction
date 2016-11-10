import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import theano.tensor.nlinalg

from os import path, listdir

class MaxPool3DLayer(lasagne.layers.Layer):
    """
    TODO: This lasagne layer should apply a 3-dimensional max-pooling
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0,0,0),
                 ignore_border=True, **kwargs):
        pass

    def get_output_shape_for(self, input_shape):
        pass
        return None

    def get_output_for(self, molecule_ids, **kwargs):
        pass
        return None




