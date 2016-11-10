import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import theano.tensor.nlinalg

from os import path, listdir


# NOTE: One would need a custom 3D ConvLayer only if CuDNN is not supported.
# Otherwise see lasagne.layers.dnn.Conv3DDNNLayer

class Conv3DLayer(lasagne.layers.BaseConvLayer):
    """
    TODO: This lasagne layer should apply a 3-dimensional convolution

    see documentation of BaseConvLayer
    """

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1), pad=0,
                 untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
                 convolution=theano.tensor.nnet.conv3d, **kwargs):

        super(Conv3DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=3,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved
