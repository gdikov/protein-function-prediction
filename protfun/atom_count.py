import numpy as np
import theano
import theano.tensor as T

import lasagne

import protfun.layers as custom

class NitroCounter():

    def __init__(self, minibatch_size=16):
        self.minibatch_size = minibatch_size

        # dictionary with keywords "x_train", "y_train", "x_val", "y_val", "x_test", "y_test"
        self.data = self._load_dataset()

        # the input has the shape of the X_train portion of the dataset
        self.input_shape = self.data['x_train'].shape
        # set the minibatch size in the input dimension
        self.input_shape[0] = minibatch_size
        self.output_shape = self.data['y_train'].shape[1:]

        # define input and output symbolic variables of the computation graph
        input_tensor_var = T.tensor5('inputs')
        target_tensor_var = T.ivector('targets')

        # build the network architecture
        self.network = self._build_network(input_tensor_var)

        # define objective and training parameters
        train_predictions = lasagne.layers.get_output(self.network)
        train_loss = lasagne.objectives.categorical_crossentropy(predictions=train_predictions,
                                                           targets=target_tensor_var).mean()

        train_params = lasagne.layers.get_all_params(self.network, trainable=True)
        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss, params=train_params,
                                              learning_rate=1e-3)

        train_accuracy = T.mean(T.eq(T.argmax(train_predictions, axis=1), target_tensor_var),
                               dtype=theano.config.floatX)

        val_predictions = lasagne.layers.get_output(self.network, deterministic=True)
        val_loss = lasagne.objectives.categorical_crossentropy(predictions=val_predictions,
                                                                targets=target_tensor_var).mean()

        val_accuracy = T.mean(T.eq(T.argmax(val_predictions, axis=1), target_tensor_var),
                               dtype=theano.config.floatX)

        self.train_function = theano.function([input_tensor_var, target_tensor_var],
                                              [train_loss, train_accuracy],
                                              updates=train_params_updates)
        self.validation_function = theano.function([input_tensor_var, target_tensor_var],
                                                   [val_loss, val_accuracy])

        # monitoring variables
        self.history = {'train_loss': list(),
                        'train_accuracy': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}

    def _load_dataset(self):
        # TODO: create labeled data from PDB parsed samples
        data = {'x_train': None, 'y_train': None,
                'x_val': None, 'y_val': None,
                'x_test': None, 'y_test': None}
        return data

    def _build_network(self, input_tensor_var=None):
        input_layer = lasagne.layers.InputLayer(shape=self.input_shape, input_var=input_tensor_var)
        network = custom.Conv3DLayer(incoming=input_layer,
                                  num_filters=16, filter_size=(5, 5, 5),
                                  nonlinearity=lasagne.nonlinearities.rectify,
                                  W=lasagne.init.GlorotNormal())
        network = custom.MaxPool3DLayer(incoming=network, pool_size=(10,10,10))
        network = lasagne.layers.DenseLayer(incoming=network,num_units=10)
        network = lasagne.layers.DenseLayer(incoming=network, num_units=1,
                                            nonlinearity=lasagne.nonlinearities.softmax)

        return network

    def _iter_minibatches(self, xs, ys):
        yield None

    def train(self, epoch_count=1):
        for e in xrange(epoch_count):
            for minibatch in self._iter_minibatches(self.data['x_train'], self.data['y_train']):
                x, y = minibatch
                loss, acc = self.train_function([x, y])
                self.history['train_loss'].append(loss)
                self.history['train_accuracy'].append(acc)

        for minibatch in self._iter_minibatches(self.data['x_val'], self.data['y_val']):
            x, y = minibatch
            loss, acc = self.validation_function([x, y])
            self.history['val_loss'].append(loss)
            self.history['val_accuracy'].append(acc)
