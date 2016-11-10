import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn

import data_prep as dp

class NitroCounter():

    def __init__(self, minibatch_size=16):
        self.minibatch_size = minibatch_size

        # dictionary with keywords "x_train", "y_train", "x_val", "y_val", "x_test", "y_test"
        self.data = dp.load_dataset("computed_grid.npy")

        # the input has the shape of the X_train portion of the dataset
        self.input_shape = self.data['x_train'].shape[1:]
        # set the minibatch size in the input dimension
        self.input_shape = tuple([minibatch_size]) + self.input_shape
        self.output_shape = self.data['y_train'].shape

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

        # save history data
        self.history = {'train_loss': list(),
                        'train_accuracy': list(),
                        'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}
        print("INFO: Computational graph compiled")
        print(self.input_shape)
        print(self.output_shape)

    def _build_network(self, input_tensor_var=None):
        input_layer = lasagne.layers.InputLayer(shape=self.input_shape, input_var=input_tensor_var)
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=input_layer,
                                                    num_filters=16, filter_size=(5, 5, 5),
                                                    nonlinearity=lasagne.nonlinearities.rectify,
                                                    W=lasagne.init.GlorotNormal())
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network, pool_size=(10, 10, 10))
        network = lasagne.layers.DenseLayer(incoming=network,num_units=10)
        network = lasagne.layers.DenseLayer(incoming=network, num_units=1,
                                            nonlinearity=lasagne.nonlinearities.softmax)

        return network

    def _iter_minibatches(self, xs, ys, shuffle=True):
        data_size = xs.shape[0]
        minibatch_count = data_size / self.minibatch_size
        if data_size % self.minibatch_size != 0:
            minibatch_count += 1

        order = np.random.permutation(data_size)
        for minibatch_index in xrange(0, minibatch_count):
            mask = order[minibatch_index:minibatch_index+self.minibatch_size]
            yield xs[mask], ys[mask]

    def train(self, epoch_count=1):
        print("INFO: Training...")
        for e in xrange(epoch_count):
            for minibatch in self._iter_minibatches(self.data['x_train'], self.data['y_train']):
                x, y = minibatch
                loss, acc = self.train_function(x, y)
                self.history['train_loss'].append(loss)
                self.history['train_accuracy'].append(acc)

        print("INFO: Validating...")
        for minibatch in self._iter_minibatches(self.data['x_val'], self.data['y_val']):
            x, y = minibatch
            loss, acc = self.validation_function(x, y)
            self.history['val_loss'].append(loss)
            self.history['val_accuracy'].append(acc)

    def test(self):
        print("INFO: Testing...")
        loss = list(); acc = list()
        for minibatch in self._iter_minibatches(self.data['x_test'], self.data['y_test']):
            x, y = minibatch
            l, a = self.validation_function(x, y)
            loss.append(l); acc.append(a)

        mean_loss = sum(loss)/float(len(loss))
        mean_acc = sum(acc)/float(len(acc))

        print("Mean test loss: {0}".format(mean_loss))
        print("Mean test accuracy: {0}".format(mean_acc))

        return mean_loss, mean_acc