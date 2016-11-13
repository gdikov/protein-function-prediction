import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers.dnn

from protfun.layers.molmap_layer import MoleculeMapLayer


class ProteinPredictor(object):
    def __init__(self, train_data, test_data, minibatch_size=1, num_output_classes=1):

        self.minibatch_size = minibatch_size
        self.num_output_classes = num_output_classes

        self.train_data = train_data
        self.test_data = test_data

        # the input has the shape of the X_train portion of the dataset
        self.train_data_size = self.train_data['y_train'].shape[0]
        self.val_data_size = self.train_data['y_val'].shape[0]
        self.output_shape = self.train_data['y_train'].shape[1:]
        self.output_shape = tuple([minibatch_size]) + self.output_shape

        # define input and output symbolic variables of the computation graph
        mol_indices = T.ivector("molecule_indices")
        targets = T.dvector('targets')

        # build the network architecture
        self.network = self._build_network(mol_indices=mol_indices)

        # define objective and training parameters
        train_predictions = lasagne.layers.get_output(self.network)
        train_loss = lasagne.objectives.binary_crossentropy(predictions=train_predictions,
                                                            targets=targets).mean()

        train_params = lasagne.layers.get_all_params(self.network, trainable=True)
        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss, params=train_params,
                                                    learning_rate=1e-6)

        train_accuracy = T.mean(T.eq(T.gt(train_predictions, 0.5), targets),
                                dtype=theano.config.floatX)

        val_predictions = lasagne.layers.get_output(self.network, deterministic=True)
        val_loss = lasagne.objectives.binary_crossentropy(predictions=val_predictions,
                                                          targets=targets).mean()

        val_accuracy = T.mean(T.eq(T.gt(val_predictions, 0.5), targets),
                              dtype=theano.config.floatX)

        self.train_function = theano.function(inputs=[mol_indices, targets],
                                              outputs=[train_loss, train_accuracy],
                                              updates=train_params_updates)

        self.validation_function = theano.function(inputs=[mol_indices, targets],
                                                   outputs=[val_loss, val_accuracy])

        # save training history data
        self.history = {'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}
        print("INFO: Computational graph compiled")

    def _build_network(self, mol_indices):
        indices_input = lasagne.layers.InputLayer(shape=(None,), input_var=mol_indices)
        data_gen = MoleculeMapLayer(incoming=indices_input,
                                    minibatch_size=self.minibatch_size)
        network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=data_gen,
                                                    num_filters=16, filter_size=(5, 5, 5),
                                                    nonlinearity=lasagne.nonlinearities.rectify,
                                                    W=lasagne.init.GlorotNormal())
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network, pool_size=(10, 10, 10))
        network = lasagne.layers.DenseLayer(incoming=network, num_units=10)
        network = lasagne.layers.DenseLayer(incoming=network, num_units=self.num_output_classes,
                                            nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    def _iter_minibatches(self, data_size, shuffle=True):
        minibatch_count = data_size / self.minibatch_size
        if data_size % self.minibatch_size != 0:
            minibatch_count += 1

        if shuffle:
            order = np.random.permutation(data_size)
        else:
            order = np.array(xrange(0, data_size))

        for minibatch_index in xrange(0, minibatch_count):
            mask = order[minibatch_index:minibatch_index + self.minibatch_size]
            yield np.asarray(mask, dtype=np.int32)

    def train(self, epoch_count=10):
        print("INFO: Training...")
        for e in xrange(epoch_count):
            for indices in self._iter_minibatches(self.train_data_size):
                y = self.labels_data['y_train'][indices]
                self.train_function(indices, y)

            # validate
            losses = []
            accs = []
            for indices in self._iter_minibatches(self.val_data_size, shuffle=False):
                y = self.labels_data['y_val'][indices]
                loss, acc = self.validation_function(indices, y)
                losses.append(loss)
                accs.append(acc)

            mean_loss = np.mean(np.array(losses))
            mean_accuracy = np.mean(np.array(accs))
            print("INFO: epoch %d val loss: %f val accuracy: %f" % (e, mean_loss, mean_accuracy))
            self.history['val_loss'].append(mean_loss)
            self.history['val_accuracy'].append(mean_accuracy)

    def test(self):
        print("INFO: Testing...")
        loss = list()
        acc = list()
        for indices in self._iter_minibatches(self.train_data_size):
            y = self.labels_data['y_test'][indices]
            l, a = self.validation_function(indices, y)
            loss.append(l)
            acc.append(a)

        mean_loss = sum(loss) / float(len(loss))
        mean_acc = sum(acc) / float(len(acc))

        print("Mean test loss: {0}".format(mean_loss))
        print("Mean test accuracy: {0}".format(mean_acc))

        return mean_loss, mean_acc

    def summarize(self):
        print("The network has been tremendously successful!")