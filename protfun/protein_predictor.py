import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers.dnn

from protfun.layers.molmap_layer import MoleculeMapLayer


class ProteinPredictor(object):
    def __init__(self, data, minibatch_size=1):

        self.minibatch_size = minibatch_size
        # self.num_output_classes = data['y_id2name'].shape[0]

        self.data = data

        # the input has the shape of the X_train portion of the dataset
        self.train_data_size = self.data['y_train'].shape[0]
        self.val_data_size = self.data['y_val'].shape[0]
        self.test_data_size = self.data['y_test'].shape[0]

        # define input and output symbolic variables of the computation graph
        mol_indices = T.ivector("molecule_indices")
        targets = T.dmatrix('targets')

        # build the network architecture
        self.out1, self.out2 = self._build_network(mol_indices=mol_indices)

        # define objective and training parameters
        train_predictions_21 = lasagne.layers.get_output(self.out1)
        train_predictions_24 = lasagne.layers.get_output(self.out2)

        train_loss_21 = lasagne.objectives.binary_crossentropy(predictions=train_predictions_21,
                                                               targets=targets[:, 0]).mean()
        train_loss_24 = lasagne.objectives.binary_crossentropy(predictions=train_predictions_24,
                                                               targets=targets[: 1]).mean()

        train_params = lasagne.layers.get_all_params([self.out1, self.out2], trainable=True)
        train_params_updates = lasagne.updates.adam(loss_or_grads=[train_loss_21, train_loss_24], params=train_params,
                                                    learning_rate=1e-6)

        train_accuracy = T.mean(
            T.eq(T.gt(T.concatenate(train_predictions_21, train_predictions_24, axis=-1), 0.5), targets),
            dtype=theano.config.floatX)

        val_predictions_21 = lasagne.layers.get_output(self.out1, deterministic=True)
        val_predictions_24 = lasagne.layers.get_output(self.out2, deterministic=True)
        val_loss_21 = lasagne.objectives.binary_crossentropy(predictions=val_predictions_21,
                                                             targets=targets[:, 0]).mean()
        val_loss_24 = lasagne.objectives.binary_crossentropy(predictions=val_predictions_24,
                                                             targets=targets[:, 1]).mean()

        val_accuracy = T.mean(
            T.eq(T.gt(T.concatenate(val_predictions_21, train_predictions_24, axis=-1), 0.5), targets),
            dtype=theano.config.floatX)

        self.train_function = theano.function(inputs=[mol_indices, targets],
                                              outputs=[train_loss_21, train_loss_24, train_accuracy],
                                              updates=train_params_updates)

        self.validation_function = theano.function(inputs=[mol_indices, targets],
                                                   outputs=[val_loss_21, val_loss_24, val_accuracy])

        # save training history data
        self.history = {'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}
        print("INFO: Computational graph compiled")

    def _build_network(self, mol_indices):
        indices_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,), input_var=mol_indices)
        data_gen = MoleculeMapLayer(incoming=indices_input,
                                    minibatch_size=self.minibatch_size)
        conv3d = lasagne.layers.dnn.Conv3DDNNLayer(incoming=data_gen,
                                                   num_filters=16, filter_size=(5, 5, 5),
                                                   nonlinearity=lasagne.nonlinearities.rectify,
                                                   W=lasagne.init.GlorotNormal())
        maxpool3d = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=conv3d, pool_size=(10, 10, 10))
        dense = lasagne.layers.DenseLayer(incoming=maxpool3d, num_units=10)
        output_layer1 = lasagne.layers.DenseLayer(incoming=dense, num_units=1,
                                                  nonlinearity=lasagne.nonlinearities.sigmoid)
        output_layer2 = lasagne.layers.DenseLayer(incoming=dense, num_units=1,
                                                  nonlinearity=lasagne.nonlinearities.sigmoid)

        return output_layer1, output_layer2

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
                y = self.data['y_train'][indices]
                self.train_function(indices, y)

            # validate
            losses21 = losses24 = []
            accs = []
            for indices in self._iter_minibatches(self.val_data_size, shuffle=False):
                y = self.data['y_val'][indices]
                loss21, loss24, acc = self.validation_function(indices, y)
                losses21.append(loss21)
                losses24.append(loss24)
                accs.append(acc)

            mean_loss21 = np.mean(np.array(losses21))
            mean_loss24 = np.mean(np.array(losses24))
            mean_accuracy = np.mean(np.array(accs))
            print("INFO: epoch %d val loss21: %f val loss24 %f val accuracy: %f" %
                  (e, mean_loss21, mean_loss24, mean_accuracy))
            self.history['val_loss'].append((mean_loss21, mean_loss24))
            self.history['val_accuracy'].append(mean_accuracy)

    def test(self):
        print("INFO: Testing...")
        losses21 = losses24 = []
        acc = list()
        for indices in self._iter_minibatches(self.test_data_size):
            y = self.data['y_test'][indices]
            loss21, loss24, a = self.validation_function(indices, y)
            losses21.append(loss21)
            losses24.append(loss24)
            acc.append(a)

        mean_loss21 = np.mean(np.array(losses21))
        mean_loss24 = np.mean(np.array(losses24))
        mean_accuracy = np.mean(np.array(acc))
        print("INFO: epoch %d test loss21: %f test loss24 %f test accuracy: %f" %
              (mean_loss21, mean_loss24, mean_accuracy))

        return mean_loss21, mean_loss24, mean_accuracy

    def summarize(self):
        print("The network has been tremendously successful!")
