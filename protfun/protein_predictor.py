import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers.dnn

from protfun.layers.molmap_layer import MoleculeMapLayer


class ProteinPredictor(object):
    def __init__(self, data, minibatch_size=2):

        self.minibatch_size = minibatch_size
        # self.num_output_classes = data['y_id2name'].shape[0]

        self.data = data

        # the input has the shape of the X_train portion of the dataset
        self.train_data_size = self.data['y_train'].shape[0]
        self.val_data_size = self.data['y_val'].shape[0]
        self.test_data_size = self.data['y_test'].shape[0]

        # define input and output symbolic variables of the computation graph
        mol_indices = T.ivector("molecule_indices")
        targets_21 = T.dvector('targets21')
        targets_24 = T.dvector('targets24')

        # build the network architecture
        self.out1, self.out2 = self._build_network(mol_indices=mol_indices)

        # define objective and training parameters
        train_predictions_21 = lasagne.layers.get_output(self.out1)
        train_predictions_24 = lasagne.layers.get_output(self.out2)

        train_loss_21 = lasagne.objectives.binary_crossentropy(predictions=train_predictions_21,
                                                               targets=targets_21).mean()
        train_loss_24 = lasagne.objectives.binary_crossentropy(predictions=train_predictions_24,
                                                               targets=targets_24).mean()

        train_params = lasagne.layers.get_all_params([self.out1, self.out2], trainable=True)
        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss_21 + train_loss_24,
                                                    params=train_params,
                                                    learning_rate=1e-3)

        train_accuracy_21 = T.mean(T.eq(T.gt(train_predictions_21, 0.5), targets_21), dtype=theano.config.floatX)
        train_accuracy_24 = T.mean(T.eq(T.gt(train_predictions_24, 0.5), targets_24), dtype=theano.config.floatX)

        val_predictions_21 = lasagne.layers.get_output(self.out1, deterministic=True)
        val_predictions_24 = lasagne.layers.get_output(self.out2, deterministic=True)

        val_loss_21 = lasagne.objectives.binary_crossentropy(predictions=val_predictions_21,
                                                             targets=targets_21).mean()
        val_loss_24 = lasagne.objectives.binary_crossentropy(predictions=val_predictions_24,
                                                             targets=targets_24).mean()

        val_accuracy_21 = T.mean(T.eq(T.gt(val_predictions_21, 0.5), targets_21), dtype=theano.config.floatX)
        val_accuracy_24 = T.mean(T.eq(T.gt(val_predictions_24, 0.5), targets_24), dtype=theano.config.floatX)

        self.train_function = theano.function(inputs=[mol_indices, targets_21, targets_24],
                                              outputs=[train_loss_21, train_loss_24, train_accuracy_21,
                                                       train_accuracy_24],
                                              updates=train_params_updates)

        self.validation_function = theano.function(inputs=[mol_indices, targets_21, targets_24],
                                                   outputs=[val_loss_21, val_loss_24, val_accuracy_21, val_accuracy_24])

        self._get_params = theano.function(inputs=[], outputs=train_params)

        # save training history data
        self.history = {'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}
        print("INFO: Computational graph compiled")

    def _build_network(self, mol_indices):
        indices_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,), input_var=mol_indices)
        data_gen = MoleculeMapLayer(incoming=indices_input,
                                    minibatch_size=self.minibatch_size)

        network = data_gen
        for i in range(0, 3):
            filter_size = (3 - i // 2,) * 3
            network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network,
                                                        num_filters=2 ^ (3 + i), filter_size=filter_size,
                                                        nonlinearity=lasagne.nonlinearities.rectify)
            # W=lasagne.init.GlorotNormal())
            if i % 2 == 0:
                network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network, pool_size=(2, 2, 2))

        dense1 = lasagne.layers.DenseLayer(incoming=network, num_units=32)
        dense2 = lasagne.layers.DenseLayer(incoming=network, num_units=32)

        output_layer1 = lasagne.layers.DenseLayer(incoming=dense1, num_units=1,
                                                  nonlinearity=lasagne.nonlinearities.sigmoid)
        output_layer2 = lasagne.layers.DenseLayer(incoming=dense2, num_units=1,
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
            losses21 = []
            losses24 = []
            accs21 = []
            accs24 = []
            for indices in self._iter_minibatches(self.train_data_size):
                y = self.data['y_train'][indices]
                loss21, loss24, acc21, acc24 = self.train_function(indices, y[:, 0], y[:, 1])
                losses21.append(loss21)
                losses24.append(loss24)
                accs21.append(acc21)
                accs24.append(acc24)
                # print("INFO: train: loss21: %f loss24 %f acc21: %f acc24: %f" %
                #       (loss21, loss24, acc21, acc24))
            mean_loss21 = np.mean(np.array(losses21))
            mean_loss24 = np.mean(np.array(losses24))
            mean_acc21 = np.mean(np.array(accs21))
            mean_acc24 = np.mean(np.array(accs24))
            print("INFO: train: epoch %d loss21: %f loss24 %f acc21: %f acc24: %f" %
                  (e, mean_loss21, mean_loss24, mean_acc21, mean_acc24))
            if np.isnan(mean_loss21) or np.isnan(mean_loss24):
                params = [param.eval() for param in self._get_params()]
                print(params)

                # # validate
                # losses21 = []
                # losses24 = []
                # accs = []
                # for indices in self._iter_minibatches(self.val_data_size, shuffle=False):
                #     y = self.data['y_val'][indices]
                #     loss21, loss24, acc = self.validation_function(indices, y)
                #     losses21.append(loss21)
                #     losses24.append(loss24)
                #     accs.append(acc)
                #
                # mean_loss21 = np.mean(np.array(losses21))
                # mean_loss24 = np.mean(np.array(losses24))
                # mean_accuracy = np.mean(np.array(accs))
                # print("INFO: epoch %d val loss21: %f val loss24 %f val accuracy: %f" %
                #       (e, mean_loss21, mean_loss24, mean_accuracy))
                # self.history['val_loss'].append((mean_loss21, mean_loss24))
                # self.history['val_accuracy'].append(mean_accuracy)

    def test(self):
        print("INFO: Testing...")
        losses21 = []
        losses24 = []
        accs21 = []
        accs24 = []
        for indices in self._iter_minibatches(self.test_data_size):
            y = self.data['y_test'][indices]
            loss21, loss24, acc21, acc24 = self.validation_function(indices, y[:, 0], y[:, 1])
            losses21.append(loss21)
            losses24.append(loss24)
            accs21.append(acc21)
            accs24.append(acc24)

        mean_loss21 = np.mean(np.array(losses21))
        mean_loss24 = np.mean(np.array(losses24))
        mean_acc21 = np.mean(np.array(accs21))
        mean_acc24 = np.mean(np.array(accs24))
        print("INFO: test: loss21: %f loss24 %f acc21: %f acc24: %f" %
              (mean_loss21, mean_loss24, mean_acc21, mean_acc24))

        return mean_loss21, mean_loss24, mean_acc21, mean_acc24

    def summarize(self):
        print("The network has been tremendously successful!")
