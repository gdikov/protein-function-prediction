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
        targets_ints_21 = T.ivector('targets21')
        targets_ints_24 = T.ivector('targets24')

        # create a one-hot encoding if an integer vector
        # using a broadcast trick (targets is reshaped from (N) to (N, 1)
        # and then each entry is compared to [1, 2, ... K] in a broadcast
        targets_21 = T.eq(targets_ints_21.reshape((-1, 1)), T.arange(2))
        targets_24 = T.eq(targets_ints_24.reshape((-1, 1)), T.arange(2))

        # build the network architecture
        self.out1, self.out2 = self._build_network(mol_indices=mol_indices)

        # define objective and training parameters
        train_predictions_21 = lasagne.layers.get_output(self.out1)
        train_predictions_24 = lasagne.layers.get_output(self.out2)

        def categorical_crossentropy_logdomain(log_predictions, targets):
            return -T.sum(targets * log_predictions, axis=1)

        train_loss_21 = categorical_crossentropy_logdomain(log_predictions=train_predictions_21,
                                                           targets=targets_21).mean()
        train_loss_24 = categorical_crossentropy_logdomain(log_predictions=train_predictions_24,
                                                           targets=targets_24).mean()

        train_params = lasagne.layers.get_all_params([self.out1, self.out2], trainable=True)
        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss_21 + train_loss_24,
                                                    params=train_params,
                                                    learning_rate=1e-3)

        train_accuracy_21 = T.mean(T.eq(T.argmax(train_predictions_21, axis=-1), targets_ints_21),
                                   dtype=theano.config.floatX)
        train_accuracy_24 = T.mean(T.eq(T.argmax(train_predictions_24, axis=-1), targets_ints_24),
                                   dtype=theano.config.floatX)

        val_predictions_21 = lasagne.layers.get_output(self.out1, deterministic=True)
        val_predictions_24 = lasagne.layers.get_output(self.out2, deterministic=True)

        val_loss_21 = categorical_crossentropy_logdomain(log_predictions=val_predictions_21,
                                                         targets=targets_21).mean()
        val_loss_24 = categorical_crossentropy_logdomain(log_predictions=val_predictions_24,
                                                         targets=targets_24).mean()

        val_accuracy_21 = T.mean(T.eq(T.argmax(val_predictions_21, axis=-1), targets_ints_21),
                                 dtype=theano.config.floatX)

        val_accuracy_24 = T.mean(T.eq(T.argmax(val_predictions_24, axis=-1), targets_ints_24),
                                 dtype=theano.config.floatX)

        self.train_function = theano.function(inputs=[mol_indices, targets_ints_21, targets_ints_24],
                                              outputs=[train_loss_21, train_loss_24, train_accuracy_21,
                                                       train_accuracy_24, train_predictions_21, targets_21],
                                              updates=train_params_updates)

        self.validation_function = theano.function(inputs=[mol_indices, targets_ints_21, targets_ints_24],
                                                   outputs=[val_loss_21, val_loss_24, val_accuracy_21, val_accuracy_24])

        self._get_params = theano.function(inputs=[], outputs=train_params)

        self._get_all_outputs = theano.function(inputs=[mol_indices], outputs=lasagne.layers.get_output(
            lasagne.layers.get_all_layers([self.out1])))

        # save training history data
        self.history = {'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}
        print("INFO: Computational graph compiled")

    def _build_network(self, mol_indices):
        indices_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,), input_var=mol_indices)
        data_gen = MoleculeMapLayer(incoming=indices_input, minibatch_size=self.minibatch_size)

        network = data_gen  # lasagne.layers.BatchNormLayer(incoming=data_gen)
        for i in range(0, 6):
            filter_size = (3, 3, 3)
            network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                        num_filters=2 ** (2 + i), filter_size=filter_size,
                                                        nonlinearity=lasagne.nonlinearities.leaky_rectify)
            network = lasagne.layers.DropoutLayer(incoming=network)
            if i % 3 == 2:
                network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network, pool_size=(2, 2, 2), stride=2)

        network1 = network
        network2 = network

        for i in range(0, 6):
            network1 = lasagne.layers.DenseLayer(incoming=network1, num_units=64,
                                                 nonlinearity=lasagne.nonlinearities.leaky_rectify)
            network2 = lasagne.layers.DenseLayer(incoming=network2, num_units=64,
                                                 nonlinearity=lasagne.nonlinearities.leaky_rectify)

        output_layer1 = lasagne.layers.DenseLayer(incoming=network1, num_units=2,
                                                  nonlinearity=T.nnet.logsoftmax)
        output_layer2 = lasagne.layers.DenseLayer(incoming=network2, num_units=2,
                                                  nonlinearity=T.nnet.logsoftmax)

        return output_layer1, output_layer2

    def _iter_minibatches_train(self):
        minibatch_count = self.train_data_size / self.minibatch_size
        y = self.data["y_train"]
        unique_labels = np.vstack({tuple(row) for row in y})

        for minibatch_index in xrange(0, minibatch_count):
            # the following collects the indices in the `y_train` array
            # which correspond to different labels
            label_buckets = [np.nonzero(np.all(y == unique_label, axis=1)) for unique_label in unique_labels]
            next_indices = []
            for i in xrange(0, self.minibatch_size):
                random_bucket = label_buckets[np.random.randint(0, len(unique_labels))][0]
                next_indices.append(np.random.choice(random_bucket))
            yield np.array(next_indices, dtype=np.int32)

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
            for indices in self._iter_minibatches_train():
                y = self.data['y_train'][indices]
                loss21, loss24, acc21, acc24, pred, tgt = self.train_function(indices, y[:, 0], y[:, 1])
                losses21.append(loss21)
                losses24.append(loss24)
                accs21.append(acc21)
                accs24.append(acc24)
                # print("INFO: train: loss21: %f loss24 %f acc21: %f acc24: %f" %
                #       (loss21, loss24, acc21, acc24))
                # outputs = self._get_all_outputs(indices)
                continue

            mean_loss21 = np.mean(np.array(losses21))
            mean_loss24 = np.mean(np.array(losses24))
            mean_acc21 = np.mean(np.array(accs21))
            mean_acc24 = np.mean(np.array(accs24))
            print("INFO: train: epoch %d loss21: %f loss24 %f acc21: %f acc24: %f" %
                  (e, mean_loss21, mean_loss24, mean_acc21, mean_acc24))
            if np.isnan(mean_loss21) or np.isnan(mean_loss24):
                params = [np.array(param) for param in self._get_params()]
                print(params)

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
