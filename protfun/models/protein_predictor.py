import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers.dnn

from protfun.layers.molmap_layer import MoleculeMapLayer
from protfun.models.model_monitor import ModelMonitor


class ProteinPredictor(object):
    def __init__(self, data, minibatch_size=1, model_name='model'):

        self.minibatch_size = minibatch_size
        # self.num_output_classes = data['y_id2name'].shape[0]

        self.data = data

        # the input has the shape of the X_train portion of the dataset
        self.train_data_size = self.data['y_train'].shape[0]
        self.val_data_size = self.data['y_val'].shape[0]
        self.test_data_size = self.data['y_test'].shape[0]

        # define input and output symbolic variables of the computation graph
        mol_indices = T.ivector("molecule_indices")

        # TODO: replace all lists with for loops

        targets_ints = [T.ivector('targets21'), T.ivector('targets24')]

        # create a one-hot encoding if an integer vector
        # using a broadcast trick (targets is reshaped from (N) to (N, 1)
        # and then each entry is compared to [1, 2, ... K] in a broadcast
        targets = [T.eq(targets_ints[0].reshape((-1, 1)), T.arange(2)),
                   T.eq(targets_ints[1].reshape((-1, 1)), T.arange(2))]

        # build the network architecture
        self.outs = self._build_network(mol_indices=mol_indices)

        # define objective and training parameters
        train_predictions = [lasagne.layers.get_output(self.outs[0]),
                             lasagne.layers.get_output(self.outs[1])]

        def categorical_crossentropy_logdomain(log_predictions, targets):
            return -T.sum(targets * log_predictions, axis=1)

        train_losses = [categorical_crossentropy_logdomain(log_predictions=train_predictions[0],
                                                           targets=targets[0]).mean(),
                        categorical_crossentropy_logdomain(log_predictions=train_predictions[1],
                                                           targets=targets[1]).mean()]

        train_params = lasagne.layers.get_all_params([self.outs[0], self.outs[1]], trainable=True)
        train_params_updates = lasagne.updates.adam(loss_or_grads=train_losses[0] + train_losses[1],
                                                    params=train_params,
                                                    learning_rate=1e-2)

        train_accuracies = [T.mean(T.eq(T.argmax(train_predictions[0], axis=-1), targets_ints[0]),
                                   dtype=theano.config.floatX),
                            T.mean(T.eq(T.argmax(train_predictions[1], axis=-1), targets_ints[1]),
                                   dtype=theano.config.floatX)]

        val_predictions = [lasagne.layers.get_output(self.outs[0], deterministic=True),
                           lasagne.layers.get_output(self.outs[1], deterministic=True)]

        val_losses = [categorical_crossentropy_logdomain(log_predictions=val_predictions[0],
                                                         targets=targets[0]).mean(),
                      categorical_crossentropy_logdomain(log_predictions=val_predictions[1],
                                                         targets=targets[1]).mean()]

        val_accuracies = [T.mean(T.eq(T.argmax(val_predictions[0], axis=-1), targets_ints[0]),
                                 dtype=theano.config.floatX),
                          T.mean(T.eq(T.argmax(val_predictions[1], axis=-1), targets_ints[1]),
                                 dtype=theano.config.floatX)]

        self.train_function = theano.function(inputs=[mol_indices, targets_ints[0], targets_ints[1]],
                                              outputs=[train_losses[0], train_losses[1], train_accuracies[0],
                                                       train_accuracies[1], train_predictions[0], targets[0]],
                                              updates=train_params_updates)

        self.validation_function = theano.function(inputs=[mol_indices, targets_ints[0], targets_ints[1]],
                                                   outputs=[val_losses[0], val_losses[1],
                                                            val_accuracies[0], val_accuracies[1]])

        self._get_params = theano.function(inputs=[], outputs=train_params)

        self._get_all_outputs = theano.function(inputs=[mol_indices], outputs=lasagne.layers.get_output(
            lasagne.layers.get_all_layers([self.outs[0]])))

        # save training history data
        self.history = {'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoche': list()}
        print("INFO: Computational graph compiled")

        self.monitor = ModelMonitor(self.outs, name=model_name)


    def _build_network(self, mol_indices):
        indices_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,), input_var=mol_indices)
        data_gen = MoleculeMapLayer(incoming=indices_input, minibatch_size=self.minibatch_size)

        network = data_gen  # lasagne.layers.BatchNormLayer(incoming=data_gen)
        for i in range(0, 1):
            filter_size = (3, 3, 3)
            # NOTE: we start with a very poor filter count.
            network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                        num_filters=2 ** (2 + i), filter_size=filter_size,
                                                        nonlinearity=lasagne.nonlinearities.leaky_rectify)
            if i % 3 == 2:
                network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network, pool_size=(2, 2, 2), stride=2)

        network1 = network
        network2 = network

        # NOTE: for just 2 molecules, having 1 deep layer speeds up the
        # required training time from 20 epochs to 5 epochs
        for i in range(0, 1):
            network1 = lasagne.layers.DenseLayer(incoming=network1, num_units=8,
                                                 nonlinearity=lasagne.nonlinearities.leaky_rectify)
            network2 = lasagne.layers.DenseLayer(incoming=network2, num_units=8,
                                                 nonlinearity=lasagne.nonlinearities.leaky_rectify)

        output_layer1 = lasagne.layers.DenseLayer(incoming=network1, num_units=2,
                                                  nonlinearity=T.nnet.logsoftmax)
        output_layer2 = lasagne.layers.DenseLayer(incoming=network2, num_units=2,
                                                  nonlinearity=T.nnet.logsoftmax)

        return output_layer1, output_layer2


    def _iter_minibatches(self, mode='train'):
        data_size = self.data['y_'+mode].shape[0]
        minibatch_count = data_size / self.minibatch_size
        if data_size % self.minibatch_size != 0:
            minibatch_count += 1

        ys = self.data['y_'+mode]
        # one hot encoding of labels which are present in the current set of samples
        num_classes = self.data['class_distribution_'+mode].shape[0]
        represented_classes = np.arange(num_classes)[self.data['class_distribution_'+mode] > 0.]

        if represented_classes.shape[0] < num_classes:
            print("WARRNING: Non-exhaustive {0}ing. Class (Classes) {1} is (are) not represented".
                  format(mode, np.arange(num_classes)[self.data['class_distribution_'+mode] <= 0.]))

        unique_labels = np.eye(represented_classes.shape[0])

        for minibatch_index in xrange(0, minibatch_count):
            # the following collects the indices in the `y_train` array
            # which correspond to different labels
            label_buckets = [np.nonzero(np.all(ys == label, axis=1)) for label in unique_labels]
            bucket_ids = np.random.choice(represented_classes, size=self.minibatch_size)
            next_indices = [label_buckets[i][0][np.random.randint(0, len(label_buckets[i][0]))]
                            for i in bucket_ids]
            yield np.array(next_indices, dtype=np.int32)

    def train(self, epoch_count=10):
        print("INFO: Training...")
        for e in xrange(epoch_count):
            losses = []
            accs = []
            for indices in self._iter_minibatches(mode='train'):
                y = self.data['y_train'][indices]
                import time
                start = time.time()
                loss21, loss24, acc21, acc24, pred, tgt = self.train_function(indices, y[:, 0], y[:, 1])
                losses.append((loss21, loss24))
                accs.append((acc21, acc24))
                # print("INFO: train: loss21: %f loss24 %f acc21: %f acc24: %f" %
                #       (loss21, loss24, acc21, acc24))
                # outputs = self._get_all_outputs(indices)

            mean_losses = np.mean(np.array(losses), axis=0)
            mean_accs = np.mean(np.array(accs), axis=0)
            print("INFO: train: epoch %d loss21: %f loss24 %f acc21: %f acc24: %f" %
                  (e+1, mean_losses[0], mean_losses[1], mean_accs[0], mean_accs[1]))
            if np.isnan(mean_losses[0]) or np.isnan(mean_losses[1]):
                print("WARNING: Something went wrong during trainig. Saving parameters...")
                self.monitor.save_model(e, "nans_during_trainig")

            # implement a better logic here
            if e-1 % 10 == 0:
                self.monitor.save_model(e)

    def test(self):
        print("INFO: Testing...")
        losses = []
        accs = []
        for indices in self._iter_minibatches(mode='test'):
            y = self.data['y_test'][indices]
            loss21, loss24, acc21, acc24 = self.validation_function(indices, y[:, 0], y[:, 1])
            losses.append((loss21, loss24))
            accs.append((acc21, acc24))

        mean_losses = np.mean(np.array(losses), axis=0)
        mean_accs = np.mean(np.array(accs), axis=0)
        print("INFO: test: loss21: %f loss24 %f acc21: %f acc24: %f" %
              (mean_losses[0], mean_losses[1], mean_accs[0], mean_accs[1]))

        return mean_losses[0], mean_losses[1], mean_accs[0], mean_accs[1]

    def summarize(self):
        print("The network has been tremendously successful!")
