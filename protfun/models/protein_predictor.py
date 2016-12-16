import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers.dnn
import colorlog as log
import logging
import threading
from os import path
from protfun.visualizer.progressview import ProgressView

from protfun.layers.molmap_layer import MoleculeMapLayer
from protfun.models.model_monitor import ModelMonitor

log.basicConfig(level=logging.DEBUG)

floatX = theano.config.floatX
intX = np.int32  # FIXME is this the best choice? (changing would require removing and recreating memmap files)


class ProteinPredictor(object):
    def __init__(self, data,
                 minibatch_size=1, initial_per_class_datasize=100,
                 model_name='protein_predictor'):

        self.minibatch_size = minibatch_size
        self.initial_per_class_datasize = initial_per_class_datasize

        self.data = data

        # the input has the shape of the X_train portion of the dataset
        self.train_data_size = self.data['y_train'].shape[0]
        self.val_data_size = self.data['y_val'].shape[0]
        self.test_data_size = self.data['y_test'].shape[0]

        # define input and output symbolic variables of the computation graph
        self.path_to_moldata = path.join(path.dirname(path.realpath(__file__)), "../../data/moldata")
        self.max_atoms = np.memmap(path.join(self.path_to_moldata, 'max_atoms.memmap'), mode='r', dtype=intX)[0]
        coords = T.tensor3('coords')
        charges = T.matrix('charges')
        vdwradii = T.matrix('vdwradii')
        n_atoms = T.ivector('n_atoms')

        # TODO: replace all lists with for loops
        targets_ints = [T.ivector('targets21'), T.ivector('targets24')]

        # create a one-hot encoding if an integer vector
        # using a broadcast trick (targets is reshaped from (N) to (N, 1)
        # and then each entry is compared to [1, 2, ... K] in a broadcast
        targets = [T.eq(targets_ints[0].reshape((-1, 1)), T.arange(2)),
                   T.eq(targets_ints[1].reshape((-1, 1)), T.arange(2))]

        # build the network architecture
        self.outs = self._build_network(coords, charges, vdwradii, n_atoms)

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
                                                    learning_rate=1e-4)

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

        self.train_function = theano.function(inputs=[coords, charges, vdwradii, n_atoms, targets_ints[0], targets_ints[1]],
                                              outputs=[train_losses[0], train_losses[1], train_accuracies[0],
                                                       train_accuracies[1], train_predictions[0], targets[0]],
                                              updates=train_params_updates)  # , profile=True)

        self.validation_function = theano.function(inputs=[coords, charges, vdwradii, n_atoms, targets_ints[0], targets_ints[1]],
                                                   outputs=[val_losses[0], val_losses[1],
                                                            val_accuracies[0], val_accuracies[1]])

        self._get_all_outputs = theano.function(inputs=[coords, charges, vdwradii, n_atoms],
                                                outputs=lasagne.layers.get_output(
                                                    lasagne.layers.get_all_layers([self.outs[0]])))

        # save training history data
        self.history = {'train_loss': list(),
                        'train_accuracy': list(),
                        'val_loss': list(),
                        'val_accuracy': list(),
                        'time_epoch': list()}
        log.info("Computational graph compiled")

        self.monitor = ModelMonitor(self.outs, name=model_name)

    def _build_network(self, coords, charges, vdwradii, n_atoms):
        coords_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None, None),
                                                 input_var=coords)
        charges_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None),
                                                  input_var=charges)
        vdwradii_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None),
                                                   input_var=vdwradii)
        natoms_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,),
                                                 input_var=n_atoms)

        data_gen = MoleculeMapLayer(incomings=[coords_input, charges_input, vdwradii_input, natoms_input],
                                    minibatch_size=self.minibatch_size)

        network = data_gen  # lasagne.layers.dnn.BatchNormLayer(incoming=data_gen)

        filter_size = (3, 3, 3)

        for i in range(0, 6):
            # NOTE: we start with a very poor filter count.
            network = lasagne.layers.dnn.Conv3DDNNLayer(incoming=network, pad='same',
                                                        num_filters=2 ** (5 + i // 2),
                                                        filter_size=filter_size,
                                                        nonlinearity=lasagne.nonlinearities.leaky_rectify)
            if i % 2 == 1:
                network = lasagne.layers.dnn.MaxPool3DDNNLayer(incoming=network,
                                                               pool_size=(2, 2, 2),
                                                               stride=2)

        network1 = network
        network2 = network

        for i in range(0, 2):
            network1 = lasagne.layers.DenseLayer(incoming=network1, num_units=512,
                                                 nonlinearity=lasagne.nonlinearities.leaky_rectify)
            network2 = lasagne.layers.DenseLayer(incoming=network2, num_units=512,
                                                 nonlinearity=lasagne.nonlinearities.leaky_rectify)

        output_layer1 = lasagne.layers.DenseLayer(incoming=network1, num_units=2,
                                                  nonlinearity=T.nnet.logsoftmax)
        output_layer2 = lasagne.layers.DenseLayer(incoming=network2, num_units=2,
                                                  nonlinearity=T.nnet.logsoftmax)

        return output_layer1, output_layer2

    def _iter_minibatches(self, mode='train', num_per_class=100):
        # load molecula data from memmaps
        coords = np.memmap(path.join(self.path_to_moldata, 'coords.memmap'), mode='r', dtype=floatX).reshape(
            (-1, self.max_atoms, 3))
        charges = np.memmap(path.join(self.path_to_moldata, 'charges.memmap'), mode='r', dtype=floatX).reshape(
            (-1, self.max_atoms))
        vdwradii = np.memmap(path.join(self.path_to_moldata, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape(
            (-1, self.max_atoms))
        n_atoms = np.memmap(path.join(self.path_to_moldata, 'n_atoms.memmap'), mode='r', dtype=intX)

        data_size = self.data['y_' + mode].shape[0]
        num_classes = self.data['class_distribution_' + mode].shape[0]
        represented_classes = np.arange(num_classes)[self.data['class_distribution_' + mode] > 0.]
        if represented_classes.shape[0] < num_classes:
            log.warning("Non-exhaustive {0}-ing. Class (Classes) {1} is (are) not represented".
                        format(mode, np.arange(num_classes)[self.data['class_distribution_' + mode] <= 0.]))

        effective_datasize = num_per_class * represented_classes.shape[0]
        if effective_datasize > data_size:
            minibatch_count = data_size / self.minibatch_size
            if data_size % self.minibatch_size != 0:
                minibatch_count += 1
        else:
            minibatch_count = effective_datasize / self.minibatch_size
            if effective_datasize % self.minibatch_size != 0:
                minibatch_count += 1

        ys = self.data['y_' + mode]
        # one hot encoding of labels which are present in the current set of samples
        unique_labels = np.eye(num_classes)[represented_classes]
        # the following collects the indices in the `y_train` array
        # which correspond to different labels
        label_buckets = [np.nonzero(np.all(ys == label, axis=1))[0][:num_per_class]
                         for label in unique_labels]

        for _ in xrange(0, minibatch_count):
            bucket_ids = np.random.choice(represented_classes, size=self.minibatch_size)
            data_indices = [np.random.choice(label_buckets[i]) for i in bucket_ids]
            memmap_indices = self.data['x_' + mode][data_indices]
            yield data_indices, (coords[memmap_indices], charges[memmap_indices],
                                 vdwradii[memmap_indices], n_atoms[memmap_indices])

    def train(self, epoch_count=10, generate_progress_plot=True):
        try:
            log.info("Training...")
            if generate_progress_plot:
                self.plot_progress()
            self._train(epoch_count)
            self.monitor.save_train_history(self.history)
            self.summarize()
        except (KeyboardInterrupt, SystemExit):
            self.monitor.save_model(msg="interrupted")
            log.info("Training is interrupted and weights have been saved")
            exit(0)

    def _train(self, epoch_count=10):
        per_class_datasize = self.initial_per_class_datasize
        current_max_mean_train_acc = np.array([0.85, 0.85])
        current_max_mean_val_acc = np.array([0., 0.])
        steps_until_validate = 0
        for e in xrange(epoch_count):
            losses = []
            accs = []
            for indices, mol_info in self._iter_minibatches(mode='train', num_per_class=per_class_datasize):
                y = self.data['y_train'][indices]
                coords, charges, vdwradii, n_atoms = mol_info
                loss21, loss24, acc21, acc24, pred, tgt = self.train_function(coords, charges, vdwradii, n_atoms,
                                                                              y[:, 0], y[:, 1])

                # this can be enabled to profile the forward pass
                # self.train_function.profile.print_summary()

                losses.append((loss21, loss24))
                accs.append((acc21, acc24))
                self.history['train_loss'].append((loss21, loss24))
                self.history['train_accuracy'].append((acc21, acc24))
                steps_until_validate += 1

            mean_losses = np.mean(np.array(losses), axis=0)
            mean_accs = np.mean(np.array(accs), axis=0)
            log.info("train: epoch %d loss21: %f loss24 %f acc21: %f acc24: %f" %
                     (e, mean_losses[0], mean_losses[1], mean_accs[0], mean_accs[1]))

            if np.alltrue(mean_accs >= current_max_mean_train_acc):
                log.info("Augmenting dataset: doubling the samples per class ({0})".
                         format(2 * per_class_datasize))
                current_max_mean_train_acc = mean_accs
                per_class_datasize *= 2

            # validate the model and save parameters if an improvement is observed
            if e % 9 == 0:
                val_samples_per_class = max(1, (10 * per_class_datasize) // 100)
                val_loss21, val_loss24, val_acc21, val_acc24 = self._test(mode='val',
                                                                          num_per_class=val_samples_per_class)
                self.history['val_loss'] += [(val_loss21, val_loss24)] * steps_until_validate
                self.history['val_accuracy'] += [(val_acc21, val_acc24)] * steps_until_validate
                steps_until_validate = 0
                if np.alltrue(np.array([val_acc21, val_acc24]) > current_max_mean_val_acc):
                    current_max_mean_val_acc = np.array([val_acc21, val_acc24])
                    self.monitor.save_model(e, "meanvalacc{0}".format(np.mean(current_max_mean_val_acc)))

    def test_final(self):
        log.warning(
            "You are testing a model with the secret test set! " +
            "You are not allowed to change the model after seeing the results!!! ")
        response = raw_input("Are you sure you want to proceed? (yes/[no]): ")
        if response != 'yes':
            return
        else:
            return self._test(mode='test')

    def _test(self, mode='val', num_per_class=100):
        if mode == 'test':
            log.info("Final model testing...")
        elif mode == 'val':
            log.info("Validating model...")
        losses = []
        accs = []
        for indices, mol_info in self._iter_minibatches(mode=mode, num_per_class=num_per_class):
            y = self.data['y_' + mode][indices]
            coords, charges, vdwradii, n_atoms = mol_info
            loss21, loss24, acc21, acc24 = self.validation_function(coords, charges, vdwradii, n_atoms,
                                                                    y[:, 0], y[:, 1])
            losses.append((loss21, loss24))
            accs.append((acc21, acc24))

        mean_losses = np.mean(np.array(losses), axis=0)
        mean_accs = np.mean(np.array(accs), axis=0)
        log.info("%s: loss21: %f loss24 %f acc21: %f acc24: %f" %
                 (mode, mean_losses[0], mean_losses[1], mean_accs[0], mean_accs[1]))

        return mean_losses[0], mean_losses[1], mean_accs[0], mean_accs[1]

    @staticmethod
    def summarize():
        log.info("The network has been tremendously successful!")

    def plot_progress(self):
        t = threading.Timer(5.0, self.plot_progress)
        t.daemon = True
        t.start()
        progress = ProgressView(model_name="prot_predictor", history_dict=self.history)
        progress.save()
