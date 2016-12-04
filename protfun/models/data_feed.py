import numpy as np
import theano
import colorlog as log
import logging
from os import path

from protfun.preprocess.data_prep import DataSetup

log.basicConfig(level=logging.DEBUG)
floatX = theano.config.floatX
intX = np.int32


class DataFeeder(object):
    def __init__(self, minibatch_size, init_samples_per_class):
        self.samples_per_class = init_samples_per_class
        self.minibatch_size = minibatch_size

    def iterate_test_data(self):
        return None

    def iterate_train_data(self):
        return None

    def iterate_val_data(self):
        return None

    def set_samples_per_class(self, samples_per_class):
        self.samples_per_class = samples_per_class

    def get_samples_per_class(self):
        return self.samples_per_class


class EnzymeDataFeeder(DataFeeder):
    def __init__(self, minibatch_size, init_samples_per_class, enzyme_classes, force_download=False,
                 force_process=False):
        super(EnzymeDataFeeder, self).__init__(minibatch_size, init_samples_per_class)
        data = DataSetup(enzyme_classes=enzyme_classes,
                         label_type='enzyme_classes',
                         force_download=force_download,
                         force_process=force_process)
        self.data = data.load_dataset()

    def iter_minibatches(self, inputs_list, mode='train'):
        data_size = self.data['y_' + mode].shape[0]
        num_classes = self.data['class_distribution_' + mode].shape[0]
        represented_classes = np.arange(num_classes)[self.data['class_distribution_' + mode] > 0.]
        if represented_classes.shape[0] < num_classes:
            log.warning("Non-exhaustive {0}-ing. Class (Classes) {1} is (are) not represented".
                        format(mode, np.arange(num_classes)[self.data['class_distribution_' + mode] <= 0.]))

        effective_datasize = self.samples_per_class * represented_classes.shape[0]
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
        label_buckets = [np.nonzero(np.all(ys == label, axis=1))[0][:self.samples_per_class]
                         for label in unique_labels]

        for _ in xrange(0, minibatch_count):
            bucket_ids = np.random.choice(represented_classes, size=self.minibatch_size)
            data_indices = [np.random.choice(label_buckets[i]) for i in bucket_ids]
            memmap_indices = self.data['x_' + mode][data_indices]

            next_targets = [ys[data_indices, i] for i in range(0, ys.shape[1])]
            next_data_points = [input_var[memmap_indices] for input_var in inputs_list]
            yield next_data_points + next_targets


class EnzymesMolDataFeeder(EnzymeDataFeeder):
    def __init__(self, path_to_moldata, minibatch_size, init_samples_per_class, enzyme_classes,
                 force_download=False,
                 force_process=False):
        super(EnzymesMolDataFeeder, self).__init__(minibatch_size, init_samples_per_class,
                                                   enzyme_classes=enzyme_classes,
                                                   force_download=force_download,
                                                   force_process=force_process)

        self.max_atoms = np.memmap(path.join(path_to_moldata, 'max_atoms.memmap'), mode='r', dtype=intX)[0]
        coords = np.memmap(path.join(path_to_moldata, 'coords.memmap'), mode='r', dtype=floatX).reshape(
            (-1, self.max_atoms, 3))
        charges = np.memmap(path.join(path_to_moldata, 'charges.memmap'), mode='r', dtype=floatX).reshape(
            (-1, self.max_atoms))
        vdwradii = np.memmap(path.join(path_to_moldata, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape(
            (-1, self.max_atoms))
        n_atoms = np.memmap(path.join(path_to_moldata, 'n_atoms.memmap'), mode='r', dtype=intX)

        self.inputs_list = [coords, charges, vdwradii, n_atoms]

    def iterate_test_data(self):
        for inputs in self.iter_minibatches(self.inputs_list, mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self.iter_minibatches(self.inputs_list, mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self.iter_minibatches(self.inputs_list, mode='val'):
            yield inputs


class EnzymesGridFeeder(EnzymeDataFeeder):
    def __init__(self, grids_dir, grid_size, minibatch_size, init_samples_per_class, enzyme_classes=list(),
                 force_download=False,
                 force_process=False):
        super(EnzymesGridFeeder, self).__init__(minibatch_size, init_samples_per_class,
                                                enzyme_classes=enzyme_classes,
                                                force_download=force_download,
                                                force_process=force_process)
        total_size = self.data['y_train'].shape[0] + self.data['y_val'].shape[0] + self.data['y_test'].shape[0]

        # read all grid files from the grids dir, they are the models inputs
        grids = list()
        for i in range(0, total_size):
            grid_file = path.join(grids_dir, "grid{}.memmap".format(i))
            grid = np.memmap(grid_file, mode='r', dtype=floatX).reshape((1, 2, grid_size, grid_size, grid_size))
            grids.append(grid)

        # convert ot a numpy array
        grids = np.vstack(grids)
        self.inputs_list = [grids]

    def iterate_test_data(self):
        for inputs in self.iter_minibatches(self.inputs_list, mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self.iter_minibatches(self.inputs_list, mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self.iter_minibatches(self.inputs_list, mode='val'):
            yield inputs
