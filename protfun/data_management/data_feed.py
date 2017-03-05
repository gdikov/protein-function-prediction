import logging

import abc
import colorlog as log
import numpy as np
import theano
from os import path

from protfun.data_management.data_manager import EnzymeDataManager
from protfun.utils import construct_hierarchical_tree

log.basicConfig(level=logging.DEBUG)
floatX = theano.config.floatX
intX = np.int32


class DataFeeder(object):
    def __init__(self, data_dir, minibatch_size, init_samples_per_class):
        self.data_dir = data_dir
        self.samples_per_class = init_samples_per_class
        self.minibatch_size = minibatch_size

    @abc.abstractmethod
    def iterate_test_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def iterate_train_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def iterate_val_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_val_data(self):
        raise NotImplementedError

    def set_samples_per_class(self, samples_per_class):
        self.samples_per_class = samples_per_class

    def get_samples_per_class(self):
        return self.samples_per_class

    def get_data_dir(self):
        return self.data_dir


class EnzymeDataFeeder(DataFeeder):
    def __init__(self, data_dir, minibatch_size, init_samples_per_class,
                 prediction_depth, enzyme_classes):
        super(EnzymeDataFeeder, self).__init__(data_dir, minibatch_size,
                                               init_samples_per_class)

        self.data_manager = EnzymeDataManager(data_dir=data_dir,
                                              enzyme_classes=enzyme_classes,
                                              force_download=False,
                                              force_memmaps=False,
                                              force_grids=False,
                                              force_split=False)
        self.prediction_depth = prediction_depth

    def iterate_test_data(self):
        for inputs in self._iter_minibatches(iter_mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self._iter_minibatches(iter_mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self._iter_minibatches(iter_mode='val'):
            yield inputs

    def get_test_data(self):
        return self.data_manager.get_test_set()

    def get_train_data(self):
        return self.data_manager.get_training_set()

    def get_val_data(self):
        return self.data_manager.get_validation_set()

    def _iter_minibatches(self, iter_mode='train'):
        if iter_mode == "train":
            samples, labels = self.data_manager.get_training_set()
            data_dir = self.data_manager.dirs['data_train']
        elif iter_mode == "test":
            samples, labels = self.data_manager.get_test_set()
            data_dir = self.data_manager.dirs['data_test']
        elif iter_mode == "val":
            samples, labels = self.data_manager.get_validation_set()
            data_dir = self.data_manager.dirs['data_train']
        else:
            log.error("iter_mode can only be 'train', 'val' or 'test'")
            raise ValueError

        grouped_samples = construct_hierarchical_tree(samples,
                                                      prediction_depth=self.prediction_depth)

        represented_classes, data_sizes = \
            map(list, zip(
                *[(cls, len(prots)) for cls, prots in grouped_samples.items() if
                  len(prots) > 0]))
        data_size = sum(data_sizes)
        num_classes = len(represented_classes)

        effective_datasize = self.samples_per_class * num_classes
        if effective_datasize > data_size:
            minibatch_count = data_size // self.minibatch_size
            if data_size % self.minibatch_size != 0:
                minibatch_count += 1
        else:
            minibatch_count = effective_datasize // self.minibatch_size
            if effective_datasize % self.minibatch_size != 0:
                minibatch_count += 1

        for _ in xrange(0, minibatch_count):
            class_choices = np.random.choice(represented_classes,
                                             size=self.minibatch_size,
                                             replace=True)

            prots_in_minibatch = [np.random.choice(
                grouped_samples[class_][:self.samples_per_class])
                                  for class_ in class_choices]

            next_samples = self._form_samples_minibatch(
                prot_codes=prots_in_minibatch, from_dir=data_dir)

            # labels are accessed at a fixed hierarchical depth counting from
            # the root
            next_targets = [np.vstack(
                [labels[prot_code][self.prediction_depth - 1].astype(intX) for
                 prot_code in prots_in_minibatch])]

            yield prots_in_minibatch, next_samples + next_targets

    @abc.abstractmethod
    def _form_samples_minibatch(self, prot_codes, from_dir):
        raise NotImplementedError


class EnzymesMolDataFeeder(EnzymeDataFeeder):
    def __init__(self, data_dir, minibatch_size, init_samples_per_class,
                 prediction_depth, enzyme_classes):
        super(EnzymesMolDataFeeder, self).__init__(data_dir, minibatch_size,
                                                   init_samples_per_class,
                                                   prediction_depth,
                                                   enzyme_classes)

    def _form_samples_minibatch(self, prot_codes, from_dir):
        assert len(prot_codes) == self.minibatch_size, \
            "prot_codes must be of the same size as minibatch_size"
        coords_tmp = []
        charges_tmp = []
        vdwradii_tmp = []
        n_atoms = np.zeros((self.minibatch_size,), dtype=intX)
        for i, prot_id in enumerate(prot_codes):
            path_to_prot = path.join(from_dir, prot_id.upper())
            coords_tmp.append(
                np.memmap(path.join(path_to_prot, 'coords.memmap'), mode='r',
                          dtype=floatX).reshape(-1, 3))
            charges_tmp.append(
                np.memmap(path.join(path_to_prot, 'charges.memmap'), mode='r',
                          dtype=floatX))
            vdwradii_tmp.append(
                np.memmap(path.join(path_to_prot, 'vdwradii.memmap'), mode='r',
                          dtype=floatX))
            n_atoms[i] = vdwradii_tmp[i].shape[0]

        max_atoms = max(n_atoms)
        coords = np.zeros((self.minibatch_size, max_atoms, 3), dtype=floatX)
        charges = np.zeros((self.minibatch_size, max_atoms), dtype=floatX)
        vdwradii = np.zeros((self.minibatch_size, max_atoms), dtype=floatX)

        for i in range(self.minibatch_size):
            coords[i, :n_atoms[i], :] = coords_tmp[i]
            charges[i, :n_atoms[i]] = charges_tmp[i]
            vdwradii[i, :n_atoms[i]] = vdwradii_tmp[i]

        return [coords, charges, vdwradii, n_atoms]


class EnzymesGridFeeder(EnzymeDataFeeder):
    def __init__(self, data_dir, minibatch_size,
                 init_samples_per_class, prediction_depth,
                 enzyme_classes, num_channels, grid_size):
        super(EnzymesGridFeeder, self).__init__(data_dir, minibatch_size,
                                                init_samples_per_class,
                                                prediction_depth,
                                                enzyme_classes)
        self.num_channels = num_channels
        self.grid_size = grid_size

    def _form_samples_minibatch(self, prot_codes, from_dir):
        assert len(prot_codes) == self.minibatch_size, \
            "prot_codes must be of the same size as minibatch_size"
        grids = list()
        for prot_id in prot_codes:
            path_to_prot = path.join(from_dir, prot_id.upper())
            grids.append(
                np.memmap(path.join(path_to_prot, 'grid.memmap'), mode='r',
                          dtype=floatX).reshape((1, -1,
                                                 self.grid_size,
                                                 self.grid_size,
                                                 self.grid_size)))

        stacked = np.vstack(grids)

        # a small hack to work around the molecules that still contain ESP
        # channel
        return [stacked[:, stacked.shape[1] - self.num_channels:]]
