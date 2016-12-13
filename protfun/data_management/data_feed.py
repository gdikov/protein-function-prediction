import logging
import abc
import colorlog as log
import numpy as np
import theano
from os import path

from protfun.data_management.data_manager import EnzymeDataManager

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
    def __init__(self, minibatch_size, init_samples_per_class):
        super(EnzymeDataFeeder, self).__init__(minibatch_size, init_samples_per_class)

        self.data_manager = EnzymeDataManager(enzyme_classes=['3.4.21', '3.4.24'],
                                              force_download=False,
                                              force_memmaps=False,
                                              force_grids=False,
                                              force_split=False)

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

        represented_classes, data_sizes = \
            map(list, zip(*[(cls, len(prots)) for cls, prots in samples.items() if len(prots) > 0]))
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
            classes_in_minibatch = np.random.choice(represented_classes,
                                                    size=self.minibatch_size,
                                                    replace=True)
            prots_in_minibatch = [np.random.choice(samples[cls][:self.samples_per_class])
                                  for cls in classes_in_minibatch]

            # labels are accessed at a fixed hierarchical depth of 3 counting from the root, e.g. 3.4.21.
            # TODO: make it a variable and incoroporate the knowledge of all labels of depth < max_depth
            next_targets = [labels[prot_code][2][0].astype(intX) for prot_code in prots_in_minibatch]

            next_samples = self._form_samples_minibatch(prot_codes=prots_in_minibatch, from_dir=data_dir)
            yield next_samples + next_targets

    @abc.abstractmethod
    def _form_samples_minibatch(self, prot_codes, from_dir):
        raise NotImplementedError


class EnzymesMolDataFeeder(EnzymeDataFeeder):
    def __init__(self, minibatch_size, init_samples_per_class):
        super(EnzymesMolDataFeeder, self).__init__(minibatch_size, init_samples_per_class)

    def iterate_test_data(self):
        for inputs in self._iter_minibatches(iter_mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self._iter_minibatches(iter_mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self._iter_minibatches(iter_mode='val'):
            yield inputs

    def _form_samples_minibatch(self, prot_codes, from_dir):
        assert len(prot_codes) == self.minibatch_size, \
            "prot_codes must be of the same size as minibatch_size"
        coords_tmp = []
        charges_tmp = []
        vdwradii_tmp = []
        n_atoms = np.zeros((self.minibatch_size,), dtype=intX)
        for i, prot_id in enumerate(prot_codes):
            path_to_prot = path.join(from_dir, prot_id.upper())
            coords_tmp.append(np.memmap(path.join(path_to_prot, 'coords.memmap'), mode='r', dtype=floatX).reshape(-1, 3))
            charges_tmp.append(np.memmap(path.join(path_to_prot, 'charges.memmap'), mode='r', dtype=floatX))
            vdwradii_tmp.append(np.memmap(path.join(path_to_prot, 'vdwradii.memmap'), mode='r', dtype=floatX))
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
    def __init__(self, minibatch_size, init_samples_per_class):
        super(EnzymesGridFeeder, self).__init__(minibatch_size, init_samples_per_class)

    def iterate_test_data(self):
        for inputs in self._iter_minibatches(iter_mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self._iter_minibatches(iter_mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self._iter_minibatches(iter_mode='val'):
            yield inputs

    def _form_samples_minibatch(self, prot_codes, from_dir):
        # TODO: load the grid memmaps from the given from_dir
        # TODO: stack them into a numpy array
        # TODO: return the array
        raise NotImplementedError
