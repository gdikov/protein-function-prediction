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
    def __init__(self, minibatch_size, init_samples_per_class, prediction_depth, enzyme_classes):
        super(EnzymeDataFeeder, self).__init__(minibatch_size, init_samples_per_class)

        self.data_manager = EnzymeDataManager(data_dirname='data_new',
                                              enzyme_classes=enzyme_classes,
                                              force_download=False,
                                              force_memmaps=False,
                                              force_grids=False,
                                              force_split=False)
        self.prediction_depth = prediction_depth

    def _construct_hierarchical_tree(self, data_dict):

        def merge_prots(subpath):
            merged = []
            for key, vals in data_dict.items():
                if key.startswith(subpath):
                    merged += vals
            return merged

        keys_at_max_hdepth = set(['.'.join(x.split('.')[:self.prediction_depth]) for x in data_dict.keys()])
        tree_at_max_hdepth = {key: merge_prots(key) for key in keys_at_max_hdepth}
        return tree_at_max_hdepth

    def _TODO_refactor_me_asap(self, stacked_1hot_labels):
        return [stacked_1hot_labels[:, i] for i in range(stacked_1hot_labels.shape[1])]

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

        grouped_samples = self._construct_hierarchical_tree(samples)

        represented_classes, data_sizes = \
            map(list, zip(*[(cls, len(prots)) for cls, prots in grouped_samples.items() if len(prots) > 0]))
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

            prots_in_minibatch = [np.random.choice(grouped_samples[class_][:self.samples_per_class])
                                  for class_ in class_choices]

            next_samples = self._form_samples_minibatch(prot_codes=prots_in_minibatch, from_dir=data_dir)

            # labels are accessed at a fixed hierarchical depth counting from the root
            next_targets = self._TODO_refactor_me_asap(
                np.vstack([labels[prot_code][self.prediction_depth-1][0].astype(intX)
                           for prot_code in prots_in_minibatch]))

            yield next_samples + next_targets

    @abc.abstractmethod
    def _form_samples_minibatch(self, prot_codes, from_dir):
        raise NotImplementedError


class EnzymesMolDataFeeder(EnzymeDataFeeder):
    def __init__(self, minibatch_size, init_samples_per_class, prediction_depth, enzyme_classes):
        super(EnzymesMolDataFeeder, self).__init__(minibatch_size, init_samples_per_class,
                                                   prediction_depth, enzyme_classes)

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
            coords_tmp.append(
                np.memmap(path.join(path_to_prot, 'coords.memmap'), mode='r', dtype=floatX).reshape(-1, 3))
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
    def __init__(self, minibatch_size, init_samples_per_class, prediction_depth, enzyme_classes):
        super(EnzymesGridFeeder, self).__init__(minibatch_size, init_samples_per_class,
                                                prediction_depth, enzyme_classes)

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
        grids = list()
        for i, prot_id in enumerate(prot_codes):
            path_to_prot = path.join(from_dir, prot_id.upper())
            # TODO: refactor this hardcoded resolution reshape
            grids.append(
                np.memmap(path.join(path_to_prot, 'grid.memmap'), mode='r', dtype=floatX).reshape((1, 2, 64, 64, 64)))

        return [np.vstack(grids)]
