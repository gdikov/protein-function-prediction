import logging
import abc
import colorlog as log
import numpy as np
import theano

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

        self.data_manager = EnzymeDataManager(force_download=False,
                                              force_memmaps=False,
                                              force_grids=False,
                                              force_split=False)

    # NOTE: this method will be removed in the very near future. I need it for a moment.
    def _iter_minibatches(self, mode='train'):
        if mode == "train":
            samples, labels = self.data_manager.get_training_set()
            data_dir = self.data_manager.dirs['data_train']
        elif mode == "test":
            samples, labels = self.data_manager.get_test_set()
            data_dir = self.data_manager.dirs['data_test']
        else:  # mode == "val"
            samples, labels = self.data_manager.get_validation_set()
            data_dir = self.data_manager.dirs['data_train']

        data_size = 0  # TODO: calculate the data size properly

        # TODO: we need to do some sort of selection of the classes here
        # TODO: because samples.keys() contains all leaf classes, and we probably don't need all of them
        classes = samples.keys()
        num_classes = len(classes)

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
            classes_in_minibatch = np.random.choice(classes, size=self.minibatch_size)
            prots_in_minibatch = [np.random.choice(samples[cls][:self.samples_per_class]) for cls in
                                  classes_in_minibatch]

            # TODO: @georgi, can you make this work? next_targets should be a numpy array of the 1-hot target labels
            # TODO: which are to be formed from the 'labels' variable
            next_targets = list()

            next_samples = self._form_samples_minibatch(prot_codes=prots_in_minibatch, from_dir=data_dir)
            yield next_samples + next_targets

    @abc.abstractmethod
    def _form_samples_minibatch(self, prot_codes, from_dir):
        raise NotImplementedError


class EnzymesMolDataFeeder(EnzymeDataFeeder):
    def __init__(self, minibatch_size, init_samples_per_class):
        super(EnzymesMolDataFeeder, self).__init__(minibatch_size, init_samples_per_class)

    def iterate_test_data(self):
        for inputs in self._iter_minibatches(mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self._iter_minibatches(mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self._iter_minibatches(mode='val'):
            yield inputs

    def _form_samples_minibatch(self, prot_codes, from_dir):
        # TODO: load the memmaps from the given from_dir
        # TODO: stack them into a numpy array
        # TODO: return the array
        raise NotImplementedError


class EnzymesGridFeeder(EnzymeDataFeeder):
    def __init__(self, minibatch_size, init_samples_per_class):
        super(EnzymesGridFeeder, self).__init__(minibatch_size, init_samples_per_class)

    def iterate_test_data(self):
        for inputs in self._iter_minibatches(mode='test'):
            yield inputs

    def iterate_train_data(self):
        for inputs in self._iter_minibatches(mode='train'):
            yield inputs

    def iterate_val_data(self):
        for inputs in self._iter_minibatches(mode='val'):
            yield inputs

    def _form_samples_minibatch(self, prot_codes, from_dir):
        # TODO: load the grid memmaps from the given from_dir
        # TODO: stack them into a numpy array
        # TODO: return the array
        raise NotImplementedError
