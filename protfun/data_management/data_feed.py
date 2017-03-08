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
    """
     DataFeeder is an abstract class (not meant to be instantiated).
     All data feeders implement iterate_{train, test, val}_data() and
     get_{train, test, val}_data() methods. The iterate methods are mini-
     batch generators (you can use for loops on them to get mini-batches),
     whereas the get_ methods return the whole data sets.

     Thus, the data feeders are meant to be used during training / testing
     of models to provide the data that must be fed into them.

     Usage:
        >>> dummy_feeder = DataFeeder(...)
        >>> for train_minibatch in dummy_feeder.iterate_train_data():
        >>>     # do something to the minibatch, e.g. feed forward into
        >>>     # your model

     """

    __metaclass__ = abc.ABCMeta

    def __init__(self, minibatch_size, init_samples_per_class):
        """
        :param minibatch_size: minibatch_size for the minibatches that the data feeder will generate
        :param init_samples_per_class: (optional) restrict the number of samples in each class in
            the data set, i.e. the data feeder will provide only that many unique objects from each
            class, and not more.
        """
        self.samples_per_class = init_samples_per_class
        self.minibatch_size = minibatch_size

    @abc.abstractmethod
    def iterate_test_data(self):
        """
        :return: a python iterator, generates test set minibatches
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterate_train_data(self):
        """
        :return: a python iterator, generates train set minibatches
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterate_val_data(self):
        """
        :return: a python iterator, generates validation set minibatches
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data(self):
        """
        :return: the whole test set
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data(self):
        """
        :return: the whole training set
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_val_data(self):
        """
        :return: the whole validation set
        """
        raise NotImplementedError

    def set_samples_per_class(self, samples_per_class):
        """
        Setter for the restricted number of samples in each class.

        :param samples_per_class: new count of maximal number of samples from each class
        """
        self.samples_per_class = samples_per_class

    def get_samples_per_class(self):
        """
        Getter for the restricted number of samples in each class.
        The data feeder will provide max. that many unique samples from each class in the
        data.
        :return: the current samples_per_class count
        """
        return self.samples_per_class



class EnzymeDataFeeder(DataFeeder):
    """
    EnzymeDataFeeder implements DataFeeder, but is also an abstract class that should not
    be instantiated. It is the basis for feeder implementations that provide enzyme proteins data.

    The class relies on an independent DataManager that can manage and provide the different splits
    of the enzymes data set (train / test / val).
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_manager, minibatch_size, init_samples_per_class,
                 prediction_depth):
        """
        :param data_manager: data manager to download and process the protein files.
        :param minibatch_size: see docs for DataFeeder.
        :param init_samples_per_class: see docs for DataFeeder.
        :param prediction_depth: integer depth in the EC tree of enzymes proteins, it is the
            depth on which we do prediction (e.g. 3)
        :param enzyme_classes: which enzyme classes in the EC tree should be included into the
            data set (white list). E.g. ['3.4.21', '3.4.24']. Can be at any level of the hierarchy,
            all child proteins of the specified classes will be in the data set.
        """
        super(EnzymeDataFeeder, self).__init__(minibatch_size,
                                               init_samples_per_class)

        # instantiate an enzyme data manager that can download, preprocess and split
        # the enzymes data
        self.data_manager = data_manager
        self.prediction_depth = prediction_depth

    def iterate_test_data(self):
        """
        See DataFeeder's doc.
        """
        for inputs in self._iter_minibatches(iter_mode='test'):
            yield inputs

    def iterate_train_data(self):
        """
        See DataFeeder's doc.
        """
        for inputs in self._iter_minibatches(iter_mode='train'):
            yield inputs

    def iterate_val_data(self):
        """
        See DataFeeder's doc.
        """
        for inputs in self._iter_minibatches(iter_mode='val'):
            yield inputs

    def get_data_dir(self):
        return self.data_manager.get_data_dir()

    def get_test_data(self):
        """
        See DataFeeder's doc.
        """
        return self.data_manager.get_test_set()

    def get_train_data(self):
        """
        See DataFeeder's doc.
        """
        return self.data_manager.get_training_set()

    def get_val_data(self):
        """
        See DataFeeder's doc.
        """
        return self.data_manager.get_validation_set()

    def _iter_minibatches(self, iter_mode='train'):
        """
        Internal method, does the actual iteration over mini-batches.
        """
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

        # group the enzymes data by the specified prediction depth,
        # the EC categories at that depth will be treated as labels.
        grouped_samples = construct_hierarchical_tree(samples,
                                                      prediction_depth=self.prediction_depth)

        # find out if some classes have no samples in them
        # also determine the data size
        represented_classes, data_sizes = \
            map(list, zip(
                *[(cls, len(prots)) for cls, prots in grouped_samples.items() if len(prots) > 0]))
        data_size = sum(data_sizes)
        num_classes = len(represented_classes)

        # determine the effective data size, based on samples_per_class
        # Note: samples_per_class tells the maximal number of protein that should be provided
        # for each class
        effective_datasize = self.samples_per_class * num_classes
        if effective_datasize > data_size:
            minibatch_count = data_size // self.minibatch_size
            if data_size % self.minibatch_size != 0:
                minibatch_count += 1
        else:
            minibatch_count = effective_datasize // self.minibatch_size
            if effective_datasize % self.minibatch_size != 0:
                minibatch_count += 1

        # produce the actual mini-batches in a python iterator
        # the process is randomized as follows: for i in range(0, minibatch_size):
        #   * first a class in the data is picked at random
        #   * then a sample from that class is picked at random
        #   * this sample is then the i-th sample in the mini-batch
        for _ in xrange(0, minibatch_count):
            class_choices = np.random.choice(represented_classes, size=self.minibatch_size,
                                             replace=True)

            prots_in_minibatch = [np.random.choice(grouped_samples[class_][:self.samples_per_class])
                                  for class_ in class_choices]

            next_samples = self._form_samples_minibatch(prot_codes=prots_in_minibatch,
                                                        from_dir=data_dir)

            # labels are accessed at a fixed hierarchical depth counting from the root
            next_targets = [np.vstack(
                [labels[prot_code][self.prediction_depth - 1].astype(intX) for prot_code in
                 prots_in_minibatch])]

            yield prots_in_minibatch, next_samples + next_targets

    @abc.abstractmethod
    def _form_samples_minibatch(self, prot_codes, from_dir):
        """
        Internal abstract method to actually form the enzyme minibatches, should be implemented
        by all classes that implement EnzymesFeeder.
        :param prot_codes: protein codes for the proteins in this mini-batch
        :param from_dir: directory under which those proteins could be loaded
        :return: the formed minibatch
        """
        raise NotImplementedError


class EnzymesMolDataFeeder(EnzymeDataFeeder):
    """
    EnzymesMolDataFeeder is an enzyme protein feeder, that can provide mini-batches of
    [coordinates, vdwradii, n_atoms], for each enzyme protein in the mini-batch.
    """

    def __init__(self, data_manager, minibatch_size, init_samples_per_class,
                 prediction_depth):
        """
        See doc for EnzymeDataFeeder.
        """
        super(EnzymesMolDataFeeder, self).__init__(data_manager, minibatch_size,
                                                   init_samples_per_class,
                                                   prediction_depth)

    def _form_samples_minibatch(self, prot_codes, from_dir):
        """
        Forms a minibatch of [coords, vdwradii, n_atoms] for each of the proteins with PDB
        code in prot_codes. Expects that the data to be loaded is located under from_dir/<prot_code>
        for each protein, in files 'coords.memmap' and 'vdwradii.memmap'.

        See doc in EnzymesDataFeeder for parameters.
        """
        assert len(prot_codes) == self.minibatch_size, \
            "prot_codes must be of the same size as minibatch_size"
        coords_tmp = []
        vdwradii_tmp = []
        n_atoms = np.zeros((self.minibatch_size,), dtype=intX)
        for i, prot_id in enumerate(prot_codes):
            path_to_prot = path.join(from_dir, prot_id.upper())
            coords_tmp.append(
                np.memmap(path.join(path_to_prot, 'coords.memmap'), mode='r',
                          dtype=floatX).reshape(-1, 3))
            vdwradii_tmp.append(
                np.memmap(path.join(path_to_prot, 'vdwradii.memmap'), mode='r',
                          dtype=floatX))
            n_atoms[i] = vdwradii_tmp[i].shape[0]

        max_atoms = max(n_atoms)
        coords = np.zeros((self.minibatch_size, max_atoms, 3), dtype=floatX)
        vdwradii = np.zeros((self.minibatch_size, max_atoms), dtype=floatX)

        for i in range(self.minibatch_size):
            coords[i, :n_atoms[i], :] = coords_tmp[i]
            vdwradii[i, :n_atoms[i]] = vdwradii_tmp[i]

        return [coords, vdwradii, n_atoms]


class EnzymesGridFeeder(EnzymeDataFeeder):
    """
    EnzymesGridFeeder is an enzyme protein feeder, that can provide mini-batches of
    already computed 3D grids, for each enzyme protein in the mini-batch.

    The grids can represent different properties of the protein, e.g. the full electron density,
    the electron density of side chains in the protein, electron density of the backbone, etc.
    """

    def __init__(self, data_manager, minibatch_size,
                 init_samples_per_class, prediction_depth,
                 num_channels, grid_size):
        """
        See EnzymeDataFeeder for remaining parameters.
        :param num_channels: how many channels do the electron density grids have (normally it
            should be 1).
        :param grid_size: what is the number of points on each side of the electron density grid,
            e.g. 128
        """
        super(EnzymesGridFeeder, self).__init__(data_manager, minibatch_size,
                                                init_samples_per_class,
                                                prediction_depth)
        self.num_channels = num_channels
        self.grid_size = grid_size

    def _form_samples_minibatch(self, prot_codes, from_dir):
        """
        Forms a minibatch of electron density grids for each of the proteins with PDB
        code in prot_codes. Expects that the data to be loaded is located under from_dir/<prot_code>
        for each protein, in a file  'grid.memmap'.

        See doc in EnzymesDataFeeder for parameters.
        """
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
        # TODO: remove this when the code is run on only electron density grids
        return [stacked[:, stacked.shape[1] - self.num_channels:]]
