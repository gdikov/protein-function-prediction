import shutil
import abc
import colorlog as log
import numpy as np
import os

import protfun.data_management.preprocess as prep
from protfun.data_management.label_factory import LabelFactory
from protfun.data_management.validation import EnzymeValidator
from protfun.utils import save_pickle, load_pickle, construct_hierarchical_tree


class DataManager(object):
    """
    DataManager is a parent class for EnzymeDataManager which stores all data directories and
    implements a *naive* split strategy described below.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir,
                 force_download=False, force_process=False, force_split=False,
                 percentage_test=10, percentage_val=20):
        """
        :param data_dir: the path to the root data directory
        :param force_download: forces the downloading of the enzymes
        :param force_process: forces the pre-processing steps
        :param force_split: forces the splitting of the data into training ,validation and test sets
        :param percentage_test: the portion in % of the test data
        :param percentage_val: the portion in % of the validation data
        """
        self.force_download = force_download
        self.force_process = force_process or force_download
        self.force_split = force_split or force_process or force_download
        self.p_test = percentage_test
        self.p_val = percentage_val

        self.dirs = {'data': data_dir,
                     'data_raw': os.path.join(data_dir, "raw"),
                     'data_processed': os.path.join(data_dir, "processed"),
                     'data_train': os.path.join(data_dir, "train"),
                     'data_test': os.path.join(data_dir, "test"),
                     'misc': os.path.join(data_dir, "misc")}

        # ensure all directories exist
        for _, d in self.dirs.items():
            if not os.path.exists(d) and not os.path.islink(d):
                os.makedirs(d)

    def get_data_dir(self):
        return self.dirs['data']

    @abc.abstractmethod
    def get_test_set(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_training_set(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_set(self):
        raise NotImplementedError

    @staticmethod
    def split_data_on_sublevel(data_dict, percentage, hierarchical_depth):
        import itertools
        first_data_dict = dict()
        second_data_dict = dict()

        target_classes = set(
            ['.'.join(cls.split('.')[:hierarchical_depth]) for cls in
             data_dict])

        for target_cls in target_classes:
            children = [(cls, enzymes) for cls, enzymes in data_dict.items() if
                        cls.startswith(target_cls + '.')]
            target_cls_prots = set(
                itertools.chain.from_iterable(zip(*children)[1]))
            required_count = ((100 - percentage) * len(target_cls_prots)) // 100
            sorted_children = sorted(children, key=lambda x: len(x[1]),
                                     reverse=True)
            collected_so_far = set()
            for cls, enzymes in sorted_children:
                if len(collected_so_far) < required_count:
                    collected_so_far |= set(enzymes)
                    second_data_dict[cls] = enzymes
                else:
                    first_data_dict[cls] = enzymes

        return first_data_dict, second_data_dict

    @staticmethod
    def split_data_on_level(data_dict, percentage, level=3):
        """
        performs a *naive* split, i.e. splitting proteins codes within a leaf-node from the
        hierarchical category tree, or a *semi-naive* split when spliting on a higher level node.
        In the latter case, the proteins within all sublevels of a higher-level node are merged
        into a pool of protein codes and then split according to the percentage value.

        :param data_dict: a dictionary with keys categories and value per key - list of pdb codes
        :param percentage: the portion of the data in % that should be put into the first split
        :param level: the hierarchical tree depth level on which the split is made.
        :return: a tuple of the two splits as data dictionaries
        """
        if not 0 <= percentage <= 100:
            log.error("Bad percentage number. Must be in [0, 100]")
            raise ValueError

        first_data_dict = {key: [] for key in data_dict.keys()}
        second_data_dict = {key: [] for key in data_dict.keys()}
        if level < 4:
            merged_on_level = construct_hierarchical_tree(data_dict, prediction_depth=level)
            prots2classes_dict = dict()
            for cls, prot_codes in data_dict.items():
                for p in prot_codes:
                    prots2classes_dict[p] = cls
        else:
            merged_on_level = data_dict
        # take percentage of data points from each hierarchical leaf class
        for cls, samples in merged_on_level.items():
            num_samples = len(samples)
            first_part_size = int((num_samples * percentage) // 100)
            second_part_size = num_samples - first_part_size
            if first_part_size == 0 or second_part_size == 0:
                log.warning(
                    "Data size of leaf class: {0} percentage: {1}".format(num_samples, percentage))
                log.warning(
                    "Class {} will not be represented in one part of the split.".format(cls))
            first_samples = np.random.choice(samples, replace=False,
                                             size=int((num_samples * percentage) // 100.0))
            second_samples = np.setdiff1d(samples, first_samples, assume_unique=True)

            if level < 4:
                for p in first_samples:
                    full_cls = prots2classes_dict[p]
                    first_data_dict[full_cls].append(p)

                for p in second_samples:
                    full_cls = prots2classes_dict[p]
                    second_data_dict[full_cls].append(p)
            else:
                first_data_dict[cls] = list(first_samples)
                second_data_dict[cls] = list(second_samples)

        return first_data_dict, second_data_dict

    @staticmethod
    def merge_data(data=None):
        """
        merges two or more data dictionaries into a single one.

        :param data: a list of data dictionaries with key - EC-class and value - list of protein
        codes.
        :return: a single data dictionary (a union over the input dictionaries)
        """
        if isinstance(data, list):
            all_keys = set(sum([d.keys() for d in data], []))
            merged_data_dict = {k: [] for k in all_keys}
            for d in data:
                for k in all_keys:
                    if k in d.keys():
                        merged_data_dict[k] += d[k]
            # remove eventual duplicates from the lists of elements for each key
            for k in all_keys:
                merged_data_dict[k] = list(set(merged_data_dict[k]))
            return merged_data_dict
        else:
            log.error("Provide a list of data dictionaries to be merged")
            raise ValueError


class EnzymeDataManager(DataManager):
    """
    EnzymeDataManager inherits from DataManager the *naive* and *sami-naive* splitting method and
    implements the essential management processes, required for the EnzymeCategory and
    ProteinDataBank data maintenance. Roughly the management pipeline can be described as:
        [download] -> [pre-process] -> [split test/train] -> provide
    where [.] designates a step that can be omitted if already done.
    """

    def __init__(self, data_dir,
                 force_download, force_memmaps,
                 force_grids, force_split,
                 grid_size,
                 enzyme_classes=None,
                 hierarchical_depth=4,
                 percentage_test=30,
                 percentage_val=30,
                 split_strategy='strict'):
        """
        :param data_dir: the path to the root data directory
        :param force_download: forces the downloading of the protein pdb files should be done
        :param force_memmaps: forces the memmapping of protein data, i.e. vdw-radii, atom coords. and charges
        :param force_grids: forces the 3D maps of electron density and potential generation
        :param force_split: forces the splitting into train/val/test sets
        :param grid_size: number of points on the side of the computed el. density grids
        :param enzyme_classes: a subset of EC given by a list of only those classes that should be considered
        :param hierarchical_depth: the maximal depth of prediction
        :param percentage_test: the portion of the data in % for the test set
        :param percentage_val: the portion of the data in % for the validation set
        """
        super(EnzymeDataManager, self).__init__(data_dir=data_dir,
                                                force_download=force_download,
                                                force_process=force_memmaps or force_grids,
                                                force_split=force_split,
                                                percentage_test=percentage_test,
                                                percentage_val=percentage_val)
        self.force_grids = force_grids or force_memmaps or force_download
        self.force_memmaps = force_memmaps or force_download
        self.grid_size = grid_size
        self.enzyme_classes = enzyme_classes
        self.max_hierarchical_depth = hierarchical_depth
        self.split_strategy = split_strategy

        self.validator = EnzymeValidator(enz_classes=enzyme_classes,
                                         dirs=self.dirs)
        self._setup_enzyme_data()

    def _setup_enzyme_data(self):
        """
        performs the abovementioned steps in the data management cycle.
        :return:
        """
        if self.enzyme_classes is None or not self.validator.check_naming(
                self.enzyme_classes):
            log.error("Unknown enzyme classes")
            raise ValueError

        # Download the data if required
        if self.force_download:
            ef = prep.EnzymeFetcher(categories=self.enzyme_classes,
                                    enzyme_dir=self.dirs['data_raw'])
            self.all_proteins = ef.fetch_enzymes()
            prep.download_pdbs(base_dir=self.dirs['data_raw'], protein_codes=self.all_proteins)
            save_pickle(file_path=os.path.join(self.dirs["data_raw"], "all_prot_codes.pickle"),
                        data=self.all_proteins)
            self._save_enzyme_list(target_dir=self.dirs["data_raw"],
                                   proteins_dict=self.all_proteins)
        else:
            log.info("Skipping downloading step")
            self.all_proteins = load_pickle(
                file_path=os.path.join(self.dirs["data_raw"],
                                       "all_prot_codes.pickle"))

        failed_downloads, n_successful, n_failed = self.validator.check_downloaded_codes()
        self._remove_failed_downloads(failed=failed_downloads)
        log.info("Total number of downloaded proteins found is {0}. Failed to download {1}".
                 format(n_successful, n_failed))

        # Process the data if required
        if self.force_memmaps or self.force_grids:
            edp = prep.EnzymeDataProcessor(protein_codes=self.all_proteins,
                                           from_dir=self.dirs['data_raw'],
                                           target_dir=self.dirs['data_processed'],
                                           grid_size=self.grid_size,
                                           process_grids=self.force_grids,
                                           process_memmaps=self.force_memmaps,
                                           use_esp=False)
            self.valid_proteins = edp.process()
            self.validator.check_class_representation(self.valid_proteins, clean_dict=True)
            save_pickle(
                file_path=os.path.join(self.dirs["data_processed"], "valid_prot_codes.pickle"),
                data=self.valid_proteins)
            self._save_enzyme_list(target_dir=self.dirs["data_processed"],
                                   proteins_dict=self.valid_proteins)
        else:
            log.info("Skipping preprocessing step")
            self.valid_proteins = load_pickle(
                file_path=os.path.join(self.dirs["data_processed"], "valid_prot_codes.pickle"))
            self.validator.check_class_representation(self.valid_proteins, clean_dict=True)

        # Split test / val data set if required
        if self.force_split:
            resp = raw_input(
                "Do you really want to split a test set into a separate directory?" +
                " This will change the existing test set / train set split! y/[n]\n")
            if resp.startswith('y'):
                if self.split_strategy == 'naive':
                    test_dataset, trainval_data = self.split_data_on_level(
                        self.valid_proteins,
                        percentage=self.p_test, level=3)
                    val_dataset, train_dataset = self.split_data_on_level(
                        trainval_data,
                        percentage=self.p_val, level=3)
                elif self.split_strategy == 'strict':
                    test_dataset, trainval_data = self.split_data_on_sublevel(
                        self.valid_proteins,
                        percentage=self.p_test, hierarchical_depth=4)
                    val_dataset, train_dataset = self.split_data_on_sublevel(
                        trainval_data,
                        percentage=self.p_val, hierarchical_depth=4)
                else:
                    log.error("Split strategy can be 'naive' or 'strict'")
                    raise ValueError

                self.validator.check_splitting(self.valid_proteins, trainval_data, test_dataset)
                self.validator.check_splitting(trainval_data, train_dataset, val_dataset)

                # recreate the train and test dirs
                shutil.rmtree(self.dirs['data_train'])
                os.makedirs(self.dirs['data_train'])
                shutil.rmtree(self.dirs['data_test'])
                os.makedirs(self.dirs['data_test'])

                # save val and train sets under dirs["data_train"], copy over all corresponding
                # data samples
                self._copy_processed(target_dir=self.dirs["data_train"],
                                     proteins_dict=trainval_data)
                self._save_enzyme_list(target_dir=self.dirs["data_train"],
                                       proteins_dict=trainval_data)
                save_pickle(file_path=[os.path.join(self.dirs["data_train"],
                                                    "train_prot_codes.pickle"),
                                       os.path.join(self.dirs["data_train"],
                                                    "val_prot_codes.pickle")],
                            data=[train_dataset, val_dataset])

                # save test set under dirs["data_test"], copy over all
                # corresponding data samples
                self._copy_processed(target_dir=self.dirs["data_test"],
                                     proteins_dict=test_dataset)
                self._save_enzyme_list(target_dir=self.dirs["data_test"],
                                       proteins_dict=test_dataset)
                save_pickle(file_path=os.path.join(self.dirs["data_test"],
                                                   "test_prot_codes.pickle"),
                            data=test_dataset)
            else:
                # only reinitialize the train and validation sets
                # the existing train and val pickles need to be merged and split
                # again
                train_dataset, val_dataset = load_pickle(
                    file_path=[os.path.join(self.dirs["data_train"],
                                            "train_prot_codes.pickle"),
                               os.path.join(self.dirs["data_train"],
                                            "val_prot_codes.pickle")])
                trainval_data = self.merge_data(
                    data=[train_dataset, val_dataset])

                # split them again
                val_dataset, train_dataset = self.split_data_on_level(
                    trainval_data,
                    percentage=self.p_val, level=3)

                self.validator.check_splitting(trainval_data, train_dataset, val_dataset)

                save_pickle(
                    file_path=[os.path.join(self.dirs["data_train"], "train_prot_codes.pickle"),
                               os.path.join(self.dirs["data_train"], "val_prot_codes.pickle")],
                    data=[train_dataset, val_dataset])
        else:
            log.info("Skipping splitting step")

        train_dataset, val_dataset, test_dataset = \
            load_pickle(file_path=[os.path.join(self.dirs["data_train"], "train_prot_codes.pickle"),
                                   os.path.join(self.dirs["data_train"], "val_prot_codes.pickle"),
                                   os.path.join(self.dirs["data_test"], "test_prot_codes.pickle")])

        # only select the enzymes classes we're interested in
        self.train_dataset = self._select_enzymes(train_dataset)
        self.val_dataset = self._select_enzymes(val_dataset)
        self.test_dataset = self._select_enzymes(test_dataset)

        # generate labels based on the data-sets
        lf = LabelFactory(self.train_dataset, self.val_dataset, self.test_dataset,
                          hierarchical_depth=self.max_hierarchical_depth)
        self.train_labels, self.val_labels, self.test_labels = lf.generate_hierarchical_labels()

        # final sanity check
        self.validator.check_labels(self.train_labels, self.val_labels, self.test_labels)

    def _select_enzymes(self, dataset):
        """
        Extracts a subset of a data dictionary according to the enzyme classes of interest.
        E.g. if the data dictionary contains the whole database and in the new experiment only a
        subset is needed, in order not to download, process and split the data again, a subset is
        extracted from the existing data.

        :param dataset: a data dictionary with keys the enzyme classes and values
        - lists of protein codes for each class
        :return: the subset of this data dictionary
        """
        filtered_set = dict()
        for cls, enzymes in dataset.items():
            if any([cls.startswith(enzyme_cls + '.') for enzyme_cls in
                    self.enzyme_classes]):
                filtered_set[cls] = enzymes
        return filtered_set

    def _remove_failed_downloads(self, failed=None):
        """
        Deprecated. It was meant to clean-up the list of all fetched proteins after the download is
        completed. This was necessary since the list of all proteins was generated from the EC
        database and the proteins were downloaded from the PDB database, hence some proteins might
        be taken into account by fail to download.

        :param failed: the list of all failed-to-download protein codes
        """
        # here the protein codes are stored in a dict according to their classes
        for cls in failed.keys():
            self.all_proteins[cls] = list(
                set(self.all_proteins[cls]) - set(failed[cls]))

    def get_training_set(self):
        return self.train_dataset, self.train_labels

    def get_validation_set(self):
        return self.val_dataset, self.val_labels

    def get_test_set(self):
        return self.test_dataset, self.test_labels

    def _copy_processed(self, target_dir, proteins_dict):
        """
        After the data is split, the test proteins are moved to a separate directory so that they do
        not interfere with the training and validation proteins. This method copies the proteins
        from one directory to another.

        :param target_dir: the target directory to which proteins are copied
        :param proteins_dict: the source directory from which proteins are copied
        :return:
        """
        src_dir = self.dirs["data_processed"]
        for prot_codes in proteins_dict.values():
            for prot_code in prot_codes:
                os.system("cp -R %s %s" % (
                    os.path.join(src_dir, prot_code.upper()),
                    os.path.join(target_dir, ".")))
                log.info("Copied {0} to {1}".format(prot_code, target_dir))

    @staticmethod
    def _save_enzyme_list(target_dir, proteins_dict):
        """
        Logger of the list of proteins, so that directories are not walked later when a list of all
        proteins in test or training set is needed.

        :param target_dir: the directory in which the lists should be stored
        :param proteins_dict: the data dictionary of protein classes and list of corresponding codes.
        """
        for cls, prot_codes in proteins_dict.items():
            with open(os.path.join(target_dir, cls + '.proteins'),
                      mode='w') as f:
                for prot_code in prot_codes:
                    f.write(prot_code + '\n')


class GOProteinsDataManager(DataManager):
    def __init__(self, data_dir, force_download=False, force_process=False,
                 force_split=False,
                 percentage_test=10,
                 percentage_val=20):
        super(GOProteinsDataManager, self).__init__(data_dir=data_dir,
                                                    force_download=force_download,
                                                    force_process=force_process,
                                                    force_split=force_split,
                                                    percentage_test=percentage_test,
                                                    percentage_val=percentage_val)

    def get_test_set(self):
        raise NotImplementedError

    def get_training_set(self):
        raise NotImplementedError

    def get_validation_set(self):
        raise NotImplementedError


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../data_test')
    # data_dir = "/usr/data/cvpr_shared/proteins/enzymes_w073/multichannel_density"
    # enzyme_classes = list()
    # for i in range(1, 100):
    #     enzyme_classes.append('1.%d' % i)
    # for i in range(1, 11):
    #     enzyme_classes.append('2.%d' % i)
    # for i in range(1, 14):
    #     enzyme_classes.append('3.%d' % i)
    # for i in range(1, 8):
    #     enzyme_classes.append('4.%d' % i)
    # enzyme_classes.append('4.99')
    # for i in range(1, 6):
    #     enzyme_classes.append('5.%d' % i)
    # enzyme_classes.append('5.99')
    # for i in range(1, 7):
    #     enzyme_classes.append('6.%d' % i)

    dm = EnzymeDataManager(data_dir=data_dir,
                           force_download=False,
                           force_memmaps=False,
                           force_grids=False,
                           force_split=True,
                           grid_size=64,
                           split_strategy='strict',
                           percentage_test=30,
                           percentage_val=30,
                           hierarchical_depth=3,
                           enzyme_classes=['3.13.1'])
