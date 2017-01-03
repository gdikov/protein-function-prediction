import abc
import os
import shutil
import colorlog as log
import cPickle
import numpy as np
import protfun.data_management.preprocess as prep
from protfun.data_management.validation import EnzymeValidator
from protfun.data_management.label_factory import LabelFactory


class DataManager(object):
    __metaclass__ = abc.ABCMeta
    """
    The data management cycle is: [[download] -> [preprocess] -> [split test/train]] -> provide
    Each datatype has its own _fetcher and _preprocessor
    """

    def __init__(self, data_dir,
                 force_download=False, force_process=False, force_split=False,
                 percentage_test=10, percentage_val=20):
        self.data_dirname = data_dir
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
            if not os.path.exists(d):
                os.makedirs(d)

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
    def split_data(data_dict, percentage):
        if percentage > 100:
            log.error("Bad percentage number. Must be in [0, 100]")
            raise ValueError

        first_data_dict = dict()
        second_data_dict = dict()

        # take percentage of data points from each hierarchical leaf class
        for cls, samples in data_dict.items():
            num_samples = len(samples)
            first_part_size = int((num_samples * percentage) // 100)
            second_part_size = num_samples - first_part_size
            if first_part_size == 0 or second_part_size == 0:
                log.warning("Data size of leaf class: {0} percentage: {1}".format(num_samples, percentage))
                log.warning("Class {} will not be represented in one part of the split.".format(cls))
            first_samples = np.random.choice(samples,
                                             replace=False,
                                             size=int((num_samples * percentage) // 100.0))
            second_samples = np.setdiff1d(samples, first_samples, assume_unique=True)
            first_data_dict[cls] = list(first_samples)
            second_data_dict[cls] = list(second_samples)

        return first_data_dict, second_data_dict

    @staticmethod
    def merge_data(data=None):
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
    def __init__(self, data_dir,
                 force_download=False, force_memmaps=False,
                 force_grids=False, force_split=False,
                 enzyme_classes=None,
                 hierarchical_depth=4,
                 percentage_test=10,
                 percentage_val=20):
        super(EnzymeDataManager, self).__init__(data_dir=data_dir, force_download=force_download,
                                                force_process=force_memmaps or force_grids, force_split=force_split,
                                                percentage_test=percentage_test, percentage_val=percentage_val)
        self.force_grids = force_grids or force_memmaps or force_download
        self.force_memmaps = force_memmaps or force_download
        self.enzyme_classes = enzyme_classes
        self.max_hierarchical_depth = hierarchical_depth

        self.validator = EnzymeValidator(enz_classes=enzyme_classes,
                                         dirs=self.dirs)
        self._setup_enzyme_data()

    def _setup_enzyme_data(self):
        if self.enzyme_classes is None or not self.validator.check_naming(self.enzyme_classes):
            log.error("Unknown enzyme classes")
            raise ValueError

        # Download the data if required
        if self.force_download:
            ef = prep.EnzymeFetcher(categories=self.enzyme_classes,
                                    enzyme_dir=self.dirs['data_raw'])
            self.all_proteins = ef.fetch_enzymes()
            prep.download_pdbs(base_dir=self.dirs['data_raw'],
                               protein_codes=self.all_proteins)
            self._save_pickle(file_path=os.path.join(self.dirs["data_raw"], "all_prot_codes.pickle"),
                              data=self.all_proteins)
            self._save_enzyme_list(target_dir=self.dirs["data_raw"], proteins_dict=self.all_proteins)
        else:
            log.info("Skipping downloading step")
            self.all_proteins = self._load_pickle(
                file_path=os.path.join(self.dirs["data_raw"], "all_prot_codes.pickle"))

        failed_downloads, n_successful, n_failed = self.validator.check_downloaded_codes()
        self._remove_failed_downloads(failed=failed_downloads)
        log.info("Total number of downloaded proteins found is {0}. Failed to download {1}".
                 format(n_successful, n_failed))

        # Process the data if required
        if self.force_memmaps or self.force_grids:
            edp = prep.EnzymeDataProcessor(protein_codes=self.all_proteins,
                                           from_dir=self.dirs['data_raw'],
                                           target_dir=self.dirs['data_processed'],
                                           process_grids=self.force_grids,
                                           process_memmaps=self.force_memmaps)
            self.valid_proteins = edp.process()
            self.validator.check_class_representation(self.valid_proteins, clean_dict=True)
            self._save_pickle(file_path=os.path.join(self.dirs["data_processed"], "valid_prot_codes.pickle"),
                              data=self.valid_proteins)
            self._save_enzyme_list(target_dir=self.dirs["data_processed"], proteins_dict=self.valid_proteins)
        else:
            log.info("Skipping preprocessing step")
            self.valid_proteins = self._load_pickle(file_path=os.path.join(self.dirs["data_processed"],
                                                                           "valid_prot_codes.pickle"))
            self.validator.check_class_representation(self.valid_proteins, clean_dict=True)

        # Split test / val data set if required
        if self.force_split:
            resp = raw_input("Do you really want to split a test set into a separate directory?" +
                             " This will change the existing test set / train set split! y/[n]\n")
            if resp.startswith('y'):
                test_dataset, trainval_data = self.split_data(self.valid_proteins, percentage=self.p_test)
                val_dataset, train_dataset = self.split_data(trainval_data, percentage=self.p_val)

                # recreate the train and test dirs
                shutil.rmtree(self.dirs['data_train'])
                os.makedirs(self.dirs['data_train'])
                shutil.rmtree(self.dirs['data_test'])
                os.makedirs(self.dirs['data_test'])

                # save val and train sets under dirs["data_train"], copy over all corresponding data samples
                self._copy_processed(target_dir=self.dirs["data_train"], proteins_dict=trainval_data)
                self._save_enzyme_list(target_dir=self.dirs["data_train"], proteins_dict=trainval_data)
                self._save_pickle(file_path=[os.path.join(self.dirs["data_train"], "train_prot_codes.pickle"),
                                             os.path.join(self.dirs["data_train"], "val_prot_codes.pickle")],
                                  data=[train_dataset, val_dataset])

                # save test set under dirs["data_test"], copy over all corresponding data samples
                self._copy_processed(target_dir=self.dirs["data_test"], proteins_dict=test_dataset)
                self._save_enzyme_list(target_dir=self.dirs["data_test"], proteins_dict=test_dataset)
                self._save_pickle(file_path=os.path.join(self.dirs["data_test"], "test_prot_codes.pickle"),
                                  data=test_dataset)

                self.validator.check_splitting(self.valid_proteins, trainval_data, test_dataset)
            else:
                # only reinitialize the train and validation sets
                # the existing train and val pickles need to be merged and split again
                train_dataset, val_dataset = self._load_pickle(file_path=[os.path.join(self.dirs["data_train"],
                                                                                       "train_prot_codes.pickle"),
                                                                          os.path.join(self.dirs["data_train"],
                                                                                       "val_prot_codes.pickle")])
                trainval_data = self.merge_data(data=[train_dataset, val_dataset])

                # split them again
                val_dataset, train_dataset = self.split_data(trainval_data, percentage=self.p_val)
                self._save_pickle(file_path=[os.path.join(self.dirs["data_train"], "train_prot_codes.pickle"),
                                             os.path.join(self.dirs["data_train"], "val_prot_codes.pickle")],
                                  data=[train_dataset, val_dataset])

            test_dataset = self._load_pickle(file_path=os.path.join(self.dirs["data_test"], "test_prot_codes.pickle"))

            # generate labels based on the new splits
            lf = LabelFactory(train_dataset, val_dataset, test_dataset,
                              hierarchical_depth=self.max_hierarchical_depth)
            self.train_labels, self.val_labels, self.test_labels, encoding = lf.generate_hierarchical_labels()
            self._save_pickle(file_path=os.path.join(self.dirs['misc'], "label_encoding.pickle"),
                              data=encoding)

            # pickle the generated labels
            self._save_pickle(file_path=[os.path.join(self.dirs["data_train"], "train_labels.pickle"),
                                         os.path.join(self.dirs["data_train"], "val_labels.pickle"),
                                         os.path.join(self.dirs["data_test"], "test_labels.pickle")],
                              data=[self.train_labels, self.val_labels, self.test_labels])

            # final sanity check
            self.validator.check_labels(self.train_labels, self.val_labels, self.test_labels)
        else:
            log.info("Skipping splitting step")

        self.train_dataset, self.val_dataset, self.test_dataset = \
            self._load_pickle(file_path=[os.path.join(self.dirs["data_train"], "train_prot_codes.pickle"),
                                         os.path.join(self.dirs["data_train"], "val_prot_codes.pickle"),
                                         os.path.join(self.dirs["data_test"], "test_prot_codes.pickle")])

        self.train_labels, self.val_labels, self.test_labels = \
            self._load_pickle(file_path=[os.path.join(self.dirs["data_train"], "train_labels.pickle"),
                                         os.path.join(self.dirs["data_train"], "val_labels.pickle"),
                                         os.path.join(self.dirs["data_test"], "test_labels.pickle")])

    def _remove_failed_downloads(self, failed=None):
        # here the protein codes are stored in a dict according to their classes
        for cls in failed.keys():
            self.all_proteins[cls] = list(set(self.all_proteins[cls]) - set(failed[cls]))

    def get_training_set(self):
        return self.train_dataset, self.train_labels

    def get_validation_set(self):
        return self.val_dataset, self.val_labels

    def get_test_set(self):
        return self.test_dataset, self.test_labels

    def _copy_processed(self, target_dir, proteins_dict):
        src_dir = self.dirs["data_processed"]
        for prot_codes in proteins_dict.values():
            for prot_code in prot_codes:
                os.system("cp -R %s %s" % (os.path.join(src_dir, prot_code.upper()), os.path.join(target_dir, ".")))
                log.info("Copied {0} to {1}".format(prot_code, target_dir))

    @staticmethod
    def _save_enzyme_list(target_dir, proteins_dict):
        for cls, prot_codes in proteins_dict.items():
            with open(os.path.join(target_dir, cls + '.proteins'), mode='w') as f:
                for prot_code in prot_codes:
                    f.write(prot_code + '\n')

    @staticmethod
    def _save_pickle(file_path, data):
        if isinstance(data, list) and isinstance(file_path, list):
            if len(data) == len(file_path):
                for path, dat in zip(file_path, data):
                    with open(path, 'wb') as f:
                        cPickle.dump(dat, f)
            else:
                log.error("File paths are not matching the number of objects to save")
                raise ValueError
        else:
            with open(file_path, 'wb') as f:
                cPickle.dump(data, f)

    @staticmethod
    def _load_pickle(file_path):
        def _load_one(path):
            if not os.path.exists(path):
                log.error("No data was saved in {0}.".format(path))
                raise IOError
            else:
                with open(path, 'r') as f:
                    data = cPickle.load(f)
                return data

        if isinstance(file_path, list):
            objs = []
            for path in file_path:
                unpickled = _load_one(path)
                objs.append(unpickled)
            return objs
        else:
            return _load_one(file_path)


class GOProteinsDataManager(DataManager):
    def __init__(self, data_dir, force_download=False, force_process=False, force_split=False,
                 percentage_test=10,
                 percentage_val=20):
        super(GOProteinsDataManager, self).__init__(data_dir=data_dir, force_download=force_download,
                                                    force_process=force_process, force_split=force_split,
                                                    percentage_test=percentage_test, percentage_val=percentage_val)

    def get_test_set(self):
        raise NotImplementedError

    def get_training_set(self):
        raise NotImplementedError

    def get_validation_set(self):
        raise NotImplementedError


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '../../data_new')
    dm = EnzymeDataManager(data_dir=data_dir,
                           force_download=False,
                           force_memmaps=False,
                           force_grids=False,
                           force_split=True,
                           percentage_test=20,
                           percentage_val=15,
                           hierarchical_depth=4,
                           enzyme_classes=['3.4.21', '3.4.24'])
    print(dm.train_dataset)
    print(dm.train_labels)
    print(dm.val_dataset)
    print(dm.val_labels)
    print(dm.test_dataset)
    print(dm.test_labels)
