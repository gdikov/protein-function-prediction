import os
import colorlog as log
import cPickle
import protfun.data_management.preprocess as prep
from protfun.data_management.sanity_checker import _SanityChecker
from protfun.data_management.splitter import _DataSplitter
from protfun.data_management.label_factory import _LabelFactory
from protfun.data_management.preprocess.grid_processor import GridProcessor


class DataManager():
    """
    The data management cycle is: [[download] -> [preprocess] -> store] -> load
    Each datatype has its own _fetcher and _preprocessor
    """

    def __init__(self, data_dirname='data', data_type='enzyme_categorical',
                 force_download=False, force_process=False, force_split=False,
                 force_memmap=False, force_gridding=False,
                 constraint_on=None,
                 hierarchical_depth=4,
                 p_test=10,
                 p_val=20):

        self.data_type = data_type

        data_dir_raw = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../", data_dirname + "_raw")
        data_dir_processed = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "../../", data_dirname + "_processed")
        self.dirs = {'data_raw': data_dir_raw,
                     'pdb_raw': os.path.join(data_dir_raw, "pdb"),
                     'go_raw': os.path.join(data_dir_raw, "go"),
                     'enzymes_raw': os.path.join(data_dir_raw, "enzymes"),
                     'moldata_raw': os.path.join(data_dir_raw, "moldata"),
                     'data': data_dir_processed,
                     'pdb': os.path.join(data_dir_processed, "pdb"),
                     'go': os.path.join(data_dir_processed, "go"),
                     'enzymes': os.path.join(data_dir_processed, "enzymes"),
                     'moldata': os.path.join(data_dir_processed, "moldata")}

        if not os.path.exists(self.dirs['data_raw']):
            os.makedirs(self.dirs['data_raw'])
        if not os.path.exists(self.dirs['pdb_raw']):
            os.makedirs(self.dirs['pdb_raw'])
        if not os.path.exists(self.dirs['go_raw']):
            os.makedirs(self.dirs['go_raw'])
        if not os.path.exists(self.dirs['enzymes_raw']):
            os.makedirs(self.dirs['enzymes_raw'])
        if not os.path.exists(self.dirs['moldata_raw']):
            os.makedirs(self.dirs['moldata_raw'])
        if not os.path.exists(self.dirs['data']):
            os.makedirs(self.dirs['data'])
        if not os.path.exists(self.dirs['pdb']):
            os.makedirs(self.dirs['pdb'])
        if not os.path.exists(self.dirs['go']):
            os.makedirs(self.dirs['go'])
        if not os.path.exists(self.dirs['enzymes']):
            os.makedirs(self.dirs['enzymes'])
        if not os.path.exists(self.dirs['moldata']):
            os.makedirs(self.dirs['moldata'])

        self.checker = _SanityChecker(data_type=data_type,
                                      enz_classes=constraint_on,
                                      dirs=self.dirs)

        self.max_hierarchical_depth = hierarchical_depth
        self.p_val = p_val
        self.p_test = p_test

        if data_type == 'enzyme_categorical':
            self._setup_enzyme_data(force_download=force_download,
                                    force_process=force_process,
                                    force_split=force_split,
                                    force_memmap=force_memmap,
                                    force_gridding=force_gridding,
                                    enzyme_classes=constraint_on)
        elif data_type == 'protein_geneontological':
            self._setup_geneont_data(force_download=force_download,
                                    force_process=force_process,
                                    force_split=force_split)
        else:
            log.error("Unknown data type. Possible values are"
                      " 'enzyme_categorical' and 'protein_geneontological'")
            raise ValueError


    def _setup_enzyme_data(self, force_download, force_process, force_split,
                           force_memmap, force_gridding,
                           enzyme_classes):
        # interpret the include variable as list of enzymes classes
        self.enzyme_classes = enzyme_classes

        if self.enzyme_classes is None or not self.checker.check_naming(self.enzyme_classes):
            log.error("Unknown enzyme classes")
            raise ValueError

        if force_download:
            ef = prep.EnzymeFetcher(categories=self.enzyme_classes,
                                    enzyme_dir=self.dirs['enzymes_raw'])
            self.all_protein_codes = ef.fetch_enzymes()
            prep.download_pdbs(pdb_dirpath=self.dirs['pdb_raw'],
                               protein_codes=self.all_protein_codes)
        else:
            log.info("Skipping downloading step")
            self.all_protein_codes = self._load_fetched_codes()
        failed_downloads = self.checker.check_downloaded_codes()
        self._remove_failed(failed=failed_downloads)
        log.info("Total number of downloaded proteins found is {0}. Failed to download {1}".
                 format(len(self.all_protein_codes), len(failed_downloads)))

        if force_process:
            pp = prep.Preprocessor(protein_codes=self.all_protein_codes,
                                   data_path=self.dirs['pdb_raw'])
            self.valid_protein_codes, valid_proteins_info_dict = pp.process()
            self._store_valid(molecule_info=valid_proteins_info_dict)
        else:
            log.info("Skipping preprocessing step")

        if force_split:
            ds = _DataSplitter(data_dir=self.dirs,
                               percentage_test=self.p_test, percentage_val=self.p_val)

            resp = raw_input("Do you want to store a secret test data? y/[n]\n")
            if resp.startswith('y'):
                self._test_dir, test_dict = ds.store_test_data()
                self.checker.check_splitting(test_dir=self._test_dir)
                prep.create_memmaps_for_enzymes(enzyme_dir=self._test_dir['enzymes'],
                                                moldata_dir=self._test_dir['moldata'],
                                                pdb_dir=self.dirs['pdb'])
            ds.split_trainval()
            train_dict, val_dict, test_dict = self._load_train_val_test_data_pickles()
            lf = _LabelFactory(train_dict, val_dict, test_dict,
                               hierarchical_depth=self.max_hierarchical_depth)
            train_labels, val_labels, test_labels, encoding = lf.generate_hierarchical_labels()
            self.checker.check_labels(train_labels, val_labels, test_labels)
            self._store_labels(train_labels, val_labels, test_labels, encoding)
        else:
            log.info("Skipping splitting step")

        if force_memmap:
            prep.create_memmaps_for_enzymes(enzyme_dir=self.dirs['enzymes'],
                                            moldata_dir=self.dirs['moldata'],
                                            pdb_dir=self.dirs['pdb'])
        else:
            log.info("Skipping memmapping step")

        if force_gridding:
            train_labels, val_labels, test_labels = self._load_train_val_test_label_pickles()
            trainval_pdbs = train_labels.keys() + val_labels.keys()
            test_pdbs = test_labels.keys()
            trainval_gp = GridProcessor(data_dirs=self.dirs, force_process=True)
            trainval_gp.process_all(trainval_pdbs)
            if force_split:
                test_gp = GridProcessor(data_dirs=self._test_dir, force_process=True)
                test_gp.process_all(test_pdbs)
        else:
            log.info("Skipping grid processing step")


    def _setup_geneont_data(self, force_download, force_process, force_split):
        self.all_protein_codes = []
        self.valid_protein_codes = []
        raise NotImplementedError


    def _load_fetched_codes(self):
        if self.data_type == 'enzyme_categorical':
            protein_codes = dict()
            for cl in self.enzyme_classes:
                try:
                    with open(os.path.join(self.dirs['enzymes_raw'], cl + '.proteins'), mode='r') as f:
                        new_protein_codes = [e.strip() for e in f.readlines()]
                        if new_protein_codes is None:
                            log.warning("Enzyme class {0} contains 0 protein codes".format(cl))
                            protein_codes[cl] = []
                        else:
                            protein_codes[cl] = new_protein_codes
                except EnvironmentError:
                    log.error("One or more enzyme classes have not been downloaded. "
                              "Re-run with force_download=True")
                    raise IOError
            return protein_codes
        else:
            protein_codes = []
            raise NotImplementedError


    def _load_train_val_test_data_pickles(self):
        path_to_train = os.path.join(self.dirs['enzymes'], 'train_data.pickle')
        path_to_val = os.path.join(self.dirs['enzymes'], 'val_data.pickle')
        path_to_test = os.path.join(self._test_dir['enzymes'], 'test_data.pickle')

        if not os.path.exists(path_to_train):
            log.error("No previously saved train data found. "
                      "Re-run with force_split=True")
            raise IOError
        else:
            with open(path_to_train, 'r') as tr:
                train_dict = cPickle.load(tr)

        if not os.path.exists(path_to_val):
            log.error("No previously saved validation data found. "
                      "Re-run with force_split=True")
            raise IOError
        else:
            with open(path_to_val, 'r') as val:
                val_dict = cPickle.load(val)

        if not os.path.exists(path_to_test):
            log.error("No previously saved validation data found. "
                      "Re-run with force_split=True")
            raise IOError
        else:
            with open(path_to_test, 'r') as tst:
                test_dict = cPickle.load(tst)

        return train_dict, val_dict, test_dict


    def _load_train_val_test_label_pickles(self):
        path_to_train = os.path.join(self.dirs['enzymes'], 'train_labels.pickle')
        path_to_val = os.path.join(self.dirs['enzymes'], 'val_labels.pickle')
        path_to_test = os.path.join(self._test_dir['enzymes'], 'test_labels.pickle')

        if not os.path.exists(path_to_train):
            log.error("No previously saved train labels found. "
                      "Re-run with force_split=True")
            raise IOError
        else:
            with open(path_to_train, 'r') as tr:
                train_labels = cPickle.load(tr)

        if not os.path.exists(path_to_val):
            log.error("No previously saved validation data found. "
                      "Re-run with force_split=True")
            raise IOError
        else:
            with open(path_to_val, 'r') as val:
                val_labels = cPickle.load(val)

        if not os.path.exists(path_to_test):
            log.error("No previously saved validation data found. "
                      "Re-run with force_split=True")
            raise IOError
        else:
            with open(path_to_test, 'r') as tst:
                test_labels = cPickle.load(tst)

        return train_labels, val_labels, test_labels


    def _store_labels(self, train_labels, val_labels, test_labels, label_encoding):
        with open(os.path.join(self.dirs['enzymes'], 'train_labels.pickle'), 'wb') as f:
            cPickle.dump(train_labels, f)
        with open(os.path.join(self.dirs['enzymes'], 'val_labels.pickle'), 'wb') as f:
            cPickle.dump(val_labels, f)
        with open(os.path.join(self._test_dir['enzymes'], 'test_labels.pickle'), 'wb') as f:
            cPickle.dump(test_labels, f)
        with open(os.path.join(self.dirs['enzymes'], 'label_encoding.pickle'), 'wb') as f:
            cPickle.dump(label_encoding, f)


    def _store_valid(self, molecule_info):
        if self.data_type == 'enzyme_categorical':
            for cls in self.valid_protein_codes.keys():
                with open(os.path.join(self.dirs['enzymes'], cls + '.proteins'), mode='w') as f:
                    for prot_code in self.valid_protein_codes[cls]:
                        f.write(prot_code + '\n')
                        pdb_filename_src = os.path.join(self.dirs['pdb_raw'],
                                                        'pdb' + prot_code.lower() + '.ent')
                        pdb_filename_dst = os.path.join(self.dirs['pdb'],
                                                        'pdb' + prot_code.lower() + '.ent')
                        os.system("cp %s %s" % (pdb_filename_src, pdb_filename_dst))
                        if not os.path.isfile(pdb_filename_dst):
                            log.warning("Failed copying pdb file {0} from raw to proccessed".
                                        format(pdb_filename_src))
            with open(os.path.join(self.dirs['pdb'], 'mol_info.pickle'), 'wb') as f:
                cPickle.dump(molecule_info, f)
        else:
            raise NotImplementedError


    def _remove_failed(self, failed=None):
        if self.data_type == 'enzyme_categorical':
            # here the protein codes are stored in a dict according to their classes
            for cls in failed.keys():
                self.all_protein_codes[cls] = list(set(self.all_protein_codes[cls]) - set(failed[cls]))
        else:
            # here the protein codes are stored in a list as there are no classes
            self.all_protein_codes = list(set(self.all_protein_codes) - set(failed))


    def load_trainval(self):
        """
        Unpickle the previously stored data and labels and pack in a dictionary
        :return: a dictionary of the training and validation dataset
        """
        train_dict, val_dict, _ = self._load_train_val_test_data_pickles()
        train_lables, val_labels, _ = self._load_train_val_test_label_pickles()
        data = {'x_train': train_dict, 'y_train':train_lables,
                'x_val': val_dict, 'y_val': val_labels}
        return data


    def load_test(self):
        """
       Unpickle the previously stored data and labels and pack in a dictionary
       :return: a dictionary of the training and validation dataset
       """
        _, _, test_dict = self._load_train_val_test_data_pickles()
        _, _, test_labels = self._load_train_val_test_label_pickles()
        test_data = {'x_test': test_dict, 'y_test': test_labels}
        return test_data


if __name__ == "__main__":
    dm = DataManager(data_dirname='experimental',
                     data_type='enzyme_categorical',
                     force_download=True,
                     force_process=True,
                     force_split=True,
                     force_memmap=True,
                     p_test=50,
                     p_val=50,
                     hierarchical_depth=4,
                     constraint_on=['3.4.21.21', '3.4.21.34'])
    # NOTES: force_download works for enzymes

