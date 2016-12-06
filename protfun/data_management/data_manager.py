import os
import re
import colorlog as log
import numpy as np

import protfun.data_management.preprocess as prep


class DataManager():
    """
    The data management cycle is: [[download] -> [preprocess] -> store] -> load
    Each datatype has its own _fetcher and _preprocessor
    """

    def __init__(self, data_dirname='data', data_type='enzyme_categorical',
                 force_download=False, force_process=False, force_split=True,
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
                     'data_proc': data_dir_processed,
                     'pdb_proc': os.path.join(data_dir_processed, "pdb"),
                     'go_proc': os.path.join(data_dir_processed, "go"),
                     'enzymes_proc': os.path.join(data_dir_processed, "enzymes"),
                     'moldata_proc': os.path.join(data_dir_processed, "moldata")}

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
        if not os.path.exists(self.dirs['data_proc']):
            os.makedirs(self.dirs['data_proc'])
        if not os.path.exists(self.dirs['pdb_proc']):
            os.makedirs(self.dirs['pdb_proc'])
        if not os.path.exists(self.dirs['go_proc']):
            os.makedirs(self.dirs['go_proc'])
        if not os.path.exists(self.dirs['enzymes_proc']):
            os.makedirs(self.dirs['enzymes_proc'])
        if not os.path.exists(self.dirs['moldata_proc']):
            os.makedirs(self.dirs['moldata_proc'])

        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../", data_dirname + "_test")
        self._test_dir = {'data': test_dir,
                          'pdb': os.path.join(test_dir, "pdb"),
                          'go': os.path.join(test_dir, "go"),
                          'enzymes': os.path.join(test_dir, "enzymes"),
                          'moldata': os.path.join(test_dir, "moldata")}

        if not os.path.exists(self._test_dir['data']):
            os.makedirs(self._test_dir['data'])
        if not os.path.exists(self._test_dir['pdb']):
            os.makedirs(self._test_dir['pdb'])
        if not os.path.exists(self._test_dir['go']):
            os.makedirs(self._test_dir['go'])
        if not os.path.exists(self._test_dir['enzymes']):
            os.makedirs(self._test_dir['enzymes'])
        if not os.path.exists(self._test_dir['moldata']):
            os.makedirs(self._test_dir['moldata'])

        self.checker = _SanityChecker(data_type=data_type,
                                      enz_classes=constraint_on,
                                      dirs=self.dirs)

        self.max_hierarchical_depth = hierarchical_depth

        if data_type == 'enzyme_categorical':
            # interpret the include variable as list of enzymes classes
            self.enzyme_classes = constraint_on

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

        elif data_type == 'protein_geneontological':
            self.all_protein_codes = []
            self.valid_protein_codes = []
            raise NotImplementedError
        else:
            log.error("Unknown data type. Possible values are"
                      " 'enzyme_categorical' and 'protein_geneontological'")
            raise ValueError

        if force_process:
            pp = prep.Preprocessor(protein_codes=self.all_protein_codes,
                                   data_path=self.dirs['pdb_raw'])
            self.valid_protein_codes = pp.process()
            self._store_valid()
            pass
        else:
            log.info("Skipping preprocessing step")

        if force_split:
            ds = _DataSplitter(trainval_dir=self.dirs, test_dir=self._test_dir,
                               percentage_test=p_test, percentage_val=p_val)
            ds.split()
        self.checker.check_splitting(test_dir=self._test_dir)

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

    def _store_valid(self):
        if self.data_type == 'enzyme_categorical':
            for cls in self.valid_protein_codes.keys():
                with open(os.path.join(self.dirs['enzymes_proc'], cls + '.proteins'), mode='w') as f:
                    for prot_code in self.valid_protein_codes[cls]:
                        f.write(prot_code + '\n')
                        pdb_filename_src = os.path.join(self.dirs['pdb_raw'],
                                                        'pdb' + prot_code.lower() + '.ent')
                        pdb_filename_dst = os.path.join(self.dirs['pdb_proc'],
                                                        'pdb' + prot_code.lower() + '.ent')
                        os.system("cp %s %s" % (pdb_filename_src, pdb_filename_dst))
                        if not os.path.isfile(pdb_filename_dst):
                            log.warning("Failed copying pdb file {0} from raw to proccessed".
                                        format(pdb_filename_src))
        else:
            raise NotImplemented

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

        :return: a dictionary of the training and validation dataset, containing a
        """
        data = dict()
        pass

    def load_test(self):
        raise NotImplementedError
        pass


class _SanityChecker():
    def __init__(self, data_type='enzyme_categorical', enz_classes=None, dirs=None):
        self.data_type = data_type
        self.enzyme_classes = enz_classes
        self.dirs = dirs

    def check_naming(self, classes):
        if self.data_type == 'enzyme_categorical':
            return sum([not bool(re.compile(r'[^0-9.]').search(cls)) for cls in classes]) == len(classes)
        else:
            raise NotImplementedError

    def check_downloaded_codes(self):
        log.info("Checking downloaded codes")
        num_errors = 0
        if self.data_type == 'enzyme_categorical':
            files_and_folders = os.listdir(self.dirs['enzymes_raw'])
            downloaded_pdbs = os.listdir(self.dirs['pdb_raw'])
            missing_enzymes = dict()
            for enzyme_class in self.enzyme_classes:
                if enzyme_class + '.proteins' not in files_and_folders:
                    log.warning("Enzyme class {0} has not been downloaded".format(enzyme_class))
                    num_errors += 1
                else:
                    with open(os.path.join(self.dirs['enzymes_raw'], enzyme_class + '.proteins')) \
                            as enz_class_file:
                        all_enzymes_in_class = [e.strip() for e in enz_class_file.readlines()]
                    # check if the codes are in the pdb folder
                    for e in all_enzymes_in_class:
                        if 'pdb' + e.lower() + '.ent' not in downloaded_pdbs:
                            log.warning("PDB file for enzyme {0} is not found (residing in class {1})"
                                        .format(e, enzyme_class))
                            if enzyme_class in missing_enzymes.keys():
                                missing_enzymes[enzyme_class].append(e.upper())
                            else:
                                missing_enzymes[enzyme_class] = [e.upper()]
                if num_errors > len(self.enzyme_classes) // 10:
                    log.error("More than 10% of the enzyme classes have not been downloaded. "
                              "Consider downloading and processing them anew")
                    return missing_enzymes

            return missing_enzymes

        else:
            raise NotImplementedError

    def check_splitting(self, test_dir=None):
        log.info("Checking the data splits for consistency and completeness")
        # take self.percentage_test data points from each hierarchical leaf class
        leaf_classes = [x for x in os.listdir(self.dirs['enzymes_proc']) if x.endswith('.proteins')]
        for cls in leaf_classes:
            path_to_cls = os.path.join(self.dirs['enzymes_proc'], cls)
            with open(path_to_cls, 'r') as f:
                prot_codes_trainval_in_cls = [pc.strip() for pc in f.readlines()]
            path_to_test_cls = os.path.join(test_dir['enzymes'], cls)
            with open(path_to_test_cls, 'r') as f:
                prot_codes_test_in_cls = [pc.strip() for pc in f.readlines()]
            if len(set(prot_codes_trainval_in_cls)) + len(set(prot_codes_test_in_cls)) != \
                len(set(prot_codes_trainval_in_cls + prot_codes_test_in_cls)):
                log.error("Incomplete or inconsistent trainval/test splitting!")

class _DataSplitter():
    def __init__(self,
                 trainval_dir=None,
                 test_dir=None,
                 percentage_test=10,
                 percentage_val=20):
        self.trainval_dir = trainval_dir
        self.test_dir = test_dir
        self.percentage_test = percentage_test
        self.percentage_val = percentage_val

    def split(self):
        self._split_and_lock_test()
        self._split_trainval()

    def _split_and_lock_test(self):
        if os.path.exists(self.test_dir['enzymes']):
            log.warning("It look like a test set has already been created. "
                        "Reinitializing may bias the actual performance of the network. "
                        "Are you sure you want to reinitialize it again? y/[n]")
            resp = raw_input()
            if not resp.startswith('y'):
                return
        log.info("Splitting data into test and trainval")
        # take self.percentage_test data points from each hierarchical leaf class
        leaf_classes = [x for x in os.listdir(self.trainval_dir['enzymes_proc']) if x.endswith('.proteins')]
        for cls in leaf_classes:
            path_to_cls = os.path.join(self.trainval_dir['enzymes_proc'], cls)
            with open(path_to_cls, 'r') as f:
                prot_codes_in_cls = [pc.strip() for pc in f.readlines()]
            test_indices = np.random.choice(np.arange(len(prot_codes_in_cls)),
                                            replace=False,
                                            size=int(len(prot_codes_in_cls)*(float(self.percentage_test)/100.0)))
            test_prot_codes_in_cls = [prot_codes_in_cls[i] for i in test_indices]
            trainval_indices = np.setdiff1d(np.arange(len(prot_codes_in_cls)), test_indices)
            trainval_prot_codes_in_cls = [prot_codes_in_cls[i] for i in trainval_indices]
            with open(path_to_cls, 'w') as f:
                for pc in trainval_prot_codes_in_cls:
                    f.write(pc + '\n')
            path_to_test_cls = os.path.join(self.test_dir['enzymes'], cls)
            with open(path_to_test_cls, 'w') as f:
                for pc in test_prot_codes_in_cls:
                    f.write(pc + '\n')

    def _split_trainval(self):
        pass


if __name__ == "__main__":
    dm = DataManager(data_dirname='experimental',
                     data_type='enzyme_categorical',
                     force_download=False,
                     force_process=False,
                     force_split=False,
                     p_test=50,
                     p_val=20,
                     hierarchical_depth=4,
                     constraint_on=['3.4.21.21', '3.4.21.34'])
    # NOTES: force_download works for enzymes

