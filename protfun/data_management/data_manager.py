import os
import colorlog as log

import protfun.data_management.preprocess as prep
from protfun.data_management.sanity_checker import _SanityChecker
from protfun.data_management.splitter import _DataSplitter


class DataManager():
    """
    The data management cycle is: [[download] -> [preprocess] -> store] -> load
    Each datatype has its own _fetcher and _preprocessor
    """

    def __init__(self, data_dirname='data', data_type='enzyme_categorical',
                 force_download=False, force_process=False, force_split=False, force_memmap=False,
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
                                    enzyme_classes=constraint_on)
        elif data_type == 'protein_geneontological':
            self._setup_geneont_data(force_download=force_download,
                                    force_process=force_process,
                                    force_split=force_split)
        else:
            log.error("Unknown data type. Possible values are"
                      " 'enzyme_categorical' and 'protein_geneontological'")
            raise ValueError


    def _setup_enzyme_data(self, force_download, force_process, force_split, force_memmap, enzyme_classes):
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
            self.valid_protein_codes = pp.process()
            self._store_valid()
        else:
            log.info("Skipping preprocessing step")

        if force_split:
            ds = _DataSplitter(data_dir=self.dirs,
                               percentage_test=self.p_test, percentage_val=self.p_val)

            resp = raw_input("Do you want to store a secret test data? y/[n]")
            if resp.startswith('y'):
                self._test_dir = ds.store_test_data()
                self.checker.check_splitting(test_dir=self._test_dir)
                prep.create_memmaps_for_enzymes(enzyme_dir=self._test_dir['enzymes'],
                                                moldata_dir=self._test_dir['moldata'],
                                                pdb_dir=self.dirs['pdb_proc'])
            train_dict, val_dict = ds.split_trainval()
        else:
            log.info("Skipping splitting step")

        if force_memmap:
            prep.create_memmaps_for_enzymes(enzyme_dir=self.dirs['enzymes_proc'],
                                            moldata_dir=self.dirs['moldata_proc'],
                                            pdb_dir=self.dirs['pdb_proc'])
        else:
            log.info("Skipping memmapping step")


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
        :return: a dictionary of the training and validation dataset, containing a
        """
        data = dict()
        pass


    def load_test(self):
        raise NotImplementedError


if __name__ == "__main__":
    dm = DataManager(data_dirname='experimental',
                     data_type='enzyme_categorical',
                     force_download=False,
                     force_process=False,
                     force_split=False,
                     force_memmap=False,
                     p_test=50,
                     p_val=20,
                     hierarchical_depth=4,
                     constraint_on=['3.4.21.21', '3.4.21.34'])
    # NOTES: force_download works for enzymes

