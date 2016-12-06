import os
import colorlog as log
import re

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
