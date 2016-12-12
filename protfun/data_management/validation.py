import os
import ntpath
import colorlog as log
import re
from glob import glob


class EnzymeValidator(object):
    def __init__(self, enz_classes=None, dirs=None):
        self.enzyme_classes = enz_classes
        self.dirs = dirs

    def check_naming(self, classes):
        return sum([not bool(re.compile(r'[^0-9.]').search(cls)) for cls in classes]) == len(classes)

    def check_downloaded_codes(self):
        log.info("Checking downloaded proteins")
        num_errors = 0
        raw_pdb_files = [ntpath.basename(y) for x in os.walk(self.dirs['data_raw']) for y in
                         glob(os.path.join(x[0], '*.ent'))]
        raw_enzyme_lists = [x for x in os.listdir(self.dirs['data_raw']) if x.endswith('.proteins')]
        missing_enzymes = dict()
        successful = 0
        failed = 0
        for enzyme_class in self.enzyme_classes:
            if enzyme_class + '.proteins' not in raw_enzyme_lists:
                log.warning("Enzyme class {0} has not been downloaded".format(enzyme_class))
                num_errors += 1
            else:
                with open(os.path.join(self.dirs['data_raw'], enzyme_class + '.proteins')) \
                        as enz_class_file:
                    all_enzymes_in_class = [e.strip() for e in enz_class_file.readlines()]
                # check if the codes are in the pdb folder
                for e in all_enzymes_in_class:
                    if "pdb" + e.lower() + ".ent" not in raw_pdb_files:
                        failed += 1
                        log.warning("PDB file for enzyme {0} is not found (residing in class {1})"
                                    .format(e, enzyme_class))
                        if enzyme_class in missing_enzymes.keys():
                            missing_enzymes[enzyme_class].append(e.upper())
                        else:
                            missing_enzymes[enzyme_class] = [e.upper()]
                    else:
                        successful += 1
            if num_errors > len(self.enzyme_classes) // 10:
                log.error("More than 10% of the enzyme classes have not been downloaded. "
                          "Consider downloading and processing them anew")
                return missing_enzymes, successful, failed

        return missing_enzymes, successful, failed


    def check_class_representation(self, data_dict, clean_dict=True):
        bad_keys = []
        for cls, prots in data_dict.items():
            if len(prots) == 0:
                log.warning("Class {0} is not represented".format(cls))
                if clean_dict:
                    bad_keys.append(cls)
        if clean_dict:
            log.warning("Class(es) %r will be deleted from the data dictionary" %bad_keys)
            for k in data_dict.keys():
                if k in bad_keys:
                    del data_dict[k]


    def check_splitting(self):
        log.info("Checking the data splits for consistency and completeness")
        leaf_classes = [x for x in os.listdir(self.dirs['data_processed']) if x.endswith('.proteins')]
        for cls in leaf_classes:
            path_to_valid_cls = os.path.join(self.dirs['data_processed'], cls)
            with open(path_to_valid_cls, 'r') as f:
                valid_prot_codes_in_cls = [pc.strip() for pc in f.readlines()]

            path_to_train_cls = os.path.join(self.dirs['data_train'], cls)
            with open(path_to_train_cls, 'r') as f:
                train_prot_codes_in_cls = [pc.strip() for pc in f.readlines()]

            path_to_test_cls = os.path.join(self.dirs['data_test'], cls)
            with open(path_to_test_cls, 'r') as f:
                test_prot_codes_in_cls = [pc.strip() for pc in f.readlines()]

            assert len(set(train_prot_codes_in_cls)) + len(set(test_prot_codes_in_cls)) == \
                   len(set(train_prot_codes_in_cls + test_prot_codes_in_cls)), "Train and test set are not disjoint!"

            assert set(train_prot_codes_in_cls + test_prot_codes_in_cls) == \
                   set(valid_prot_codes_in_cls), "Train and test set are not a partition of all processed proteins "

    def check_labels(self, train_labels=None, val_labels=None, test_lables=None):
        log.warning("Label checking is not implemented yet")
        pass
