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
        raw_enzyme_lists = [x.strip('.proteins')
                            for x in os.listdir(self.dirs['data_raw'])
                            if x.endswith('.proteins')]
        missing_enzymes = dict()
        successful = 0
        failed = 0
        for enzyme_class in self.enzyme_classes:
            if not any(enzyme_class in end_class for end_class in raw_enzyme_lists):
                log.warning("Enzyme class {0} has not been downloaded".format(enzyme_class))
                num_errors += 1
            else:
                # for all leaf nodes check if their enzymes are there
                for enzyme_class_leaf in raw_enzyme_lists:
                    if not enzyme_class_leaf.startswith(enzyme_class):
                        continue
                    with open(os.path.join(self.dirs['data_raw'], enzyme_class_leaf + '.proteins')) \
                            as enz_class_file:
                        all_enzymes_in_class = [e.strip() for e in enz_class_file.readlines()]
                    # check if the codes are in the pdb folder
                    for e in all_enzymes_in_class:
                        if "pdb" + e.lower() + ".ent" not in raw_pdb_files:
                            failed += 1
                            log.warning("PDB file for enzyme {0} is not found (residing in class {1})"
                                        .format(e, enzyme_class_leaf))
                            if enzyme_class_leaf in missing_enzymes.keys():
                                missing_enzymes[enzyme_class_leaf].append(e.upper())
                            else:
                                missing_enzymes[enzyme_class_leaf] = [e.upper()]
                        else:
                            successful += 1

        return missing_enzymes, successful, failed

    def check_class_representation(self, data_dict, clean_dict=True, clean_duplicates=True):
        bad_keys = []
        duplicates = set()
        checked_classes = set()
        for cls, first_prots in data_dict.items():
            duplicates |= set(first_prots)
            checked_classes.add(cls)
            if clean_duplicates:
                for c, prots in data_dict.items():
                    if c not in checked_classes:
                        for p in prots[:]:
                            if p in duplicates:
                                prots.remove(p)

        for cls, prots in data_dict.items():
            if len(prots) == 0:
                log.warning("Class {0} is not represented".format(cls))
                if clean_dict:
                    bad_keys.append(cls)

        if clean_dict and len(bad_keys) > 0:
            log.warning("Class(es) %r will be deleted from the data dictionary" % bad_keys)
            for k in data_dict.keys():
                if k in bad_keys:
                    del data_dict[k]

    def check_splitting(self, all_proteins, first_partition, second_partition):
        # log.info("Checking the data splits for consistency and completeness")
        # leaf_classes = [x for x in os.listdir(self.dirs['data_processed']) if x.endswith('.proteins')]
        for cls in all_proteins.keys():
            # path_to_valid_cls = os.path.join(self.dirs['data_processed'], cls)
            # with open(path_to_valid_cls, 'r') as f:
            all_prot_codes_in_cls = all_proteins[cls]

            # path_to_train_cls = os.path.join(self.dirs['data_train'], cls)
            # with open(path_to_train_cls, 'r') as f:
            second_partition_prot_codes_in_cls = second_partition[cls]

            # path_to_test_cls = os.path.join(self.dirs['data_test'], cls)
            # with open(path_to_test_cls, 'r') as f:
            first_partition_prot_codes_in_cls = first_partition[cls]

            assert len(set(second_partition_prot_codes_in_cls)) + len(set(first_partition_prot_codes_in_cls)) == \
                   len(set(second_partition_prot_codes_in_cls + first_partition_prot_codes_in_cls)), \
                "The splits are not disjoint!"

            assert set(second_partition_prot_codes_in_cls + first_partition_prot_codes_in_cls) == \
                   set(all_prot_codes_in_cls), "The splits are not a partition of all proteins!"

    def check_labels(self, train_labels=None, val_labels=None, test_lables=None):
        log.warning("Label checking is not implemented yet")
        pass
