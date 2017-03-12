import os
import ntpath
import re
from glob import glob
import itertools

from protfun.utils.log import get_logger

log = get_logger("validations")


class EnzymeValidator(object):
    """
    EnzymeValidator has the task to validate the correctness and completeness of the essential data management steps,
    e.g. downloading and splitting. This should help finding bugs.
    """

    def __init__(self, enz_classes=None, dirs=None):
        self.enzyme_classes = enz_classes
        self.dirs = dirs

    def check_naming(self, classes):
        """
        checks if the EC classes listed comply with the naming convention, e.g. 1.1.1.1

        :param classes: a list of the EC classes
        :return:
        """
        return sum([not bool(re.compile(r'[^0-9.]').search(cls)) for cls in
                    classes]) == len(classes)

    def check_downloaded_codes(self):
        """
        verifies if the to-be-downloaded proteins have actually been downloaded

        :return: a tuple of a list of the missing protein codes, the number of successfully downloaded and failed.
        """
        log.info("Checking downloaded proteins")
        num_errors = 0
        raw_pdb_files = [ntpath.basename(y) for x in
                         os.walk(self.dirs['data_raw']) for y in
                         glob(os.path.join(x[0], '*.ent'))]
        raw_enzyme_lists = [x.strip('.proteins')
                            for x in os.listdir(self.dirs['data_raw'])
                            if x.endswith('.proteins')]
        missing_enzymes = dict()
        successful = 0
        failed = 0
        for enzyme_class in self.enzyme_classes:
            if not any(enzyme_class in end_class for end_class in
                       raw_enzyme_lists):
                log.warning("Enzyme class {0} has not been downloaded".format(
                    enzyme_class))
                num_errors += 1
            else:
                # for all leaf nodes check if their enzymes are there
                for enzyme_class_leaf in raw_enzyme_lists:
                    if not enzyme_class_leaf.startswith(enzyme_class):
                        continue
                    with open(os.path.join(self.dirs['data_raw'],
                                           enzyme_class_leaf + '.proteins')) \
                            as enz_class_file:
                        all_enzymes_in_class = [e.strip() for e in
                                                enz_class_file.readlines()]
                    # check if the codes are in the pdb folder
                    for e in all_enzymes_in_class:
                        if "pdb" + e.lower() + ".ent" not in raw_pdb_files:
                            failed += 1
                            log.warning(
                                "PDB file for enzyme {0} is not found (residing in class {1})"
                                    .format(e, enzyme_class_leaf))
                            if enzyme_class_leaf in missing_enzymes.keys():
                                missing_enzymes[enzyme_class_leaf].append(
                                    e.upper())
                            else:
                                missing_enzymes[enzyme_class_leaf] = [e.upper()]
                        else:
                            successful += 1

        return missing_enzymes, successful, failed

    def check_class_representation(self, data_dict, clean_dict=True,
                                   clean_duplicates=True):
        """
        checks if there are classes with no proteins inside and removes them from the list of all classes.
        removes proteins from two different classes so that they do not end up in train and test sets after splitting.

        :param data_dict: the main data dictionary
        :param clean_dict: a flag whether the dictionary should be cleaned from empty classes
        :param clean_duplicates: a flag whether the dictionry should be cleaned from duplicates
        :return:
        """
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
            log.warning(
                "Class(es) %r will be deleted from the data dictionary" % bad_keys)
            for k in data_dict.keys():
                if k in bad_keys:
                    del data_dict[k]

    def check_splitting(self, all_proteins, first_partition, second_partition):
        """
        checks if the splits are disjoint and complete, i.e. their union amounts to the whole dataset

        :param all_proteins: the data dict of all proteins that has been split
        :param first_partition: the first data partition
        :param second_partition: the second data partition
        :return:
        """

        # flatten all protein codes
        all_prot_codes = set(list(itertools.chain.from_iterable(all_proteins.values())))
        # flatten first part protein codes
        first_prot_codes = set(list(itertools.chain.from_iterable(first_partition.values())))
        # flatten second part protein codes
        second_prot_codes = set(list(itertools.chain.from_iterable(second_partition.values())))

        # check that the intersection is empty
        assert len(first_prot_codes & second_prot_codes) == 0, "The splits are not disjoint!"

        # check that the union gives the full set
        assert (first_prot_codes | second_prot_codes) == all_prot_codes, \
            "The splits are not a partition of all proteins!"

        if len(first_prot_codes) == 0 or len(second_prot_codes) == 0:
            log.error("During data set splitting, one part contains 0 samples. "
                      "Please choose a bigger data set or a different split strategy")
            log.error("Protein codes in first part: {}".format(first_prot_codes))
            log.error("Protein codes in second part: {}".format(second_prot_codes))
            raise ValueError
