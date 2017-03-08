import numpy as np


class LabelFactory(object):
    """
    LabelFactory is generating the labels from the hierarchical structure of the data classes.
    """

    def __init__(self, train_dict, val_dict, test_dict, hierarchical_depth=1):
        """
        a data dictionady is a dict with keys - classes and values - list of protein codes

        :param train_dict: data dictionary for the training set
        :param val_dict: data dictionary for the validation set
        :param test_dict: data dictionary for the test set
        :param hierarchical_depth: the hierarchical depth of prediction
        """
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        self.h_depth = hierarchical_depth

    def generate_hierarchical_labels(self):
        """
        Generates a hierarchical labels consisting in one-hot encoded vectors for each class-level.
        If a protein is present in more classes then the one-hot labels are merged. For example [0,0,1] and [0,1,0]
        will result in [0,1,1].

        ::Example
            >>> tr = {'3.1.1': ['a'], '2.1.2': ['b', 'a', 'c'],
            >>>       '2.1.3': ['d', 'e'], '2.1.4': ['f', 'g']}
            >>> va = {'3.1.2': ['h'], '2.1.2': ['i']}
            >>> te = {'3.1.1': ['j'], '2.1.8': ['k']}
            >>> lf = LabelFactory(tr, va, te, 3)
            >>> tr, va, te = lf.generate_hierarchical_labels()
            >>> print(tr)
            >>> print(va)
            >>> print(te)

            >>> {'a': [array([1, 1], dtype=int32), array([1, 1], dtype=int32), array([1, 0, 0, 0, 1, 0], dtype=int32)],
            >>>  'c': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([1, 0, 0, 0, 0, 0], dtype=int32)],
            >>>  'b': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([1, 0, 0, 0, 0, 0], dtype=int32)],
            >>>  'e': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([0, 1, 0, 0, 0, 0], dtype=int32)],
            >>>  'd': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([0, 1, 0, 0, 0, 0], dtype=int32)],
            >>>  'g': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([0, 0, 1, 0, 0, 0], dtype=int32)],
            >>>  'f': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([0, 0, 1, 0, 0, 0], dtype=int32)]}

            >>> {'i': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([1, 0, 0, 0, 0, 0], dtype=int32)],
            >>>  'h': [array([0, 1], dtype=int32), array([0, 1], dtype=int32), array([0, 0, 0, 0, 0, 1], dtype=int32)]}
            >>> {'k': [array([1, 0], dtype=int32), array([1, 0], dtype=int32), array([0, 0, 0, 1, 0, 0], dtype=int32)],
            >>>  'j': [array([0, 1], dtype=int32), array([0, 1], dtype=int32), array([0, 0, 0, 0, 1, 0], dtype=int32)]}

        :return: a tuple of three data dictionaries for training, validation and testing
        """

        # merge all classes form train, val and test data dicts
        all_classes = list(set(
            self.train_dict.keys() + self.val_dict.keys() + self.test_dict.keys()))

        train_labels = dict()
        val_labels = dict()
        test_labels = dict()

        # for each of the hierarchical levels in depth, check how many different protein classes
        # there are for the given example, this would result in:
        # [['2', '3'], ['2.1', '3.1'], ['2.1.2', '2.1.3', '2.1.4', '2.1.8', '3.1.1', '3.1.2']]
        unique_labels_at_depth = [
            sorted(list(set(['.'.join(x.split('.')[:h + 1]) for x in all_classes])))
            for h in range(self.h_depth)]

        # go over all sets (train, test, val) and for each go over all items (i.e. class and list
        # of protein codes) and for each item go over all protein codes and create a label for each
        # class-level
        for data_dict, label_dict in zip(
                [self.train_dict, self.val_dict, self.test_dict],
                [train_labels, val_labels, test_labels]):
            for cls, enzymes in data_dict.items():
                for enz in enzymes:
                    if enz not in label_dict.keys():
                        label_dict[enz] = [
                            np.zeros(len(unique_labels_at_depth[h]),
                                     dtype=np.int32) for h in
                            range(self.h_depth)]
                    for h in range(self.h_depth):
                        class_index = unique_labels_at_depth[h].index(
                            '.'.join(cls.split('.')[:h + 1]))
                        label_dict[enz][h][class_index] = 1

        return train_labels, val_labels, test_labels


if __name__ == "__main__":
    tr = {'3.1.1': ['a'], '2.1.2': ['b', 'a', 'c'],
          '2.1.3': ['d', 'e'], '2.1.4': ['f', 'g']}
    va = {'3.1.2': ['h'], '2.1.2': ['i']}
    te = {'3.1.1': ['j'], '2.1.8': ['k']}
    lf = LabelFactory(tr, va, te, 3)
    tr, va, te = lf.generate_hierarchical_labels()
    print(tr)
    print(va)
    print(te)
