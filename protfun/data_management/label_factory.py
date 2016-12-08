import numpy as np


class _LabelFactory():
    def __init__(self, train_dict, val_dict, test_dict, hierarchical_depth=1):
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        self.h_depth = hierarchical_depth

    def generate_hierarchical_labels(self):
        all_classes = list(set(self.train_dict.keys() + self.val_dict.keys() + self.test_dict.keys()))

        train_labels = dict()
        val_labels = dict()
        test_labels = dict()

        for h in range(self.h_depth):
            unique_labels_at_depth_h = list(set(['.'.join(x.split('.')[:h + 1]) for x in all_classes]))
            onehot_labels_at_depth_h = np.eye(len(unique_labels_at_depth_h))
            label_dict_at_depth_h = {k: v for k, v in
                                     zip(unique_labels_at_depth_h, onehot_labels_at_depth_h)}
            for data_dict, label_dict in zip([self.train_dict, self.val_dict, self.test_dict],
                                             [train_labels, val_labels, test_labels]):
                for cls in data_dict.keys():
                    for enz in data_dict[cls]:
                        if enz not in label_dict.keys():
                            label_dict[enz] = [[] for _ in range(self.h_depth)]
                        label_dict[enz][h].append(label_dict_at_depth_h['.'.join(cls.split('.')[:h + 1])])

if __name__ == "__main__":
    tr = {'3.1.1': ['ab', 'ac'], '2.1.2': ['de', 'ef']}
    va = {'2.1.2': ['ab', 'ac'], '3.1.1': ['ab', 'ac']}
    te = {'3.1.1': ['ab', 'ac'], '3.1.2': ['ab', 'ac']}
    lf = _LabelFactory(tr, va, te, 3)
    lf.generate_hierarchical_labels()

