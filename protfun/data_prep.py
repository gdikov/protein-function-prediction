import os
import numpy as np


def load_dataset(filename):
    # TODO: create labeled data from PDB parsed samples

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/" + filename)

    data = np.load(data_path)
    labels = np.array([[1]], dtype=np.float32)
    # data = _split_dataset(data)
    data_dict = {'x_train': data, 'y_train': labels,
                 'x_val': data, 'y_val': labels,
                 'x_test': data, 'y_test': labels}

    print("INFO: Data loaded")

    return data_dict


def load_labels(filename):
    # TODO: load the labels for the proteins from a memmap
    print("INFO: labels loaded")
    y_train = np.array([1, 0], dtype=np.float32)
    y_val = y_train
    y_test = y_train

    return {'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test}


def _split_dataset(data, train_percentage):
    pass
