import cPickle
import colorlog as log
import os


def save_pickle(file_path, data):
    if isinstance(data, list) and isinstance(file_path, list):
        if len(data) == len(file_path):
            for path, dat in zip(file_path, data):
                with open(path, 'wb') as f:
                    cPickle.dump(dat, f)
        else:
            log.error("File paths are not matching the number of objects to save")
            raise ValueError
    else:
        with open(file_path, 'wb') as f:
            cPickle.dump(data, f)


def load_pickle(file_path):
    def _load_one(path):
        if not os.path.exists(path):
            log.error("No data was saved in {0}.".format(path))
            raise IOError
        else:
            with open(path, 'r') as f:
                data = cPickle.load(f)
            return data

    if isinstance(file_path, list):
        objs = []
        for path in file_path:
            unpickled = _load_one(path)
            objs.append(unpickled)
        return objs
    else:
        return _load_one(file_path)

def construct_hierarchical_tree(data_dict, prediction_depth=4):

    def merge_prots(subpath, is_leaf):
        merged = []
        if not is_leaf:
            for key, vals in data_dict.items():
                if key.startswith(subpath + '.'):
                    merged += vals
        else:
            for key, vals in data_dict.items():
                if key == subpath:
                    merged += vals
        return merged

    keys_at_max_hdepth = set(['.'.join(x.split('.')[:prediction_depth]) for x in data_dict.keys()])
    tree_at_max_hdepth = {key: merge_prots(key, is_leaf=False)
                          if prediction_depth < 4 else merge_prots(key, is_leaf=True)
                          for key in keys_at_max_hdepth}
    return tree_at_max_hdepth