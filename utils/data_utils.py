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
