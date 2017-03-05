import re
import numpy as np
import contextlib


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """
    temporarily sets the print options, mostly for the sake of pretty-printing
    :param args:
    :param kwargs:
    :return:
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def pp_array(arr):
    """
    pretty-prints a numpy array of float number. Truncated the precision to 3 decimal places.

    :param arr: the array to be pretty-printed
    :return:
    """
    with printoptions(precision=3, suppress=True):
        pretyfied_arr = str(re.sub(r'\ +', '|', str(arr)))
    return pretyfied_arr
