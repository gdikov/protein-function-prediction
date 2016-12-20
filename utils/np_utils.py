import re
import numpy as np
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def pp_array(arr):
    with printoptions(precision=3, suppress=True):
        pretyfied_arr = str(re.sub(r'\ +', '|', str(arr)))
    return pretyfied_arr
