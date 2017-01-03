import numpy as np

params_file = np.load("params_90.npz")
param_values = [params_file['arr_%d' % i] for i in range(len(params_file.files))]
pass