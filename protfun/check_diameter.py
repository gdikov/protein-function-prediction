import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.mlab as mlab
import seaborn as sns

sns.set_style("whitegrid")
import matplotlib.pyplot as plt

from utils import load_pickle
data_dir = "/usr/prakt/w073/DLCV_ProtFun/data"

prots = os.path.join(data_dir, "processed/valid_prot_codes.pickle")
prot_codes = load_pickle(prots)

radiuses = []
for cls, enzymes in prot_codes.items():
    for enzyme in enzymes:
        coords_file = os.path.join(data_dir, "processed/{}/coords.memmap".format(enzyme.upper()))
        coords = np.memmap(coords_file, mode='r', dtype=np.float32).reshape((-1, 3))
        norms = np.sqrt(np.sum(coords ** 2, axis=1))
        max_length = np.max(norms)
        radiuses.append(max_length)

n, bins, patches = plt.hist(2*radiuses, 50, normed=1, alpha=0.75)

plt.xlabel('diameter')
plt.ylabel('Probability')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.savefig(os.path.join(os.path.dirname(__file__), 'diameters_3.4.21_3.4.24.png'))
