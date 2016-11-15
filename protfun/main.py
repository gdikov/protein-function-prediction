from os import path
import numpy as np

from protfun.visualizer.molview import MoleculeView
from protfun.data_prep import DataSetup
from protfun.protein_predictor import ProteinPredictor

grid_file = path.join(path.dirname(path.realpath(__file__)), "../data/computed_grid.npy")


def visualize():
    grids = np.load(grid_file)

    viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
    viewer.density3d()
    viewer.potential3d()


if __name__ == "__main__":
    import os

    path_to_enz = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/enzymes/3_4_21.labels")
    with open(path_to_enz, 'r') as f:
        enzymes = [e.strip() for e in f.readlines()]

    data = DataSetup(prot_codes=enzymes[:100],
                     download_again=False,
                     process_again=False)

    # data dict with keys:
    # 'x_id2name', 'y_id2name'  :   the encoding of names and ids for molecules and labels
    # 'x_train', 'y_train'      :   all training samples
    # 'x_val', 'y_val'          :   all validation samples used during training
    # 'x_test', 'y_test'        :   all test samples for performance evaluation
    train_test_data = data.load_dataset()

    predictor = ProteinPredictor(data=train_test_data,
                                 minibatch_size=1)

    predictor.train(epoch_count=10)
    predictor.test()
    predictor.summarize()
