import os
os.environ["THEANO_FLAGS"] = "device=gpu1,lib.cnmem=1"
import lasagne
import theano
import numpy as np

from protfun.layers import MoleculeMapLayer
from protfun.models.protein_predictor import ProteinPredictor
from protfun.preprocess.data_prep import DataSetup
from protfun.visualizer.molview import MoleculeView

grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/computed_grid")


def visualize():
    # import time
    for i in range(2, 5):
        print(i)
        dummy = lasagne.layers.InputLayer(shape=(None,))
        preprocess = MoleculeMapLayer(incoming=dummy, minibatch_size=1)
        # start = time.time()
        molecule_ids = theano.shared(np.array([i], dtype=np.int32))
        grids = preprocess.get_output_for(molecule_ids=molecule_ids).eval()
        np.save(grid_file+str(i), grids)
        # print(time.time() - start)
        # viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
        # viewer.density3d()
        # viewer.potential3d()


def train_enzymes():

    data = DataSetup(enzyme_classes=['3.4.21', '3.4.24'],
                     label_type='enzyme_classes',
                     force_download=False,
                     force_process=False)

    train_test_data = data.load_dataset()

    predictor = ProteinPredictor(data=train_test_data,
                                 minibatch_size=8,
                                 initial_per_class_datasize=100)

    predictor.train(epoch_count=100)
    predictor.test()
    predictor.summarize()


if __name__ == "__main__":
    train_enzymes()
    # visualize()
