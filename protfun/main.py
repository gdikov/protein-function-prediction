import os
# os.environ["THEANO_FLAGS"] = "device=gpu7,lib.cnmem=1"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import lasagne
import theano
import numpy as np

from protfun.layers import MoleculeMapLayer
from protfun.models.protein_predictor import ProteinPredictor
from protfun.preprocess.data_prep import DataSetup
from protfun.visualizer.molview import MoleculeView

grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/computed_grid")


def visualize():
    for i in range(80, 100):
        dummy = lasagne.layers.InputLayer(shape=(None,))
        preprocess = MoleculeMapLayer(incoming=dummy, minibatch_size=1)
        molecule_ids = theano.shared(np.array([i], dtype=np.int32))
        grids = preprocess.get_output_for(molecule_ids=molecule_ids).eval()
        # np.save(grid_file+str(i), grids)
        viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
        viewer.density3d()
        viewer.potential3d()


def train_enzymes():

    data = DataSetup(enzyme_classes=['3.4.21', '3.4.24'],
                     label_type='enzyme_classes',
                     force_download=False,
                     force_process=False)

    train_test_data = data.load_dataset()

    predictor = ProteinPredictor(data=train_test_data,
                                 minibatch_size=8,
                                 initial_per_class_datasize=1)

    predictor.train(epoch_count=100)

if __name__ == "__main__":
    train_enzymes()
    # visualize()
