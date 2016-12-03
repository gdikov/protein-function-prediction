import os
os.environ["THEANO_FLAGS"] = "device=gpu1,lib.cnmem=1"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import lasagne
import theano
import numpy as np

from protfun.layers import MoleculeMapLayer
from protfun.models.protein_predictor import ProteinPredictor
from protfun.preprocess.data_prep import DataSetup
from protfun.visualizer.molview import MoleculeView
from protfun.preprocess.grid_processor import GridProcessor

grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/computed_grid")


def visualize():
    path_to_moldata = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/moldata")
    max_atoms = np.memmap(os.path.join(path_to_moldata, 'max_atoms.memmap'), mode='r', dtype=np.int32)[0]
    coords = np.memmap(os.path.join(path_to_moldata, 'coords.memmap'), mode='r', dtype=np.float32).reshape(
        (-1, max_atoms, 3))
    charges = np.memmap(os.path.join(path_to_moldata, 'charges.memmap'), mode='r', dtype=np.float32).reshape(
        (-1, max_atoms))
    vdwradii = np.memmap(os.path.join(path_to_moldata, 'vdwradii.memmap'), mode='r', dtype=np.float32).reshape(
        (-1, max_atoms))
    n_atoms = np.memmap(os.path.join(path_to_moldata, 'n_atoms.memmap'), mode='r', dtype=np.int32)
    for i in range(3, 100):
        dummy = lasagne.layers.InputLayer(shape=(None,))
        preprocess = MoleculeMapLayer(incomings=[dummy, dummy], minibatch_size=1)
        mol_info = [theano.shared(np.array(coords[[i]], dtype=np.float32)),
                   theano.shared(np.array(charges[[i]], dtype=np.float32)),
                   theano.shared(np.array(vdwradii[[i]], dtype=np.float32)),
                   theano.shared(np.array(n_atoms[[i]], dtype=np.int32))]

        grids = preprocess.get_output_for(mol_info=mol_info).eval()
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
    predictor.train(epoch_count=10000)


def preprocess_grids():
    data = DataSetup(enzyme_classes=['3.4.21', '3.4.24'],
                     label_type='enzyme_classes',
                     force_download=False,
                     force_process=False)
    data = data.load_dataset()
    gridder = GridProcessor()
    for i in data['x_train']:
        gridder.process(i)

    gridder = GridProcessor(folder_name="test_grids")
    for i in data['x_test']:
        gridder.process(i)


if __name__ == "__main__":
    # train_enzymes()
    # visualize()
    preprocess_grids()
