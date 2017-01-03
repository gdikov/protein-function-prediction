import os

os.environ["THEANO_FLAGS"] = "device=gpu6,lib.cnmem=1"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import lasagne
import theano
import numpy as np

from protfun.layers import MoleculeMapLayer
from protfun.preprocess.data_prep import DataSetup
from protfun.visualizer.molview import MoleculeView
from protfun.preprocess.grid_processor import GridProcessor
from protfun.models import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import ModelTrainer
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import basic_convnet

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

        grids = preprocess.get_output_for(mols_info=mol_info).eval()
        # np.save(grid_file+str(i), grids)
        viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
        viewer.density3d()
        viewer.potential3d()


def train_enz_from_memmaps():
    path_to_moldata = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/moldata")
    data_feeder = EnzymesMolDataFeeder(path_to_moldata=path_to_moldata,
                                       enzyme_classes=['3.4.21', '3.4.24'],
                                       minibatch_size=8,
                                       init_samples_per_class=1)
    model = MemmapsDisjointClassifier(n_classes=2, network=basic_convnet, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=100)


def train_enz_from_grids():
    grids_dir = os.path.join(os.path.dirname(__file__), "../data_old/grids")
    data_feeder = EnzymesGridFeeder(grids_dir=grids_dir,
                                    grid_size=64,
                                    enzyme_classes=['3.4.21', '3.4.24'],
                                    minibatch_size=32,
                                    init_samples_per_class=2000)
    model = GridsDisjointClassifier(n_classes=2, network=basic_convnet, grid_size=64, minibatch_size=32)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=1000)


# def test_enz_from_grids():
#     grids_dir = os.path.join(os.path.dirname(__file__), "../data_old/grids")
#     data_feeder = EnzymesGridFeeder(grids_dir=grids_dir,
#                                     grid_size=64,
#                                     enzyme_classes=['3.4.21', '3.4.24'],
#                                     minibatch_size=8,
#                                     init_samples_per_class=1)
#
#     model = GridsDisjointClassifier(n_classes=2, network=basic_convnet, grid_size=64, minibatch_size=8)
#     trainer = ModelTrainer(model=model, data_feeder=data_feeder)
#     trainer.monitor.load_model("params_90.npz", network=model.get_output_layers())
#     trainer.test()


def preprocess_grids():
    data = DataSetup(enzyme_classes=['3.4.21', '3.4.24'],
                     label_type='enzyme_classes',
                     foldername="data_old",
                     force_download=False,
                     force_process=False)
    data = data.load_dataset()
    gridder = GridProcessor(force_process=False)
    for i in data['x_train']:
        gridder.process(i)
    for i in data['x_test']:
        gridder.process(i)
    for i in data['x_val']:
        gridder.process(i)


if __name__ == "__main__":
    # train_enz_from_memmaps()
    # preprocess_grids()
    train_enz_from_grids()
    # visualize()
