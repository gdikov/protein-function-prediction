import os

os.environ["THEANO_FLAGS"] = "device=gpu2,lib.cnmem=0"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import lasagne
import theano
import theano.tensor as T
import numpy as np

from protfun.layers import MoleculeMapLayer
from protfun.visualizer.molview import MoleculeView
from protfun.data_management.data_feed import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import ModelTrainer
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import basic_convnet

grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/computed_grid")


def visualize():
    floatX = theano.config.floatX
    coords_var = T.tensor3('coords')
    charges_var = T.matrix('charges')
    vdwradii_var = T.matrix('vdwradii')
    n_atoms_var = T.ivector('n_atoms')
    coords_input = lasagne.layers.InputLayer(shape=(1, None, None),
                                             input_var=coords_var)
    charges_input = lasagne.layers.InputLayer(shape=(1, None),
                                              input_var=charges_var)
    vdwradii_input = lasagne.layers.InputLayer(shape=(1, None),
                                               input_var=vdwradii_var)
    natoms_input = lasagne.layers.InputLayer(shape=(1,),
                                             input_var=n_atoms_var)
    preprocess = MoleculeMapLayer(incomings=[coords_input, charges_input, vdwradii_input, natoms_input],
                                  minibatch_size=1,
                                  rotate=False)

    path_to_moldata = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/processed")
    for i in ['3V7T']:
        coords = np.memmap(os.path.join(path_to_moldata, i, 'coords.memmap'),
                           mode='r', dtype=floatX).reshape(1, -1, 3)
        charges = np.memmap(os.path.join(path_to_moldata, i, 'charges.memmap'),
                            mode='r', dtype=floatX).reshape(1, -1)
        vdwradii = np.memmap(os.path.join(path_to_moldata, i, 'vdwradii.memmap'),
                             mode='r', dtype=floatX).reshape(1, -1)
        n_atoms = np.array(vdwradii.shape[1], dtype=np.int32).reshape(1,)

        mol_info = [theano.shared(coords), theano.shared(charges),
                    theano.shared(vdwradii), theano.shared(n_atoms)]

        grids = preprocess.get_output_for(mols_info=mol_info).eval()
        np.save(grid_file + i, grids)
        print("Saving grid for " + i)
        # viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": i})
        # viewer.density3d()
        # viewer.potential3d()


def train_enz_from_memmaps():
    data_feeder = EnzymesMolDataFeeder(minibatch_size=8,
                                       init_samples_per_class=1)
    model = MemmapsDisjointClassifier(n_classes=2, network=basic_convnet, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=100)


def train_enz_from_grids():
    data_feeder = EnzymesGridFeeder(minibatch_size=1,
                                    init_samples_per_class=1,
                                    prediction_depth=3,
                                    enzyme_classes=['3.4.21', '3.4.24'])
    model = GridsDisjointClassifier(n_classes=2, network=basic_convnet, grid_size=64, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder, val_frequency=1)
    trainer.train(epochs=1)


def test_enz_from_grids():
    data_feeder = EnzymesGridFeeder(minibatch_size=8,
                                    init_samples_per_class=2000)
    model = GridsDisjointClassifier(n_classes=2, network=basic_convnet, grid_size=64, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.monitor.load_model(model_name="params_54ep_meanvalacc[ 0.90322578  0.88306451].npz",
                               network=model.get_output_layers())
    trainer.test()


if __name__ == "__main__":
    # train_enz_from_memmaps()
    train_enz_from_grids()
    # visualize()
