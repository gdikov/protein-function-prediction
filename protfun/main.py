import time
from os import path

import lasagne
import numpy as np
import theano

import layers.molmap_layer as mml

grid_file = path.join(path.dirname(path.realpath(__file__)), "../data/computed_grid.npy")


def preprocess():
    batch_size = 1
    inputs = theano.tensor.tensor4()
    network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=inputs)

    network = mml.MoleculeMapLayer(network, minibatch_size=batch_size)
    # network = old.MoleculeMapLayer(incoming=network, active_or_inactive=1, minibatch_size=batch_size)

    start = time.time()
    grids = network.get_output_for(molecule_ids=range(0, batch_size)).eval()
    # grids = network.get_output_for(molecule_numbers01=[range(0, batch_size), range(0, batch_size)]).eval()
    end = time.time()
    print(end - start)
    np.save(grid_file, grids)


def visualize():
    grids = np.load(grid_file)
    print(grids.shape)
    print(grids[0, 0].mean())
    # viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
    # viewer.density3d()
    # viewer.potential3d()


if __name__ == "__main__":
    # preprocess()
    visualize()
