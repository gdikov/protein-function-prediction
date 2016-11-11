import time
from os import path

import lasagne
import numpy as np
import theano

import layers.molmap_layer as mml
from protfun.visualizer.molview import MoleculeView

grid_file = path.join(path.dirname(path.realpath(__file__)), "../data/computed_grid.npy")


def preprocess():
    batch_size = 1
    inputs = theano.tensor.tensor4()
    network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=inputs)

    network = mml.MoleculeMapLayer(network, minibatch_size=batch_size)

    start = time.time()
    grids = network.get_output_for(molecule_ids=range(0, batch_size)).eval()

    end = time.time()
    print(end - start)
    np.save(grid_file, grids)


# def get_original():
#     orig = original._3D_generate(sdf_file)
#     grids = orig.get_maps()
#     np.save(grid_file, grids)


def visualize():
    grids = np.load(grid_file)

    viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
    viewer.density3d()
    viewer.potential3d()


def train():
    from protein_predictor import ProteinPredictor
    predictor = ProteinPredictor(minibatch_size=1)
    predictor.train()
    predictor.test()


if __name__ == "__main__":
    # preprocess()
    # get_original()
    # visualize()
    # grids = np.load(grid_file)
    # print("max potential %s" % grids[0, 0].max())
    # print("min potential %s" % grids[0, 0].min())
    # print("potential at corner %s" % grids[0, 0, 0, 0, 0])
    train()
