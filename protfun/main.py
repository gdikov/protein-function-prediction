import time
from os import path

import lasagne
import numpy as np
import theano

import layers.molmap_layer as mml
from protfun.visualizer.molview import MoleculeView
from protfun.data_prep import DataSetup
from protfun.protein_predictor import ProteinPredictor

grid_file = path.join(path.dirname(path.realpath(__file__)), "../data/computed_grid.npy")


def preprocess(index=0):
    batch_size = 1
    inputs = theano.tensor.tensor4()
    network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=inputs)

    network = mml.MoleculeMapLayer(network, minibatch_size=batch_size)

    start = time.time()
    grids = network.get_output_for(molecule_ids=range(index, index + batch_size)).eval()

    end = time.time()
    print(end - start)
    np.save(grid_file, grids)


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
    # preprocess(index=0)
    # grids = np.load(grid_file)
    # print("max potential %s" % grids[0, 0].max())
    # print("min potential %s" % grids[0, 0].min())
    # print("potential at corner %s" % grids[0, 0, 0, 0, 0])
    # visualize()
    # train()

    data = DataSetup(prot_codes=["1UBI"])

    train_test_data, num_gene_ontologies = data.load_dataset()
    #
    # predictor = ProteinPredictor(train_data=train_mol_ids,
    #                              test_data=test_mol_ids,
    #                              minibatch_size=1,
    #                              num_output_classes=num_gene_ontologies)
    #
    # predictor.train(epoch_count=1)
    # predictor.test()
    # predictor.summarize()




