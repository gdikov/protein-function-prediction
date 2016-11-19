import os
import lasagne

from protfun.visualizer.molview import MoleculeView
from protfun.data_prep import DataSetup
from protfun.protein_predictor import ProteinPredictor
from protfun.layers import MoleculeMapLayer

grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/computed_grid.npy")


def visualize():
    for i in range(0, 182):
        dummy = lasagne.layers.InputLayer(shape=(None,))
        preprocess = MoleculeMapLayer(incoming=dummy, minibatch_size=1)
        grids = preprocess.get_output_for(molecule_ids=[i]).eval()
        viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
        viewer.density3d()
        viewer.potential3d()


def collect_proteins():
    path_to_enz = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/enzymes/3_4_21.labels")
    with open(path_to_enz, 'r') as f:
        enzymes = [e.strip() for e in f.readlines()[:500]]
    path_to_enz = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/enzymes/3_4_24.labels")
    with open(path_to_enz, 'r') as f:
        enzymes += [e.strip() for e in f.readlines()[:500]]
    return enzymes


def train_enzymes():
    enzymes = collect_proteins()

    data = DataSetup(prot_codes=enzymes,
                     download_again=True,
                     process_again=True)

    train_test_data = data.load_dataset()

    predictor = ProteinPredictor(data=train_test_data,
                                 minibatch_size=1)

    predictor.train(epoch_count=100)
    predictor.test()
    predictor.summarize()


if __name__ == "__main__":
    train_enzymes()
    # visualize()
