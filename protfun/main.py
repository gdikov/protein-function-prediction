import os
# os.environ["THEANO_FLAGS"] = "device=gpu2,lib.cnmem=1"
import lasagne

from protfun.layers import MoleculeMapLayer
from protfun.models.protein_predictor import ProteinPredictor
from protfun.preprocess.data_prep import DataSetup
from protfun.visualizer.molview import MoleculeView

grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/computed_grid.npy")


def visualize():
    # import time
    for i in range(77, 78):
        dummy = lasagne.layers.InputLayer(shape=(None,))
        preprocess = MoleculeMapLayer(incoming=dummy, minibatch_size=1)
        # start = time.time()
        grids = preprocess.get_output_for(molecule_ids=[i]).eval()
        # print(time.time() - start)
        viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
        viewer.density3d()
        viewer.potential3d()


def train_enzymes():

    data = DataSetup(enzyme_classes=['3.4.21', '3.4.24'],
                     label_type='enzyme_classes',
                     max_prot_per_class=100,
                     force_download=False,
                     force_process=False)

    train_test_data = data.load_dataset()

    predictor = ProteinPredictor(data=train_test_data,
                                 minibatch_size=1)

    predictor.train(epoch_count=100)
    predictor.test()
    predictor.summarize()


if __name__ == "__main__":
    # train_enzymes()
    visualize()
