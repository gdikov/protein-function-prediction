import theano
import lasagne

import draft.MoleculeMapLayer as mml
from visualizer.molview import MoleculeView

if __name__ == "__main__":
    inputs = theano.tensor.tensor4()

    network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=inputs)
    network = mml.MoleculeMapLayer(network)
    grids = network.get_output_for(molecule_ids=range(10,11)).eval()

    viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
    viewer.density2d()
