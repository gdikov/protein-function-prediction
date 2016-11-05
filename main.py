import theano
import lasagne
import time

import draft.MoleculeMapLayer as mml
import draft.MoleculeMapOld as old
from visualizer.molview import MoleculeView

if __name__ == "__main__":
    start = time.time()

    inputs = theano.tensor.tensor4()

    network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=inputs)
    network = mml.MoleculeMapLayer(network)
    # network = old.MoleculeMapLayer(incoming=network, active_or_inactive=1)
    grids = network.get_output_for(molecule_ids=range(0, 1)).eval()
    # grids = network.get_output_for(molecule_numbers01=[[0],[0]]).eval()

    end = time.time()
    print(end - start)
    viewer = MoleculeView(data={"potential": grids[0, 0], "density": grids[0, 1]}, info={"name": "test"})
    viewer.density2d()
