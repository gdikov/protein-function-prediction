import draft.MoleculeMapLayer as mml
import theano
import lasagne

if __name__ == "__main__":
    inputs = theano.tensor.tensor4()

    network = lasagne.layers.InputLayer(shape=(None, 1, None, None), input_var=inputs)
    network = mml.MoleculeMapLayer(network)
    network.get_output_for(molecule_ids=range(0, 10))
