import numpy as np
import theano
import theano.tensor as T
import lasagne
import colorlog as log
import logging

from protfun.layers.molmap_layer import MoleculeMapLayer
from protfun.layers.grid_rotate_layer import GridRotationLayer

log.basicConfig(level=logging.DEBUG)
floatX = theano.config.floatX
intX = np.int32


class DisjointClassModel(object):
    def __init__(self, name, n_classes, learning_rate):
        self.name = name
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.train_function = None
        self.validation_function = None
        self.get_hidden_activations = None
        self.output_layers = None

    def define_forward_pass(self, input_vars, output_layer):
        train_params = lasagne.layers.get_all_params(output_layer, trainable=True)
        targets = T.imatrix('targets')

        # define objective and training parameters
        train_predictions = lasagne.layers.get_output(output_layer)
        train_loss = T.sum(lasagne.objectives.binary_crossentropy(train_predictions, targets))
        train_accuracy = T.mean(T.all(T.eq(train_predictions > 0.5, targets), axis=-1), axis=0,
                                dtype=theano.config.floatX)
        per_class_train_accuracies = T.mean(T.eq(train_predictions > 0.5, targets), axis=0, dtype=theano.config.floatX)

        val_predictions = lasagne.layers.get_output(output_layer, deterministic=True)
        val_loss = T.sum(lasagne.objectives.binary_crossentropy(val_predictions, targets))
        val_accuracy = T.mean(T.all(T.eq(val_predictions > 0.5, targets), axis=-1), axis=0, dtype=theano.config.floatX)
        per_class_val_accuracies = T.mean(T.eq(val_predictions > 0.5, targets), axis=0, dtype=theano.config.floatX)

        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss,
                                                    params=train_params,
                                                    learning_rate=self.learning_rate)

        self.train_function = theano.function(inputs=input_vars + [targets],
                                              outputs={'loss': train_loss, 'accuracy': train_accuracy,
                                                       'per_class_accs': per_class_train_accuracies,
                                                       'predictions': T.stack(train_predictions)},
                                              updates=train_params_updates)  # , profile=True)

        self.validation_function = theano.function(inputs=input_vars + [targets],
                                                   outputs={'loss': val_loss, 'accuracy': val_accuracy,
                                                            'per_class_accs': per_class_val_accuracies,
                                                            'predictions': T.stack(val_predictions)})

        self.get_hidden_activations = theano.function(inputs=input_vars,
                                                      outputs=lasagne.layers.get_output(
                                                          lasagne.layers.get_all_layers(output_layer))
                                                              + [T.stack(val_predictions)])
        log.info("Computational graph compiled")

    def get_output_layers(self):
        return self.output_layers

    def get_name(self):
        return self.name


class MemmapsDisjointClassifier(DisjointClassModel):
    def __init__(self, name, n_classes, network, minibatch_size, learning_rate=1e-4):
        super(MemmapsDisjointClassifier, self).__init__(name, n_classes, learning_rate)
        self.minibatch_size = minibatch_size

        coords = T.tensor3('coords')
        charges = T.matrix('charges')
        vdwradii = T.matrix('vdwradii')
        n_atoms = T.ivector('n_atoms')
        coords_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None, None),
                                                 input_var=coords)
        charges_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None),
                                                  input_var=charges)
        vdwradii_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None),
                                                   input_var=vdwradii)
        natoms_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,),
                                                 input_var=n_atoms)
        grids = MoleculeMapLayer(incomings=[coords_input, charges_input, vdwradii_input, natoms_input],
                                 minibatch_size=self.minibatch_size,
                                 use_esp=False)

        # apply the network to the preprocessed input
        self.output_layers = network(grids, n_outputs=n_classes, last_nonlinearity=lasagne.nonlinearities.sigmoid)
        self.define_forward_pass(input_vars=[coords, charges, vdwradii, n_atoms], output_layer=self.output_layers)


class GridsDisjointClassifier(DisjointClassModel):
    def __init__(self, name, n_classes, network, grid_size, n_channels, minibatch_size, learning_rate=1e-4):
        super(GridsDisjointClassifier, self).__init__(name, n_classes, learning_rate)

        self.minibatch_size = minibatch_size
        grids = T.TensorType(floatX, (False,) * 5)()
        input_layer = lasagne.layers.InputLayer(
            shape=(self.minibatch_size, n_channels, grid_size, grid_size, grid_size),
            input_var=grids)
        rotated_grids = GridRotationLayer(incoming=input_layer, grid_side=grid_size, n_channels=n_channels)

        # apply the network to the preprocessed input
        self.output_layers = network(rotated_grids, n_outputs=n_classes,
                                     last_nonlinearity=lasagne.nonlinearities.sigmoid)
        self.define_forward_pass(input_vars=[grids], output_layer=self.output_layers)
