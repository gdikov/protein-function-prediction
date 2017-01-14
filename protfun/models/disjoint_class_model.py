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
        self.output_layers = None

    def define_forward_pass(self, input_vars, output_layers):
        train_params = lasagne.layers.get_all_params(output_layers, trainable=True)

        # define the losses
        targets_ints = [T.ivector('targets' + str(i)) for i in range(0, self.n_classes)]
        targets = [T.eq(targets_ints[i].reshape((-1, 1)), T.arange(2)) for i in range(0, self.n_classes)]

        def categorical_crossentropy_logdomain(log_predictions, targets):
            return -T.sum(targets * log_predictions, axis=1)

        # define objective and training parameters
        train_predictions = [lasagne.layers.get_output(output) for output in output_layers]
        train_loss = T.sum(T.stack([categorical_crossentropy_logdomain(log_predictions=train_predictions[i],
                                                                       targets=targets[i]).mean()
                                    for i in range(0, self.n_classes)]))
        train_compared = T.stack(
            [T.eq(T.argmax(train_predictions[i], axis=-1), targets_ints[i]) for i in range(0, self.n_classes)],
            axis=1)
        train_accuracies = T.mean(train_compared, axis=0, dtype=theano.config.floatX)
        train_accuracy = T.mean(T.all(train_compared, axis=-1), dtype=theano.config.floatX)

        val_predictions = [lasagne.layers.get_output(output_layers[i], deterministic=True)
                           for i in range(0, self.n_classes)]
        val_loss = T.sum(T.stack([categorical_crossentropy_logdomain(log_predictions=val_predictions[i],
                                                                     targets=targets[i]).mean()
                                  for i in range(0, self.n_classes)]))
        val_compared = T.stack(
            [T.eq(T.argmax(val_predictions[i], axis=-1), targets_ints[i]) for i in range(0, self.n_classes)],
            axis=1)
        val_accuracies = T.mean(val_compared, axis=0, dtype=theano.config.floatX)
        val_accuracy = T.mean(T.all(val_compared, axis=-1), dtype=theano.config.floatX)

        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss,
                                                    params=train_params,
                                                    learning_rate=self.learning_rate)

        self.train_function = theano.function(inputs=input_vars + targets_ints,
                                              outputs={'loss': train_loss, 'accuracy': train_accuracy,
                                                       'per_class_accs': train_accuracies,
                                                       'predictions': T.stack(train_predictions)},
                                              updates=train_params_updates)  # , profile=True)

        self.validation_function = theano.function(inputs=input_vars + targets_ints,
                                                   outputs={'loss': val_loss, 'accuracy': val_accuracy,
                                                            'per_class_accs': val_accuracies,
                                                            'predictions': T.stack(val_predictions)})
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
        self.output_layers = network(grids, n_outputs=n_classes)
        self.define_forward_pass(input_vars=[coords, charges, vdwradii, n_atoms], output_layers=self.output_layers)


class GridsDisjointClassifier(DisjointClassModel):
    def __init__(self, name, n_classes, network, grid_size, minibatch_size, learning_rate=1e-4):
        super(GridsDisjointClassifier, self).__init__(name, n_classes, learning_rate)

        self.minibatch_size = minibatch_size
        grids = T.TensorType(floatX, (False,) * 5)()
        input_layer = lasagne.layers.InputLayer(shape=(self.minibatch_size, 2, grid_size, grid_size, grid_size),
                                                input_var=grids)
        rotated_grids = GridRotationLayer(incoming=input_layer, grid_side=grid_size)

        # apply the network to the preprocessed input
        self.output_layers = network(rotated_grids, n_outputs=n_classes)
        self.define_forward_pass(input_vars=[grids], output_layers=self.output_layers)
