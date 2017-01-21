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


class JointClassModel(object):
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
        train_loss = T.sum(lasagne.objectives.categorical_crossentropy(train_predictions, targets))
        train_accuracy = T.mean(lasagne.objectives.categorical_accuracy(train_predictions, targets))

        per_class_train_accuracies = T.mean(T.eq(T.argmax(train_predictions, axis=1), T.argmax(targets, axis=1)),
                                            axis=0, dtype=theano.config.floatX)

        val_predictions = lasagne.layers.get_output(output_layer, deterministic=True)
        val_loss = T.sum(lasagne.objectives.categorical_crossentropy(val_predictions, targets))
        val_accuracy = T.mean(lasagne.objectives.categorical_accuracy(val_predictions, targets))
        per_class_val_accuracies = T.mean(T.eq(T.argmax(val_predictions, axis=1), T.argmax(targets, axis=1)),
                                          axis=0, dtype=theano.config.floatX)

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
                                                          lasagne.layers.get_all_layers(output_layer)))
        log.info("Computational graph compiled")

    def get_output_layers(self):
        return self.output_layers

    def get_name(self):
        return self.name


class GridsJointClassifier(JointClassModel):
    def __init__(self, name, n_classes, network, grid_size, minibatch_size, learning_rate=1e-4):
        super(GridsJointClassifier, self).__init__(name, n_classes, learning_rate)

        self.minibatch_size = minibatch_size
        grids = T.TensorType(floatX, (False,) * 5)()
        input_layer = lasagne.layers.InputLayer(shape=(self.minibatch_size, 1, grid_size, grid_size, grid_size),
                                                input_var=grids)
        rotated_grids = GridRotationLayer(incoming=input_layer, grid_side=grid_size)

        # apply the network to the preprocessed input
        self.output_layers = network(rotated_grids, n_outputs=n_classes,
                                     last_nonlinearity=lasagne.nonlinearities.softmax)
        self.define_forward_pass(input_vars=[grids], output_layer=self.output_layers)
