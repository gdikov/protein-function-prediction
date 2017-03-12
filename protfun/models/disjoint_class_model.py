import abc
import numpy as np
import theano
import theano.tensor as T
import lasagne

from protfun.layers.molmap_layer import MoleculeMapLayer
from protfun.layers.grid_rotate_layer import GridRotationLayer
from protfun.utils.log import get_logger

log = get_logger("disjoint_class_model")

floatX = theano.config.floatX
intX = np.int32


class DisjointClassModel(object):
    """
    Abstract class, not meant to be instantiated.

    DisjointClassModel is a generic multi-class classifier model, that uses a separate sigmoid
    output for each of the classes, instead of a shared softmax. This allows for an instance to be
    classified as more than one class (e.g. if two sigmoids are active at once).
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, n_classes, learning_rate):
        """
        :param name: name of the model, used by external mechanisms for saving training history etc.
        :param n_classes: total number of different classes for the classification.
        :param learning_rate: initial learning rate
        """
        self.name = name
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        # these functions are defined in define_forward_pass()
        self.train_function = None
        self.validation_function = None
        self.get_hidden_activations = None

        # child classes must define self.output_layers before calling define_forward_pass()
        self.output_layers = None

    def define_forward_pass(self, input_vars, output_layer, penalty=0):
        """
        define_forward_pass is meant to be called in the constructors of models that inherit
        from DisjointClassModel. It defines the core functionality of the model:
            * defines the loss function (multi-class classification)
            * defines the optimization methods (e.g. adam)
            * defines the accuracy measure
            * defines the train_function (forward pass during training)
            * defines the validation_function (forward pass during testing / validation)

        :param input_vars: list of the Theano variables for the input data the model is applied on.
        :param output_layer: the last Lasagne layer of the neural network, child classes should apply the
            neural network before calling define_forward_pass()
        :param penalty: if applicable, and L2 penalty to add to the loss (otherwise 0).
        """
        train_params = lasagne.layers.get_all_params(output_layer, trainable=True)
        targets = T.imatrix('targets')

        # define objective and training parameters
        train_predictions = lasagne.layers.get_output(output_layer)
        train_loss = T.sum(
            lasagne.objectives.binary_crossentropy(train_predictions, targets)) + penalty
        train_accuracy = T.mean(T.all(T.eq(train_predictions > 0.5, targets), axis=-1), axis=0,
                                dtype=theano.config.floatX)
        per_class_train_accuracies = T.mean(T.eq(train_predictions > 0.5, targets), axis=0,
                                            dtype=theano.config.floatX)

        val_predictions = lasagne.layers.get_output(output_layer, deterministic=True)
        val_loss = T.sum(lasagne.objectives.binary_crossentropy(val_predictions, targets)) + penalty
        val_accuracy = T.mean(T.all(T.eq(val_predictions > 0.5, targets), axis=-1), axis=0,
                              dtype=theano.config.floatX)
        per_class_val_accuracies = T.mean(T.eq(val_predictions > 0.5, targets), axis=0,
                                          dtype=theano.config.floatX)

        train_params_updates = lasagne.updates.adam(loss_or_grads=train_loss,
                                                    params=train_params,
                                                    learning_rate=self.learning_rate)

        # define training forward pass function
        self.train_function = theano.function(inputs=input_vars + [targets],
                                              outputs={'loss': train_loss,
                                                       'accuracy': train_accuracy,
                                                       'per_class_accs': per_class_train_accuracies,
                                                       'predictions': T.stack(train_predictions)},
                                              updates=train_params_updates)  # , profile=True)

        # define testing forward pass function
        self.validation_function = theano.function(inputs=input_vars + [targets],
                                                   outputs={'loss': val_loss,
                                                            'accuracy': val_accuracy,
                                                            'per_class_accs': per_class_val_accuracies,
                                                            'predictions': T.stack(
                                                                val_predictions)})

        # define a utility function for getting the activations of all layers
        self.get_hidden_activations = theano.function(inputs=input_vars,
                                                      outputs=lasagne.layers.get_output(
                                                          lasagne.layers.get_all_layers(
                                                              output_layer))
                                                              + [T.stack(val_predictions)])
        log.info("Computational graph for {} compiled".format(self.name))

    def get_output_layers(self):
        """
        :return: last lasagne layers of the neural network for this model
        """
        return self.output_layers

    def get_name(self):
        return self.name


class MemmapsDisjointClassifier(DisjointClassModel):
    """
    MemmapsDisjointClassifier extends the DisjointClassModel.
    It uses the molmap_layer to compute the electron density grids of proteins on the fly,
    thus it is rather slow.

    The inputs of this model are directly the atom coordinates, vdwradii and n_atoms for the
    molecules in the minibatch.
    """

    def __init__(self, name, n_classes, network, minibatch_size, learning_rate=1e-4):
        """
        :param name: name of the model, used by external mechanisms for saving training history etc.
        :param n_classes: total number of different classes for the classification.
        :param network: neural network function which should be applied on the input variables
        :param minibatch_size: -
        :param learning_rate: initial learning rate
        """
        super(MemmapsDisjointClassifier, self).__init__(name, n_classes, learning_rate)
        self.minibatch_size = minibatch_size

        # define the input variables for this model
        coords = T.tensor3('coords')
        vdwradii = T.matrix('vdwradii')
        n_atoms = T.ivector('n_atoms')
        coords_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None, None),
                                                 input_var=coords)
        vdwradii_input = lasagne.layers.InputLayer(shape=(self.minibatch_size, None),
                                                   input_var=vdwradii)
        natoms_input = lasagne.layers.InputLayer(shape=(self.minibatch_size,),
                                                 input_var=n_atoms)
        grids = MoleculeMapLayer(
            incomings=[coords_input, vdwradii_input, natoms_input],
            minibatch_size=self.minibatch_size, rotate=True)

        # apply the network to the preprocessed input
        self.output_layers, self.penalty = network(grids, n_outputs=n_classes,
                                                   last_nonlinearity=lasagne.nonlinearities.sigmoid)
        self.define_forward_pass(input_vars=[coords, vdwradii, n_atoms],
                                 output_layer=self.output_layers,
                                 penalty=self.penalty)


class GridsDisjointClassifier(DisjointClassModel):
    """
    GridsDisjointClassifier extends the DisjointClassModel.
    It allows for a sample to be in multiple classes at once.
    It uses the GridRotationLayer to **only rotate** the already computed electron density grids,
    i.e. it does require the electron densities as input and does not compute them on the fly.
    Thus this model is very efficient.

    The inputs of this model are the already precomputed electron density grids.
    """

    def __init__(self, name, n_classes, network, grid_size, n_channels, minibatch_size,
                 learning_rate=1e-4):
        """
        :param name: name of the model, used by external mechanisms for saving training history etc.
        :param n_classes: total number of different classes for the classification.
        :param network: neural network function which should be applied on the input variables
        :param grid_size: number of points on each side of the incoming el. density grids
        :param n_channels: number of channels in the input grids (should be 1)
        :param minibatch_size: -
        :param learning_rate: initial learning rate
        """
        super(GridsDisjointClassifier, self).__init__(name, n_classes, learning_rate)
        self.minibatch_size = minibatch_size

        # define the model inputs
        grids = T.TensorType(floatX, (False,) * 5)()
        input_layer = lasagne.layers.InputLayer(
            shape=(self.minibatch_size, n_channels, grid_size, grid_size, grid_size),
            input_var=grids)
        rotated_grids = GridRotationLayer(incoming=input_layer, grid_side=grid_size,
                                          n_channels=n_channels)

        # apply the network to the preprocessed input
        self.output_layers, self.penalty = network(rotated_grids,
                                                   n_outputs=n_classes,
                                                   last_nonlinearity=lasagne.nonlinearities.sigmoid)
        self.define_forward_pass(input_vars=[grids],
                                 output_layer=self.output_layers,
                                 penalty=self.penalty)
