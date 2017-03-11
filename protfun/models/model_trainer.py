import datetime
import random
import string
import os
import numpy as np

from protfun.utils import save_pickle
from protfun.config import save_config
from protfun.data_management.data_feed import EnzymesGridFeeder
from protfun.data_management.data_manager import EnzymeDataManager
from protfun.models import GridsDisjointClassifier
from protfun.models.model_monitor import ModelMonitor
from protfun.networks import get_network
from protfun.utils.np_utils import pp_array
from protfun.visualizer.netview import NetworkView
from protfun.visualizer.progressview import ProgressView
from protfun.utils.log import setup_logger

log = setup_logger("model_trainer")


class ModelTrainer(object):
    """
    ModelTrainer is responsible for the training & testing of a model. It supervises the training
    procedure, saves information about the training into files and can also validate & test
    a trained model in the end.

    It takes a data feeder as an argument in the constructor, and then fetches mini-batches from the
    data feeder during each training iteration and forwards them to the model under training.
    The model to be trained is provided as a parameter to the constructor of the model trainer.

    Usage::
        >>> model = GridsDisjointClassifier(...)
        >>> feeder = EnzymesGridFeeder(...)
        >>> trainer = ModelTrainer(model=model, data_feeder=feeder)
        >>> # train the model
        >>> trainer.train(epochs=1000)
        >>> # test the model performance
        >>> trainer.test()

    """

    def __init__(self, model, data_feeder, checkpoint_frequency=1, first_epoch=0):
        """
        On initialization, the ModelTrainer will check if there is previous training history for
        the current model (based on its unique name), and if so will load it and continue from
        there. Otherwise it will start from scratch.

        IMPORTANT: in the case of continued training, the user is required to load the model weights
        into the model himself. The ModelTrainer will only load the training history so far, but
        not the checkpointed model weights.

        :param model: the model to be trained (supervised)
        :param data_feeder: a data feeder that can provide the minibatches from the training, test
            or validation set.
        :param checkpoint_frequency: specifies the frequency (in terms of epochs) for model
            validation and checkpointing of trained weights.
        :param first_epoch: in case the training was interrupted, one can tell the trainer from
            which epoch it should start.
        """
        self.model = model
        self.data_feeder = data_feeder
        self.checkpoint_frequency = checkpoint_frequency
        self.monitor = ModelMonitor(outputs=model.get_output_layers(),
                                    data_dir=data_feeder.get_data_dir(),
                                    name=model.get_name())
        self.network_view = NetworkView(data_dir=self.monitor.get_model_dir())
        self.current_max_train_acc = np.array(0.85)
        self.current_max_val_acc = np.array(0.0)
        self.first_epoch = first_epoch
        # save training history data
        history = self.monitor.load_train_history(epoch=first_epoch)
        if history is not None:
            self.history = history
        else:
            self.history = {'train_loss': list(),
                            'train_accuracy': list(),
                            'train_per_class_accs': list(),
                            'train_predictions': list(),
                            'val_loss': list(),
                            'val_accuracy': list(),
                            'val_per_class_accs': list(),
                            'val_predictions': list(),
                            'val_targets': list(),
                            'time_epoch': list()}

    def train(self, epochs=100):
        """
        Train the model for that many epochs, on the training set.
        Under the model's directory, the ModelTrainer also saves:
            * the current train / val / test split information
            * a diagram of the neural network architecture of the model
            * continuously saves the training history (diagram and .pickle) for the trained model.
            * continuously saves the best model weights determined during training

        :param epochs: for how many epochs to train.
        """
        try:
            log.info("Training model {}".format(self.model.get_name()))
            save_pickle(
                file_path=[os.path.join(self.monitor.get_model_dir(), "train_prot_codes.pickle"),
                           os.path.join(self.monitor.get_model_dir(), "val_prot_codes.pickle"),
                           os.path.join(self.monitor.get_model_dir(), "test_prot_codes.pickle")],
                data=[self.data_feeder.get_train_data(),
                      self.data_feeder.get_val_data(),
                      self.data_feeder.get_test_data()])
            self.network_view.save_network_graph(self.model.get_output_layers(), "network.png")
            self._train(epochs)
            self.monitor.save_history_and_model(self.history, epoch_count=epochs)
        except (KeyboardInterrupt, SystemExit):
            log.warning("Training is interrupted")
            self.monitor.save_history_and_model(self.history)

    def _train(self, epochs=100):
        steps_before_validate = 0

        # iterate over epochs
        for e in xrange(self.first_epoch, self.first_epoch + epochs):
            epoch_losses = []
            epoch_accs = []

            # iterate over minibatches (via the data_feeder)
            for proteins, samples, targets in self.data_feeder.iterate_train_data():
                output = self.model.train_function(*(samples + targets))
                loss = output['loss']
                accuracy = output['accuracy']
                per_class_accs = output['per_class_accs']
                predictions = output['predictions']

                # this can be enabled to profile the forward pass
                # self.model.train_function.profile.print_summary()

                # save history information
                epoch_losses.append(loss)
                epoch_accs.append(accuracy)
                self.history['train_loss'].append(loss)
                self.history['train_accuracy'].append(accuracy)
                self.history['train_per_class_accs'].append(per_class_accs)
                self.history['train_predictions'].append(predictions)

                steps_before_validate += 1

            epoch_loss_means = np.mean(np.array(epoch_losses), axis=0)
            epoch_acc_means = np.mean(np.array(epoch_accs), axis=0)
            log.info("train: epoch {0} loss mean: {1} acc mean: {2}".format(e, epoch_loss_means,
                                                                            epoch_acc_means))

            # validate the model, save model weights if a new best accuracy was achieved
            if e % self.checkpoint_frequency == self.checkpoint_frequency - 1:
                self.validate(steps_before_validate, e)
                self.monitor.save_train_history(self.history, e, False, msg="best")
                steps_before_validate = 0
                progress = ProgressView(model_name=self.model.get_name(),
                                        data_dir=self.monitor.get_model_dir(),
                                        history_dict=self.history)
                progress.save()

    def validate(self, steps_before_validate, epoch):
        """
        Validate the perfomance of the model currently under training, on the validation set.
        Saves the model parameters in case a new best accuracy score was reached.
        Meant to be used in self.train(), but could be called separately if required.

        :param steps_before_validate: how many minibatches have passed since the last validation.
        :param epoch: current epoch
        """
        val_loss_means, val_acc_means, val_per_class_accs_means, val_predictions, val_targets, _ = self._test(
            mode='val')
        self.history['val_loss'] += [val_loss_means] * steps_before_validate
        self.history['val_accuracy'] += [val_acc_means] * steps_before_validate
        self.history['val_per_class_accs'] += [val_per_class_accs_means] * steps_before_validate
        self.history['val_predictions'].append(val_predictions)
        self.history['val_targets'].append(val_targets)

        # save parameters if an improvement in accuracy observed
        if np.alltrue(val_acc_means > self.current_max_val_acc):
            self.current_max_val_acc = val_acc_means
            self.monitor.save_model(epoch,
                                    "meanvalacc{0}".format(pp_array(self.current_max_val_acc)))

    def test(self):
        """
        Test the overall performance of the model on the test set.
        Returns detail information about the:
            * mean loss over the test set
            * mean accuracy over the test set
            * mean per class accuracy over the test set
            * predictions for the test set
            * targets (ground truths) corresponding to the predictions
            * protein codes of the proteins that were tested

        :return: loss_means, accuracy_means, per_class_accs_means,
                 predictions, targets, proteins
        """
        log.warning(
            "You are testing a model with the secret test set! " +
            "You are not allowed to change the model after seeing the results!!! ")
        return self._test(mode='test')

    def _test(self, mode='test'):
        """
        Private method, does one iteration over either the validation or test set (controlled by
        mode) and tests the model performance.
        """
        if mode == 'test':
            log.info("Testing model...")
            data_iter_function = self.data_feeder.iterate_test_data
        elif mode == 'val':
            log.info("Validating model...")
            data_iter_function = self.data_feeder.iterate_val_data
        else:
            log.error("Unknown mode {} when calling _test()".format(mode))
            raise ValueError
        epoch_losses = []
        epoch_accs = []
        epoch_per_class_accs = []
        epoch_predictions = []
        epoch_targets = []
        proteins = []
        for prots, samples, targets in data_iter_function():
            output = self.model.validation_function(*(samples + targets))
            loss = output['loss']
            accuracy = output['accuracy']
            per_class_accs = output['per_class_accs']
            predictions = output['predictions']
            epoch_losses.append(loss)
            epoch_accs.append(accuracy)
            epoch_per_class_accs.append(per_class_accs)
            epoch_predictions.append(predictions)
            proteins.append(prots)
            epoch_targets.append(targets)

        epoch_loss_means = np.mean(np.array(epoch_losses), axis=0)
        epoch_acc_means = np.mean(np.array(epoch_accs), axis=0)
        epoch_per_class_accs_means = np.mean(np.array(epoch_per_class_accs), axis=0)
        log.info(
            "{0}: loss mean: {1} acc mean: {2}".format(mode, epoch_loss_means, epoch_acc_means))
        return epoch_loss_means, epoch_acc_means, epoch_per_class_accs_means, epoch_predictions, epoch_targets, proteins

    def get_test_hidden_activations(self):
        """
        Get example activations of all hidden layers in the current model by running a forward
        pass on a single mini-batch from the test set.

        :return: protein codes, targets (ground truths), activations, predictions
            for the single mini-batch from the test set that was used.
        """
        for prots, samples, targets in self.data_feeder.iterate_test_data():
            # do just a single minibatch
            output = self.model.get_hidden_activations(samples)
            return prots, targets, output[:-1], output[-1]


def _build_enz_feeder_model_trainer(config, model_name=None, start_epoch=0,
                                    force_download=False, force_memmaps=False,
                                    force_grids=False, force_split=False):
    """
    Helper function, that constructs a GridsDisjointClassifer model given a config, and constructs
    the respective EnzymesGridFeeder (provides the minibatches from the train / test / val. sets)
    and the ModelTrainer that will be used to train the model.

    :param config: the contents of a config.yaml that specifies all the details around the model.
    :param model_name: if left out, a unique name will be given to the model.
    :param start_epoch: default is 0, can be set to something else if a training is being continued.
    :param force_download: see EnzymeDataManager
    :param force_memmaps: see EnzymeDataManager
    :param force_grids: see EnzymeDataManager
    :param force_split: see EnzymeDataManager
    :return: data_feeder, model, model_trainer
    """
    data_manager = EnzymeDataManager(data_dir=config['data']['dir'],
                                     enzyme_classes=config['proteins']['enzyme_trees'],
                                     force_download=force_download,
                                     force_memmaps=force_memmaps,
                                     force_grids=force_grids,
                                     force_split=force_split,
                                     grid_size=config['proteins']['grid_side'],
                                     split_strategy=config['training']['split_strategy'])

    data_feeder = EnzymesGridFeeder(data_manager=data_manager,
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training'][
                                        'init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    num_channels=config['proteins']['n_channels'],
                                    grid_size=config['proteins']['grid_side'])
    if model_name is None:
        current_time = datetime.datetime.now()
        suffix = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(10))
        model_name = "grids_{}_{}-classes_{}-{}-{}_{}-{}".format(suffix,
                                                                 config["proteins"]["n_classes"],
                                                                 current_time.month,
                                                                 current_time.day,
                                                                 current_time.year,
                                                                 current_time.hour,
                                                                 current_time.minute)
    model = GridsDisjointClassifier(name=model_name,
                                    n_classes=config['proteins']['n_classes'],
                                    network=get_network(config['training']['network']),
                                    grid_size=config['proteins']['grid_side'],
                                    n_channels=config['proteins']['n_channels'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    learning_rate=config['training']['learning_rate'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder, first_epoch=start_epoch)
    return data_feeder, model, trainer


def train_enz_from_grids(config, model_name=None, start_epoch=0,
                         force_download=False, force_memmaps=False,
                         force_grids=False, force_split=False):
    """
    Utility function to train a GridsDisjointClassifier model on the train set, consisting of
    already precomputed electron density grids. The model need not exist, but if it does exist
    the model trainer will try to load the previous training history and continue (without loading
    any checkpointed parameters, though - this is left to the user).

    :param config: the contents of a config.yaml, specifying the details for the training
    :param model_name: can be left out, then a unique name will be given to the newly trained model
    :param start_epoch: default is 0; can be set to something else in case a previous training must
        be continued.
    :param force_download: see EnzymeDataManager
    :param force_memmaps: see EnzymeDataManager
    :param force_grids: see EnzymeDataManager
    :param force_split: see EnzymeDataManager
    """
    _, _, trainer = _build_enz_feeder_model_trainer(config, model_name=model_name,
                                                    start_epoch=start_epoch,
                                                    force_download=force_download,
                                                    force_memmaps=force_memmaps,
                                                    force_grids=force_grids,
                                                    force_split=force_split)
    save_config(config, os.path.join(trainer.monitor.get_model_dir(), "config.yaml"))
    if start_epoch != 0:
        trainer.monitor.load_model("params_{}ep_best.npz".format(start_epoch),
                                   network=trainer.model.get_output_layers())
    trainer.train(epochs=config['training']['epochs'])


def test_enz_from_grids(config, model_name, params_file, mode='test'):
    """
    Utility function to test a GridsDisjointClassifer model on the test (or optionally validation)
    set. The model must already exist (and must have been trained).

    The function also saves the predictions, targets and protein codes resulting from the testing
    into pickles (under the model directory).

    :param config: the contents of config.yaml for the model. Must match the configuration with
        which the model was originally trained.
    :param model_name: name of the model (should be unique)
    :param params_file: file with parameter weights (from a previous training) that should be
        loaded into the model before it gets tested.
    :param mode: whether to test on the test set ('test') or validation set ('val')
    """
    _, model, trainer = _build_enz_feeder_model_trainer(config,
                                                        model_name=model_name)
    trainer.monitor.load_model(params_filename=params_file, network=model.get_output_layers())
    if mode == 'test':
        _, _, _, test_predictions, test_targets, proteins = trainer.test()
    else:  # mode == 'val'
        _, _, _, test_predictions, test_targets, proteins = trainer._test(mode='val')

    # make the shapes to be (N x n_classes)
    test_predictions = np.asarray(test_predictions).reshape((-1, config['proteins']['n_classes']))
    test_targets = np.asarray(test_targets).reshape((-1, config['proteins']['n_classes']))

    save_pickle(os.path.join(trainer.monitor.get_model_dir(), "{}_predictions.pickle".format(mode)),
                test_predictions)
    save_pickle(os.path.join(trainer.monitor.get_model_dir(), "{}_targets.pickle".format(mode)),
                test_targets)
    save_pickle(os.path.join(trainer.monitor.get_model_dir(), "{}_proteins.pickle".format(mode)),
                proteins)


def get_hidden_activations(config, model_name, params_file):
    """
    Utility function to get the hidden activations of the hidden layers in a GridsDisjointClassifier
    model. The model must have already been trained. The hidden activations are produced for a
    single minibatch from the test set.

    :param config: the contents of config.yaml for the model. Must match the configuration with
        which the model was originally trained.
    :param model_name: name of the model (should be unique)
    :param params_file: file with parameter weights (from a previous training) that should be
        loaded into the model before the hidden activations are extracted.
    :return: protein codes, targets (ground truths), activations, predictions
        for the single mini-batch from the test set that was used to get the hidden activations.
    """
    _, model, trainer = _build_enz_feeder_model_trainer(config, model_name=model_name)
    trainer.monitor.load_model(params_filename=params_file, network=model.get_output_layers())
    prots, targets, activations, preds = trainer.get_test_hidden_activations()
    return prots, targets, preds, activations
