import numpy as np
import colorlog as log
import logging
import threading
import string
import random
import datetime
import os

from protfun.visualizer.netview import NetworkView
from utils.np_utils import pp_array

from protfun.visualizer.progressview import ProgressView
from protfun.models.model_monitor import ModelMonitor
from protfun.visualizer.performance_view import PerformanceAnalyser
from protfun.data_management.data_feed import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import get_network
from protfun.config import save_config
from utils import save_pickle, load_pickle

log.basicConfig(level=logging.DEBUG)


class ModelTrainer(object):
    def __init__(self, model, data_feeder, val_frequency=10):
        self.model = model
        self.data_feeder = data_feeder
        self.val_frequency = val_frequency
        self.monitor = ModelMonitor(outputs=model.get_output_layers(), data_dir=data_feeder.get_data_dir(),
                                    name=model.get_name())
        self.network_view = NetworkView(data_dir=self.monitor.get_model_dir())
        self.current_max_train_acc = np.array(0.85)
        self.current_max_val_acc = np.array(0.0)
        # save training history data
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

    def train(self, epochs=100, generate_progress_plot=True):
        try:
            log.info("Training...")
            save_pickle(file_path=[os.path.join(self.monitor.get_model_dir(), "train_prot_codes.pickle"),
                                   os.path.join(self.monitor.get_model_dir(), "val_prot_codes.pickle"),
                                   os.path.join(self.monitor.get_model_dir(), "test_prot_codes.pickle")],
                        data=[self.data_feeder.get_train_data(),
                              self.data_feeder.get_val_data(),
                              self.data_feeder.get_test_data()])
            self.network_view.save_network_graph(self.model.get_output_layers(), "network.png")
            if generate_progress_plot:
                self.plot_progress()
            self._train(epochs)
            self.monitor.save_history_and_model(self.history, epoch_count=epochs)
        except (KeyboardInterrupt, SystemExit):
            log.warning("Training is interrupted")
            self.monitor.save_history_and_model(self.history)

    def _train(self, epochs=100):
        steps_before_validate = 0
        used_proteins = set()
        for e in xrange(epochs):
            log.info("Unique proteins used during training so far: {}".format(len(used_proteins)))
            epoch_losses = []
            epoch_accs = []
            for proteins, inputs in self.data_feeder.iterate_train_data():
                output = self.model.train_function(*inputs)
                loss = output['loss']
                accuracy = output['accuracy']
                per_class_accs = output['per_class_accs']
                predictions = output['predictions']

                # this can be enabled to profile the forward pass
                # self.model.train_function.profile.print_summary()

                epoch_losses.append(loss)
                epoch_accs.append(accuracy)
                self.history['train_loss'].append(loss)
                self.history['train_accuracy'].append(accuracy)
                self.history['train_per_class_accs'].append(per_class_accs)
                self.history['train_predictions'].append(predictions)
                steps_before_validate += 1

                used_proteins = used_proteins | set(proteins)

            epoch_loss_means = np.mean(np.array(epoch_losses), axis=0)
            epoch_acc_means = np.mean(np.array(epoch_accs), axis=0)
            log.info("train: epoch {0} loss mean: {1} acc mean: {2}".format(e, epoch_loss_means, epoch_acc_means))

            if np.alltrue(epoch_acc_means >= self.current_max_train_acc):
                samples_per_class = self.data_feeder.get_samples_per_class()
                log.info("Augmenting dataset: doubling the samples per class ({0})".format(
                    2 * self.data_feeder.get_samples_per_class()))
                self.current_max_train_acc = epoch_acc_means
                samples_per_class *= 2
                self.data_feeder.set_samples_per_class(samples_per_class)

            # validate the model
            if e % self.val_frequency == 0:
                self.validate(steps_before_validate, e)
                steps_before_validate = 0

    def validate(self, steps_before_validate, epoch):
        val_loss_means, val_acc_means, val_per_class_accs_means, val_predictions, val_targets, _ = self._test(
            mode='val')
        self.history['val_loss'] += [val_loss_means] * steps_before_validate
        self.history['val_accuracy'] += [val_acc_means] * steps_before_validate
        self.history['val_per_class_accs'] += [val_per_class_accs_means] * steps_before_validate
        self.history['val_predictions'].append(val_predictions)
        self.history['val_targets'].append(val_targets)
        # save parameters if an improvement is observed
        if np.alltrue(val_acc_means > self.current_max_val_acc):
            self.current_max_val_acc = val_acc_means
            self.monitor.save_model(epoch, "meanvalacc{0}".format(pp_array(self.current_max_val_acc)))

    def test(self):
        log.warning(
            "You are testing a model with the secret test set! " +
            "You are not allowed to change the model after seeing the results!!! ")
        return self._test(mode='test')

    def _test(self, mode='test'):
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
        for prots, inputs in data_iter_function():
            output = self.model.validation_function(*inputs)
            loss = output['loss']
            accuracy = output['accuracy']
            per_class_accs = output['per_class_accs']
            predictions = output['predictions']
            epoch_losses.append(loss)
            epoch_accs.append(accuracy)
            epoch_per_class_accs.append(per_class_accs)
            epoch_predictions.append(predictions)
            proteins.append(prots)
            # TODO: this will break when the DataFeeder is different, so refactor too.
            epoch_targets.append(inputs[1:])

        epoch_loss_means = np.mean(np.array(epoch_losses), axis=0)
        epoch_acc_means = np.mean(np.array(epoch_accs), axis=0)
        epoch_per_class_accs_means = np.mean(np.array(epoch_per_class_accs), axis=0)
        log.info("{0}: loss mean: {1} acc mean: {2}".format(mode, epoch_loss_means, epoch_acc_means))
        return epoch_loss_means, epoch_acc_means, epoch_per_class_accs_means, epoch_predictions, epoch_targets, proteins

    def plot_progress(self):
        t = threading.Timer(5.0, self.plot_progress)
        t.daemon = True
        t.start()
        progress = ProgressView(model_name=self.model.get_name(),
                                data_dir=self.monitor.get_model_dir(), history_dict=self.history)
        progress.save()


def train_enz_from_memmaps(config):
    data_feeder = EnzymesMolDataFeeder(data_dir=config['data']['dir'],
                                       minibatch_size=config['training']['minibatch_size'],
                                       init_samples_per_class=config['training']['init_samples_per_class'],
                                       prediction_depth=config['proteins']['prediction_depth'],
                                       enzyme_classes=config['proteins']['enzyme_trees'])
    current_time = datetime.datetime.now()
    suffix = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(10))
    model_name = "molmap-{}_{}-classes_{}-{}-{}_{}-{}".format(suffix, config["proteins"]["n_classes"],
                                                              current_time.month,
                                                              current_time.day,
                                                              current_time.year,
                                                              current_time.hour,
                                                              current_time.minute)
    model = MemmapsDisjointClassifier(name=model_name, n_classes=config['proteins']['n_classes'],
                                      network=get_network(config['training']['network']),
                                      minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=config['training']['epochs'])
    save_config(config, os.path.join(trainer.monitor.get_model_dir(), "config.yaml"))


def _build_enz_feeder_model_trainer(config, model_name=None):
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'])
    if model_name is None:
        current_time = datetime.datetime.now()
        suffix = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(10))
        model_name = "grids_{}_{}-classes_{}-{}-{}_{}-{}".format(suffix, config["proteins"]["n_classes"],
                                                                 current_time.month,
                                                                 current_time.day,
                                                                 current_time.year,
                                                                 current_time.hour,
                                                                 current_time.minute)
    model = GridsDisjointClassifier(name=model_name,
                                    n_classes=config['proteins']['n_classes'],
                                    network=get_network(config['training']['network']),
                                    grid_size=64,
                                    minibatch_size=config['training']['minibatch_size'],
                                    learning_rate=config['training']['learning_rate'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder, val_frequency=10)
    return data_feeder, model, trainer


def train_enz_from_grids(config):
    _, _, trainer = _build_enz_feeder_model_trainer(config)
    trainer.train(epochs=config['training']['epochs'])
    save_config(config, os.path.join(trainer.monitor.get_model_dir(), "config.yaml"))


def test_enz_from_grids(config, model_name, params_file, mode='test'):
    _, model, trainer = _build_enz_feeder_model_trainer(config, model_name=model_name)
    trainer.monitor.load_model(params_filename=params_file,
                               network=model.get_output_layers())
    if mode == 'test':
        _, _, _, test_predictions, test_targets, proteins = trainer.test()
    else:  # mode == 'val'
        _, _, _, test_predictions, test_targets, proteins = trainer._test(mode='val')

    # make the shapes to be (N x n_classes)
    test_predictions = np.asarray(test_predictions).reshape((-1, config['proteins']['n_classes']))
    test_targets = np.asarray(test_targets).reshape((-1, config['proteins']['n_classes']))

    # compute the ROC curve
    pa = PerformanceAnalyser(n_classes=config['proteins']['n_classes'], y_expected=test_targets,
                             y_predicted=test_predictions, data_dir=trainer.monitor.get_model_dir(),
                             model_name="grids_{}".format(mode))
    pa.plot_ROC()
