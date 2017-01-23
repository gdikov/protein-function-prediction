import datetime
import logging
import random
import string
import threading

import colorlog as log
import numpy as np
import os
from protfun.utils import save_pickle

from protfun.config import save_config
from protfun.data_management.data_feed import EnzymesGridFeeder
from protfun.models import GridsDisjointClassifier
from protfun.models.model_monitor import ModelMonitor
from protfun.networks import get_network
from protfun.utils.np_utils import pp_array
from protfun.visualizer.netview import NetworkView
from protfun.visualizer.performance_view import PerformanceAnalyser
from protfun.visualizer.progressview import ProgressView

log.basicConfig(level=logging.DEBUG)


class ModelTrainer(object):
    def __init__(self, model, data_feeder, checkpoint_frequency=200, first_epoch=0):
        self.model = model
        self.data_feeder = data_feeder
        self.checkpoint_frequency = checkpoint_frequency
        self.monitor = ModelMonitor(outputs=model.get_output_layers(), data_dir=data_feeder.get_data_dir(),
                                    name=model.get_name())
        self.network_view = NetworkView(data_dir=self.monitor.get_model_dir())
        self.current_max_train_acc = np.array(0.85)
        self.current_max_val_acc = np.array(0.0)
        self.first_epoch = first_epoch
        # save training history data
        history = self.monitor.load_train_history()
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

    def train(self, epochs=100, generate_progress_plot=True):
        try:
            log.info("Training model {}".format(self.model.get_name()))
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
        used_proteins = set()
        validations = 0
        val_accuracies = []
        max_val_acc = 0
        current_minibatch = 0

        iterate_val = self.data_feeder.iterate_val_data()
        for e in xrange(self.first_epoch, self.first_epoch + epochs):
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
                used_proteins = used_proteins | set(proteins)

                # validate now
                try:
                    val_proteins, val_inputs = iterate_val.next()
                except StopIteration:
                    iterate_val = self.data_feeder.iterate_val_data()
                    val_proteins, val_inputs = iterate_val.next()
                validations += 1

                output = self.model.validation_function(*val_inputs)
                val_loss = output['loss']
                val_accuracy = output['accuracy']
                val_per_class_accs = output['per_class_accs']
                val_predictions = output['predictions']

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                val_accuracies.append(val_accuracy)
                self.history['val_per_class_accs'].append(val_per_class_accs)
                self.history['val_predictions'].append(val_predictions)
                self.history['val_targets'].append(val_inputs[1:])

                current_minibatch += 1
                if current_minibatch % self.checkpoint_frequency == self.checkpoint_frequency - 1:
                    log.info("Attempt to checkpoint: {} minibatches have passed".format(current_minibatch))
                    if sum(val_accuracies) / validations > max_val_acc:
                        max_val_acc = sum(val_accuracies) / validations
                        self.monitor.save_history_and_model(self.history, epoch_count=e, msg="best")

            epoch_loss_means = np.mean(np.array(epoch_losses), axis=0)
            epoch_acc_means = np.mean(np.array(epoch_accs), axis=0)
            log.info("train: epoch {0} loss mean: {1} acc mean: {2}".format(e, epoch_loss_means, epoch_acc_means))
            log.info("average val accuracy: {}".format(sum(val_accuracies) / validations))

            # if np.alltrue(epoch_acc_means >= self.current_max_train_acc):
            #     samples_per_class = self.data_feeder.get_samples_per_class()
            #     log.info("Augmenting dataset: doubling the samples per class ({0})".format(
            #         2 * self.data_feeder.get_samples_per_class()))
            #     self.current_max_train_acc = epoch_acc_means
            #     samples_per_class *= 2
            #     self.data_feeder.set_samples_per_class(samples_per_class)

            # # validate the model
            # if e % self.val_frequency == 0:
            #     self.validate(steps_before_validate, e)
            #     self.monitor.save_history_and_model(self.history, epoch_count=epochs)
            #     steps_before_validate = 0

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

    def get_test_hidden_activations(self):
        for prots, inputs in self.data_feeder.iterate_test_data():
            # do just a single minibatch
            return prots, self.model.get_hidden_activations(inputs[0])

    def plot_progress(self):
        t = threading.Timer(5.0, self.plot_progress)
        t.daemon = True
        t.start()
        progress = ProgressView(model_name=self.model.get_name(),
                                data_dir=self.monitor.get_model_dir(), history_dict=self.history)
        progress.save()


def _build_enz_feeder_model_trainer(config, model_name=None, start_epoch=0):
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'],
                                    num_channels=config['proteins']['n_channels'],
                                    grid_size=config['proteins']['grid_side'])
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
                                    grid_size=config['proteins']['grid_side'],
                                    n_channels=config['proteins']['n_channels'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    learning_rate=config['training']['learning_rate'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder, first_epoch=start_epoch)
    return data_feeder, model, trainer


def train_enz_from_grids(config, model_name=None, start_epoch=0):
    _, _, trainer = _build_enz_feeder_model_trainer(config, model_name=model_name, start_epoch=start_epoch)
    save_config(config, os.path.join(trainer.monitor.get_model_dir(), "config.yaml"))
    if start_epoch != 0:
        trainer.monitor.load_model("params_{}ep_best.npz".format(start_epoch),
                                   network=trainer.model.get_output_layers())
    trainer.train(epochs=config['training']['epochs'])


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


def get_hidden_activations(config, model_name, params_file):
    _, model, trainer = _build_enz_feeder_model_trainer(config, model_name=model_name)
    trainer.monitor.load_model(params_filename=params_file,
                               network=model.get_output_layers())
    prots, activations = trainer.get_test_hidden_activations()
    return prots, activations
