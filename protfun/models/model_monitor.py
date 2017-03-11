import numpy as np
import os
import lasagne
import cPickle

from protfun.utils.log import setup_logger

log = setup_logger("model_monitor")


class ModelMonitor(object):
    """
    Monitors the model during training and testing. Logs the error and accuracy values
    and can creates checkpoints of the model parameters (triggered in the ModelTrainer whenever the
    mean validation error is being improved).

    Optionally dumps the model status on KeyInterrupt.
    """

    def __init__(self, outputs, data_dir, name):
        """
        :param outputs: lasagne output layers of the neural network of the monitored model.
            Used to checkpoint the model parameters during training.
        :param data_dir: data directory under which the monitor will create a folder for the
            currently monitored model (or use an existing one, if already present). The path is:
            data_dir/models/<model_name>
        :param name: name of the currently monitored model
        """
        self.network_outputs = outputs
        self.name = name
        self.path_to_model_dir = os.path.join(data_dir, "models", self.name)
        if not os.path.exists(self.path_to_model_dir):
            os.makedirs(self.path_to_model_dir)
        pass

    def save_model(self, epoch_count=-1, msg=''):
        """
        Dumps the model parameters into a file. The number of epochs on which it is trained is
        logged in the filename.
        :param epoch_count: the number of epochs that the model is trained for
        :param msg: an optional suffix for the produced parameters file
        """
        log.info("Saving {0} model parameters".format(
            lasagne.layers.count_params(self.network_outputs, trainable=True)))
        filename = 'params'
        if epoch_count >= 0:
            filename += '_{0}ep'.format(epoch_count)
        if msg != '':
            filename += '_' + msg
        np.savez(os.path.join(self.path_to_model_dir, '{0}.npz'.format(filename)),
                 *lasagne.layers.get_all_param_values(self.network_outputs, trainable=True))

    def load_model(self, params_filename, network):
        """
        Loads the weights from file and initializes the provided network with them.

        :param params_filename: the filename (not full path) from which to load the parameters.
            The file should be in the monitored model's directory.
        :param network: the last lasagne layer(s) of the network into which the parameters should
            be loaded.
        """
        if params_filename[-4:] != '.npz':
            log.error("Model not found")
            raise ValueError

        with np.load(
                os.path.join(self.path_to_model_dir, params_filename)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(network, param_values, trainable=True)

    def save_train_history(self, history, epoch_count, save_human_readable=False, msg=''):
        """
        save_train_history saves the training history of the model so far in the form of either
        a pickle or a human readable .tsv file.

        :param history: dictionary containing various info about the training history of the model.
        :param epoch_count: number of epochs into the training so far
        :param save_human_readable: whether to save as a .tsv human readable format or as a pickle.
        :param msg: an optional suffix to the file name
        """
        log.info("Saving training history")
        # Refactored: instead of saving in a text file, dump the history as a pickle and provide
        # the saving to a human-readable format as an option
        filename = 'train_history_ep{}'.format(epoch_count)
        if msg != '':
            filename += '_' + msg
        with open(os.path.join(self.path_to_model_dir, "{0}.pickle".format(filename)),
                  mode='wb') as f:
            cPickle.dump(history, f)

        if save_human_readable:
            # TODO: save in a plain text, tsv, csv, or something better
            with open(os.path.join(self.path_to_model_dir, "train_history.tsv"),
                      mode='w') as f:
                f.write("# BEGIN PREAMBLE")
                f.write("# {}".format(history.keys()))
                f.write("# END PREAMBLE")
                f.write("# BEGIN DATA")
                # TODO: fill in with the formatted data
                f.write("# END DATA")

    def load_train_history(self, epoch):
        """
        Loads an already existing training history in order to continue the training from there.
        Note that this only loads the history, but the model parameters need to be loaded into the
        neural network independently to properly continue a training from where it stopped.

        :param epoch: used to determine the correct train_history_ep{epoch}_best.pickle file to
            load the history from.
        :return: the loaded history dict.
        """
        try:
            with open(os.path.join(self.path_to_model_dir,
                                   'train_history_ep{}_best.pickle'.format(
                                       epoch)), mode='rb') as f:
                history = cPickle.load(f)
                log.info(
                    "Loaded history from previous training, continuing from where it was stopped.")
                return history
        except:
            log.info(
                "No previous history was loaded, proceeding with a new training.")
            return None

    def save_history_and_model(self, history, epoch_count=-1, msg='',
                               save_human_readable=False):
        """
        Saves both the training history and the model parameters.

        :param history: the training history dict
        :param epoch_count: the current training epoch (used in the file names)
        :param msg: an optional suffix to the file names
        :param save_human_readable: to save the training history in a human readable format (.tsv)
            or not (.pickle)
        """
        self.save_model(epoch_count, msg)
        self.save_train_history(history, epoch_count, save_human_readable, msg)

    def get_model_dir(self):
        """
        Get the (newly created) model directory path for the monitored model.

        :return: the model dir path
        """
        return self.path_to_model_dir
