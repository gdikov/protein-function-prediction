import numpy as np
import os
import lasagne
import colorlog as log
import logging
import cPickle

log.basicConfig(level=logging.DEBUG)


class ModelMonitor(object):
    """
    Monitors the model during training and testing. Logs the error and accuracy values
    and creates checkpoints whenever the mean validation error is being improved.
    Optionally dumps the model status on KeyInterrupt.
    """

    def __init__(self, outputs, data_dir, name):
        self.network_outputs = outputs
        self.name = name
        self.path_to_model_dir = os.path.join(data_dir, "models", self.name)
        if not os.path.exists(self.path_to_model_dir):
            os.makedirs(self.path_to_model_dir)
        pass

    def save_model(self, epoch_count=-1, msg=''):
        """
        Dumps the model weights into a file. The number of epochs on which it is trained is
        logged in the filename.
        :param epoch_count: the number of epochs that the model is trained
        :return:
        """
        log.info("Saving {0} model parameters".format(lasagne.layers.count_params(self.network_outputs,
                                                                                  trainable=True)))
        filename = 'params'
        if epoch_count >= 0:
            filename += '_{0}ep'.format(epoch_count)
        if msg != '':
            filename += '_' + msg
        np.savez(os.path.join(self.path_to_model_dir, '{0}.npz'.format(filename)),
                 *lasagne.layers.get_all_param_values(self.network_outputs, trainable=True))

    def load_model(self, params_filename, network):
        """
        Loads the weigths from file and initialize the network.

        :param params_filename: the filename to be used
        :param network: the network to be initialised
        :return:
        """
        if params_filename[-4:] != '.npz':
            log.error("Model not found")
            raise ValueError

        with np.load(os.path.join(self.path_to_model_dir, params_filename)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(network, param_values, trainable=True)

    def save_train_history(self, history, save_human_readable=False, msg=''):
        log.info("Saving training history")
        # Refactored: instead of saving in a text file, dump the history as a pickle and provide
        # the saving to a human-readable format as an option
        filename = 'train_history'
        if msg != '':
            filename += '_' + msg
        with open(os.path.join(self.path_to_model_dir, "{0}.pickle".format(filename)), mode='wb') as f:
            cPickle.dump(history, f)

        if save_human_readable:
            # TODO: save in a plain text, tsv, csv, or something better
            with open(os.path.join(self.path_to_model_dir, "train_history.tsv"), mode='w') as f:
                f.write("# BEGIN PREAMBLE")
                f.write("# {}".format(history.keys()))
                f.write("# END PREAMBLE")
                f.write("# BEGIN DATA")
                # TODO: fill in with the formatted data
                f.write("# END DATA")

    def load_train_history(self):
        try:
            with open(os.path.join(self.path_to_model_dir, 'train_history_best.pickle'), mode='rb') as f:
                history = cPickle.load(f)
                log.info("Loaded history from previous training, continuing from where it was stopped.")
                return history
        except:
            log.info("No previous history was loaded, proceeding with a new training.")
            return None

    def save_history_and_model(self, history, epoch_count=-1, msg='', save_human_readable=False):
        self.save_model(epoch_count, msg)
        self.save_train_history(history, save_human_readable, msg)

    def get_model_dir(self):
        return self.path_to_model_dir
