import numpy as np
import os
import lasagne

class ModelMonitor():
    """
    Monitors the model during training and testing. Logs the error and accuracy values
    and creates checkpoints whenever the mean validation error is being improved.
    Optionally dumps the model status on KeyInterrupt.
    """
    def __init__(self, outputs, name='dummy_model'):
        self.network_outputs = outputs
        self.name = name
        self.path_to_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              "../../data/models/", self.name)
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
        print("INFO: Saving {0} model parameters...".format(lasagne.layers.count_params(self.network_outputs,
                                                                                        trainable=True)))
        filename = 'params'
        if epoch_count >= 0:
            filename += '_{0}ep'.format(epoch_count)
        if msg != '':
            filename += '_'+msg
        np.savez(os.path.join(self.path_to_model_dir, '{0}.npz'.format(filename)),
                 *lasagne.layers.get_all_param_values(self.network_outputs, trainable=True))

    def load_model(self, model_name, network):
        """
        Loads the weigths from file and initialize the network.

        :param model_name: the filename to be used
        :param network: the network to be initialised
        :return:
        """
        if model_name[-4:] != '.npz':
            print("ERROR: Model not found")
            raise ValueError

        with np.load(os.path.join(self.path_to_model_dir, model_name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(network, param_values, trainable=True)

    def save_train_history(self, history):
        with open(os.path.join(self.path_to_model_dir, "train_history.tsv", 'w')) as f:
            f.write("# train_loss21 train_loss24 train_acc21 train_acc24 "
                    "val_loss21 val_loss24 val_acc21 val_acc24 time_in_epochs")
            for tl, ta, vl, va, t in zip(history['train_loss'],
                                         history['train_accuracy'],
                                         history['val_loss'],
                                         history['val_accuracy'],
                                         history['time_epoch']):
                f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".
                        format(tl[0], tl[1], ta[0], ta[1], vl[0], vl[1], va[0], va[1], t))

