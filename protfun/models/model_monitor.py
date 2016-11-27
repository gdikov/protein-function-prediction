import numpy as np
import os
import lasagne

class ModelMonitor():
    """
    Monitors the model during training and testing. Logs the error and accuracy values
    and creates checkpoints whenever the mean validation error is being improved.
    Optionally dumps the model status on KeyInterrupt.
    """
    def __init__(self, outputs, name='model'):
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

    def gather_train_history(self, history):

        pass
