import numpy as np
import matplotlib.pyplot as plt
import os

class ProgressView():
    """
    Plot train, validation and test error and accuracy over time (epoches).
    """
    def __init__(self, model_name='dummy_model', history_file=None, history_dict=None):
        self.name = model_name
        self.path_to_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              "../../data/models/", model_name)
        if history_file is None:
            # plot from history dict
            self.data = history_dict
        else:
            self.data = dict()
            with open(history_file, 'r') as f:
                # skip the preamble
                lines = f.readlines()[1:]
                # TODO: fill in self.data with the values read

    def plot_loss_vs_time(self):
        train_losses = np.asarray(self.data['train_loss'])
        valid_losses = np.asarray(self.data['val_loss'])
        for i in range(train_losses.shape[1]):
            plt.plot(train_losses[:, i], label='train loss {0}'.format(i))
            plt.plot(valid_losses[:, i], label='valid loss {0}'.format(i))
        # TODO: make x ticks to be synced with self.data['time_epoch']
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        fig_path = os.path.join(self.path_to_model_dir, 'figures')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, 'loss_history.png'))