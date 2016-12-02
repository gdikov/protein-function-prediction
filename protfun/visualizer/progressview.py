import numpy as np
import pickle
import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


class ProgressView(object):
    """
    Plot train, validation and test error and accuracy over time (epoches).
    """

    def __init__(self, model_name='model1', history_file=None, history_dict=None):
        self.name = model_name
        self.model_figures_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               "../../data/figures/", model_name)
        if history_dict is not None:
            self.data = history_dict
        elif history_file is not None:
            self.data = pickle.load(history_file)
        else:
            log.error("You must pass a dictionary or a file with the loss history" +
                      "to the ProgressView.")
            raise ValueError

    def save(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        train_losses = np.asarray(self.data['train_loss'])
        valid_losses = np.asarray(self.data['val_loss'])
        if train_losses.size != 0 and valid_losses.size != 0:
            for i in range(train_losses.shape[1]):
                plt.plot(train_losses[:, i], label='train loss {0}'.format(i))
                plt.plot(valid_losses[:, i], label='valid loss {0}'.format(i))
            # TODO: make x ticks to be synced with self.data['time_epoch']
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='best')
            if not os.path.exists(self.model_figures_path):
                os.makedirs(self.model_figures_path)
            plt.savefig(os.path.join(self.model_figures_path, 'loss_history.png'))
            plt.clf()
        else:
            log.warning("No loss history to visualize")
