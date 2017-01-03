import numpy as np
import pickle
import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


class ProgressView(object):
    """
    Plot train, validation and test error and accuracy over time (epochs).
    """

    def __init__(self, model_name, data_dir, history_file=None, history_dict=None, mean_window=5):
        self.name = model_name
        self.model_figures_path = os.path.join(data_dir, "figures", model_name)
        self.mean_window = mean_window
        if history_dict is not None:
            self.data = history_dict
        elif history_file is not None:
            self.data = pickle.load(history_file)
        else:
            log.error("You must pass a dictionary or a file with the loss history" +
                      "to the ProgressView.")
            raise ValueError

    def save(self):
        self._save(artifacts=['train_loss', 'val_loss'], y_range=[-0.5, 3], filename='loss_history.png')
        self._save(artifacts=['train_accuracy', 'val_accuracy'], y_range=[-1, 2], filename='accuracy_history.png')

    def _save(self, artifacts=list(['train_loss', 'val_loss']), y_range=list([-1, 1]), filename='loss_history.png'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        empty = True
        for artifact in artifacts:
            values = np.asarray(self.data[artifact])
            if values.size != 0:
                empty = False
                for i in range(values.shape[1]):
                    running_mean_vals = self.running_mean(values[:, i], self.mean_window)
                    if artifact.startswith('train'):
                        ax.plot(running_mean_vals, label='{} {}'.format(artifact, i))
                    else:
                        ax.plot(running_mean_vals, '--', label='{} {}'.format(artifact, i))
            else:
                log.warning("No history for {}".format(artifact))
                continue
        if not empty:
            ax.set_ylim(y_range)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=False, shadow=False, ncol=5, prop={'size': 8})
            # ax.legend()
            if not os.path.exists(self.model_figures_path):
                os.makedirs(self.model_figures_path)
            fig.savefig(os.path.join(self.model_figures_path, filename))
        plt.close(fig)

    @staticmethod
    def running_mean(x, window):
        return np.convolve(x, np.ones((window,)) / window, mode='valid')
