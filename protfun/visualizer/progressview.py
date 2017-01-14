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
        self.model_name = model_name
        self.model_figures_path = os.path.join(data_dir, "figures")
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
        self._save(artifacts=['train_loss', 'val_loss'], filename='loss_history.png')
        self._save(artifacts=['train_accuracy', 'val_accuracy'], y_range=[-0.5, 1.5], filename='accuracy_history.png')
        self._save(artifacts=['train_per_class_accs', 'val_per_class_accs'], y_range=[-0.5, 1.5],
                   filename='per_class_accs.png')

    def _save(self, artifacts=list(['train_loss', 'val_loss']), y_range=None, filename='loss_history.png'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        empty = True
        for artifact in artifacts:
            if artifact not in self.data:
                log.warning("{} does not have artifact {}".format(self.model_name, artifact))
                continue
            values = np.asarray(self.data[artifact])
            if values.size != 0:
                empty = False
                if len(values.shape) < 2:
                    self._plot_single(ax, values, artifact)
                else:
                    self._plot_multiple(ax, values, artifact)
            else:
                log.warning("No history for {}".format(artifact))
                continue
        if not empty:
            if y_range is not None:
                ax.set_ylim(y_range)
            # ax.set_xlim([0, 10000])
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

    def _plot_single(self, fig, values, artifact):
        if artifact.startswith('train'):
            values = self.running_mean(values, self.mean_window)
            fig.plot(values, label=artifact)
        else:
            fig.plot(values, '--', label=artifact)

    def _plot_multiple(self, fig, values, artifact):
        for i in range(0, values.shape[1]):
            vals = values[:, i]
            if artifact.startswith('train'):
                vals = self.running_mean(vals, self.mean_window)
                fig.plot(vals, label="train_class_{}".format(i))
            else:
                fig.plot(vals, '--', label="val_class_{}".format(i))

    @staticmethod
    def running_mean(x, window):
        return np.convolve(x, np.ones((window,)) / window, mode='valid')
