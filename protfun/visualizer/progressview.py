import numpy as np
import pickle
import os
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

from protfun.utils.log import setup_logger

log = setup_logger("progressview")

sns.set_style("whitegrid")
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
sns.set_palette(colors)

text = {
    'titles': {
        'loss': 'Loss progression during training',
        'accuracy': 'Accuracy progression during training',
        'per_class_accs': 'Accuracy progression per class during training'
    },
    'y_labels': {
        'loss': 'Loss',
        'accuracy': 'Accuracy',
        'per_class_accs': 'Accuracy'
    }
}

classes = ['3.4.21', '3.4.24']


class ProgressView(object):
    """
    Plot train error, validation error, train accuracy and validation accuracy
    over time (mini-batches).
    """

    def __init__(self, model_name, data_dir, history_file=None,
                 history_dict=None, mean_window=50):
        """
        :param model_name: name of the model the plots are produced for
        :param data_dir: directory in which the plots should be saved
        :param history_file: file containing a history_dict.
        :param history_dict: dictionary containing the error and accuracy
        history. Should have the following keys:
            {
            'train_loss' : [...],
            'val_loss' : [...],
            'train_accuracy' : [...],
            'val_accuracy' : [...],
            'train_per_class_accs' : [...],
            'val_per_class_accs' : [...]
            }
        where the first 4 are 1D arrays of the respective histories, and the
        last 2 are 2D arrays where for each time step there are n_classes many
        values (1 for each class).
        :param mean_window: size of the running mean window, used to smooth the
        curves when they are visualized.
        :raises: ValueError if neither history_dict nor history_file is
        provided.
        """
        self.model_name = model_name
        self.model_figures_path = os.path.join(data_dir, "figures")
        self.mean_window = mean_window
        if history_dict is not None:
            self.data = history_dict
        elif history_file is not None:
            self.data = pickle.load(history_file)
        else:
            log.error("You must pass a dictionary or a file with " +
                      "the loss history to the ProgressView.")
            raise ValueError

    def save(self, checkpoint=None):
        """
        Save the produced progress history plots.

        :param checkpoint: visualize a vertical line at the specified
        checkpoint, meant to indicate the mini-batch at which the model's
        parameters have been saved for the last time.
        """
        self._save(artifacts=['train_loss', 'val_loss'], type='loss',
                   filename='loss_history.png', checkpoint=checkpoint)
        self._save(artifacts=['train_accuracy', 'val_accuracy'],
                   type='accuracy', y_range=[0, 1],
                   filename='accuracy_history.png', checkpoint=checkpoint)
        self._save(artifacts=['train_per_class_accs', 'val_per_class_accs'],
                   type='per_class_accs', y_range=[0, 1],
                   filename='per_class_accs.png', checkpoint=checkpoint)

    def _save(self, artifacts, type, y_range=None, filename='loss_history.png', checkpoint=None):
        """

        :param artifacts: keys for the metrics that should be plotted
        :param type: 'loss', 'accuracy' or 'per_class_accs'
        :param y_range: limits for the y axis in the plot
        :param filename: name of the file that will be saved for this plot
        :param checkpoint: checkpoint mini-batch to put a vertical line on
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        empty = True
        max_length = 0
        for artifact in artifacts:
            if artifact not in self.data:
                log.warning("{} does not have artifact {}".format(self.model_name, artifact))
                continue
            values = np.asarray(self.data[artifact])
            if max_length < values.shape[0]:
                max_length = values.shape[0]
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
            ax.set_xlim([0, max_length])

            if checkpoint:
                ax.axvline(x=checkpoint, c='#e7298a')

            # define the legend
            legend_position = 'upper right' if artifacts[0].endswith('loss') else 'lower right'

            ax.legend(loc=legend_position,
                      fancybox=True,
                      shadow=True,
                      ncol=1,
                      prop={'size': 9},
                      frameon=True)
            # ax.set_title(text['titles'][type], size=15)
            ax.set_ylabel(text['y_labels'][type], size=12)
            ax.set_xlabel("Mini-batch count", size=12)

            # adjust styles before saving

            if not os.path.exists(self.model_figures_path):
                os.makedirs(self.model_figures_path)
            fig.savefig(os.path.join(self.model_figures_path, filename), bbox_inches='tight')
        plt.close(fig)

    def _plot_single(self, fig, values, artifact):
        """
        Plots a signle curve.

        :param fig: a pyplot figure object
        :param values: a 1D numpy array of values that will be plotted
        :param artifact: key of the plotted artifact
        """
        if artifact.startswith('train'):
            values = self.running_mean(values, self.mean_window)
            fig.plot(values,
                     label="Training set",
                     alpha=0.6,
                     linewidth=0.5)
        else:
            values = self.running_mean(values, self.mean_window)
            fig.plot(values, '-',
                     label="Validation set",
                     c="black",
                     linewidth=1,
                     solid_capstyle="projecting")

    def _plot_multiple(self, fig, values, artifact):
        """
        Plots multiple curves.

         :param fig: a pyplot figure object
        :param values: a 1D numpy array of values that will be plotted
        :param artifact: key of the plotted artifact.
        """
        for i in range(0, values.shape[1]):
            vals = values[:, i]
            class_name = classes[i] if values.shape[1] == 2 else "{}".format(i)
            if artifact.startswith('train'):
                vals = self.running_mean(vals, self.mean_window)

                fig.plot(vals,
                         label="Training set, class: {}".format(class_name),
                         alpha=0.6,
                         linewidth=0.5)
            else:
                vals = self.running_mean(vals, self.mean_window)
                fig.plot(vals, '-',
                         label="Validation set, class: {}".format(class_name),
                         c="black",
                         linewidth=1,
                         solid_capstyle="projecting")

    @staticmethod
    def running_mean(x, window):
        """
        Computes a running mean smoothing with a given window size.

        :param x: values to smooth
        :param window: window size
        :return: the smoothed values
        """
        return np.convolve(x, np.ones((window,)) / window, mode='valid')
