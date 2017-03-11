import numpy as np
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.metrics import auc, roc_curve
from protfun.utils.data_utils import load_pickle

classes = ['3.4.21', '3.4.24']
colors = ['#991012', '#c4884e', '#93bf8d', '#a3dbff']
sns.set_palette(colors)


class ROCView(object):
    """
    ROCView generates and plots the ROC curves of a model.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ax, self.fig = self._init_ROC()

    def _init_ROC(self):
        """
        initialise the plots (figure, axes)
        :return:
        """
        sns.set_style("whitegrid")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_aspect(1)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.axes().set_aspect('equal', 'datalim')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False positive rate', size=10)
        plt.ylabel('True positive rate', size=10)
        plt.title('Receiver operating characteristic', size=15)

        return ax, fig

    def add_curve(self, predicted, expected, label):
        """
        computes and draws a ROC curve for a given label (class)

        :param predicted: a numpy array of predictions
        :param expected: a numpy array of targets
        :param label: the class for which the ROC is computed and drawn
        :return:
        """
        fpr, tpr, _ = roc_curve(expected, predicted, drop_intermediate=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='{0} (AUC = {1:0.2f})'.format(label, roc_auc))

    def save_anc_close(self, filename):
        """
        saves the figure into a file.

        :param filename: the name of the file for the figure of the ROC curve
        :return:
        """
        # Put a legend below current axis
        self.ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=1, prop={'size': 9},
                       frameon=True)
        path_to_fig = os.path.join(self.data_dir, 'figures', filename)
        self.fig.savefig(filename=path_to_fig, bbox_inches='tight')


def add_curve(view, dir, label, suffix=""):
    """
    unpacks the pickeled data and passes it to the method responsible for ROC curve computation.

    :param view: the ROCViewer object
    :param dir: the directory where the predictions and targets are stored
    :param label: the class for which a ROC curve is computed and drawn
    :param suffix: additional identifying information for the filename, e.g. epoch count or accuracy
    :return:
    """
    predictions = np.asarray(
        load_pickle(os.path.join(dir, "test_predictions{}.pickle".format(suffix))))[:, 0]
    targets = np.asarray(load_pickle(os.path.join(dir, "test_targets{}.pickle".format(suffix))))[:,
              0]
    view.add_curve(predictions, targets, label)


if __name__ == "__main__":
    base_dir = "/home/valor/workspace/DLCV_ProtFun/data/final"

    experiment_map = [
        ("restricted_single_64", "single input channel, 64 x 64 x 64"),
        # ("restricted_single_128", "single input channel, 128 x 128 x 128"),
        # ("restricted_multi_64", "multiple input channels, 64 x 64 x 64"),
        # ("restricted_multi_128", "multiple input channels, 128 x 128 x 128"),
    ]

    # view = ROCView(data_dir=os.path.dirname(__file__))
    # for dir, label in experiment_map:
    #     add_curve(view, os.path.join(base_dir, "strict", dir), label)
    # view.save_anc_close("ROC_combined_strict.png")

    view = ROCView(data_dir=os.path.dirname(__file__))
    for dir, label in experiment_map:
        add_curve(view, os.path.join(base_dir, "naive", dir), label)
    view.save_anc_close("ROC_combined_naive.png")

    # with open('/home/valor/workspace/DLCV_ProtFun/small_molecules/history_test_randtransl_xent_230k.pkl', 'rb') as f:
    #     history_test = cPickle.load(f)
    #     y = np.asarray(
    #         history_test).squeeze().transpose()  # AUC
    #     t = ((0,), (1,)) * np.ones_like(y)
    #     view = ROCView(data_dir=os.path.dirname(__file__))
    #     view.add_curve(y, t, "")
    #     view.save_anc_close("ROC_small_molecules.png")
