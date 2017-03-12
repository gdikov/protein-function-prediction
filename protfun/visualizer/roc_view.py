import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve

colors = ['#991012', '#c4884e', '#93bf8d', '#a3dbff']
sns.set_palette(colors)


class ROCView(object):
    """
    ROCView generates and plots the ROC curves of a model.
    The view is created in a way that allows multiple ROC curves to be added to it before
    it is saved.

    Usage:
        >>> tpr = [0.3, 1, 1]
        >>> fpr = [0, 0.4, 1]
        >>> view = ROCView("my/data/dir")
        >>> view.add_curve(fpr=fpr, tpr=tpr, label="ROC of model 1")
        >>> # you can call view.add_curve() again if needed
        >>> view.save_and_close("example_file.png")
    """

    def __init__(self):
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
        plt.axes().set_aspect('equal')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False positive rate', size=10)
        plt.ylabel('True positive rate', size=10)
        plt.title('Receiver operating characteristic', size=15)

        return ax, fig

    def add_curve(self, fpr, tpr, label):
        """
        computes and draws a ROC curve for the given TPR and FPR, adds a legend with the specified
        label and the AUC score

        :param fpr: array, false positive rate
        :param tpr: array, true positive rate
        :param label: text to be put into the legend entry for this curve
        """
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='{0} (AUC = {1:0.2f})'.format(label, roc_auc))

    def save_and_close(self, file_path):
        """
        saves the figure into a file.

        :param file_path: path to the file for the figure of the ROC curve
        :return:
        """
        # Put a legend below current axis
        self.ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=1, prop={'size': 9},
                       frameon=True)
        self.fig.savefig(filename=file_path, bbox_inches='tight')


def micro_macro_roc(n_classes, y_expected, y_predicted):
    """
    MicroMacroROC can create the TPR (True positive rate) and FPR (false positive rate)
    for two different ROC curves based on multi-class classification results:
    * "micro" : fpr, tpr are computed for the flattened predictions for all
    classes (i.e. all predictions combined). Weakly represented classes
    thus contribute less to the "micro" curve.
    * "macro" : fpr, tpr are computed as an average of the ROC
    curves for each of the classes. Thus every class is treated as
    equally important in the "macro" curve.

    :param n_classes: how many classes does the classifier predict for
    :param y_expected: a numpy array of expected class labels
    (1-hot encoded)
    :param y_predicted: a numpy array of prediction scores
    :return: {
                "micro": (fpr, tpr),
                "macro": (fpr, tpr)
            }
    """

    # Compute micro-average ROC curve
    micro_fpr, micro_tpr, _ = roc_curve(y_expected.ravel(), y_predicted.ravel())

    # Compute macro-average ROC curve
    # First aggregate all false positive rates per class into one array
    per_class_fpr = list()
    per_class_tpr = list()
    for i in range(n_classes):
        per_class_fpr[i], per_class_tpr[i], _ = roc_curve(y_expected[:, i], y_predicted[:, i])
    all_fpr = np.unique(np.concatenate([per_class_fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, per_class_fpr[i], per_class_tpr[i])

    # Finally average it
    mean_tpr /= float(n_classes)

    macro_fpr = all_fpr
    macro_tpr = mean_tpr

    return {
        "micro": (micro_fpr, micro_tpr),
        "macro": (macro_fpr, macro_tpr)
    }
