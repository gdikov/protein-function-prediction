import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle
import os

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp


class PerformanceAnalyser(object):
    def __init__(self, n_classes, y_expected, y_predicted, data_dir, model_name):
        self.n_classes = n_classes
        self.y_expected = np.asarray(y_expected)
        self.y_predicted = np.asarray(y_predicted)
        self.data_dir = data_dir
        self.model_name = model_name

    def plot_ROC(self, export_figure=True):
        """
        Plot the ROC for all classes
        :return:
        """
        false_positive_rate, true_positive_rate, roc_auc = self._compute_ROC()
        if export_figure:
            fig = self._plot_ROC(false_positive_rate, true_positive_rate, roc_auc)
            path_to_fig = os.path.join(self.data_dir, 'figures/{0}_ROC.png'.format(self.model_name))
            fig.savefig(filename=path_to_fig)

    def _plot_ROC(self, false_positive_rate, true_positive_rate, roc_auc):
        # Plot all ROC curves
        # TODO: make variable number of colors, according to the class number
        lw = 2
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(false_positive_rate["micro"], true_positive_rate["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(false_positive_rate["macro"], true_positive_rate["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=False, shadow=False, ncol=3, prop={'size': 8})
        # plt.legend(loc="lower right")

        return fig

    def _compute_ROC(self):
        """
        Compute ROC curve and ROC area for each class
        :return: false positive rate, true positive rate, roc_auc
        """
        false_positive_rate = dict()
        true_positive_rate = dict()
        roc_auc = dict()

        for i in range(self.n_classes):
            false_positive_rate[i], true_positive_rate[i], _ = roc_curve(self.y_expected[:, i], self.y_predicted[:, i])
            roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

        # Compute micro-average ROC curve and ROC area
        false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(self.y_expected.ravel(),
                                                                                 self.y_predicted.ravel())
        roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([false_positive_rate[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, false_positive_rate[i], true_positive_rate[i])

        # Finally average it and compute AUC
        mean_tpr /= float(self.n_classes)

        false_positive_rate["macro"] = all_fpr
        true_positive_rate["macro"] = mean_tpr
        roc_auc["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])

        return false_positive_rate, true_positive_rate, roc_auc


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    path_to_hist_dict = os.path.join(data_dir, 'models', 'from_grids_disjoint_classifier',
                                     'train_history_from_grids_disjoint_classifier_interrupted.pickle')
    with open(path_to_hist_dict, 'rb') as f:
        history_dict = cPickle.load(f)
    print(history_dict.keys())
    val_predictions = history_dict['val_predictions']
    val_targets = history_dict['val_targets']

    val_predictions = np.asarray(val_predictions)
    # take the score of predicting 1, since it's a softmax per class and not sigmoid
    val_predictions = np.exp(val_predictions[:, :, :, :, 1])
    # transpose to (epochs x mini_batches x samples in mini_batch x classes)
    val_predictions = np.transpose(val_predictions, (0, 1, 3, 2))
    # reshape to a flattened version, N x num_classes
    val_predictions = np.reshape(val_predictions, (-1, val_predictions.shape[-1]))

    # transform the shape of the targets in the same way
    val_targets = np.asarray(val_targets)
    val_targets = np.transpose(val_targets, (0, 1, 3, 2))
    val_targets = np.reshape(val_targets, (-1, val_targets.shape[-1]))

    pa = PerformanceAnalyser(n_classes=2, y_expected=val_targets, y_predicted=val_predictions, data_dir=data_dir,
                             model_name="test")
    pa.plot_ROC()
