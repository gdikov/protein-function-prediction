import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


from sklearn.metrics import roc_curve, auc
from scipy import interp

class PerformanceAnalyser(object):
    def __init__(self, n_classes, y_expected, y_predicted, model_name='dummy_model'):
        self.n_classes = n_classes
        self.y_expected = y_expected
        self.y_predicted = y_predicted
        self.model_name = model_name
        pass

    def plot_ROC(self, export_figure=True):
        """
        Plot the ROC for all classes
        :return:
        """
        false_positive_rate, true_positive_rate, roc_auc = self._compute_ROC()
        if export_figure:
            fig = self._plot_ROC(false_positive_rate, true_positive_rate, roc_auc)
            plt.savefig(filename='../../data/figures/{0}_ROC.png'.format(self.model_name))


    def _plot_ROC(self, false_positive_rate, true_positive_rate, roc_auc):
        # Plot all ROC curves
        # TODO: make variable number of colors, according to the class number
        lw = 2
        fig = plt.figure()
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
        plt.legend(loc="lower right")

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