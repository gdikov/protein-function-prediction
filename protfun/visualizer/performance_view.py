import numpy as np
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import cPickle
import os

from sklearn.metrics import roc_curve, auc
from scipy import interp

classes = ['3.4.21', '3.4.24']


class PerformanceAnalyser(object):
    """
    PerformanceAnalyser can create ROC plots for different classifiers.
    """

    def __init__(self, n_classes, y_expected, y_predicted, data_dir, model_name):
        """
        :param n_classes: how many classes does the classifier predict for
        :param y_expected: a numpy array of expected class labels
        (1-hot encoded)
        :param y_predicted: a numpy array of prediction scores
        :param data_dir: data directory under which the ROC image will be saved
        :param model_name: name of the model the ROC is produced for
        """
        self.n_classes = n_classes
        self.y_expected = np.asarray(y_expected)
        self.y_predicted = np.asarray(y_predicted)
        self.data_dir = data_dir
        self.model_name = model_name

    def plot_ROC(self, export_figure=True):
        """
        plot_ROC plots an ROC curve and saves it under the data_dir.
        :param export_figure - whether to save a figure under data_dir/figures
        """
        false_positive_rate, true_positive_rate, roc_auc = self._compute_ROC()
        if export_figure:
            sns.set_style("whitegrid")
            fig = self._plot_ROC(false_positive_rate, true_positive_rate, roc_auc)
            path_to_fig = os.path.join(self.data_dir, 'figures')
            if not os.path.exists(path_to_fig):
                os.makedirs(path_to_fig)
            fig.savefig(filename=os.path.join(path_to_fig, '{0}_ROC.png'.format(self.model_name)))

    def _plot_ROC(self, false_positive_rate, true_positive_rate, roc_auc):
        """

        :param false_positive_rate: false positive rate coordinates of the
        points in the ROC curve
        :param true_positive_rate: true positive rate coordinates of the points
        in the ROC curve
        :param roc_auc: AUC score (plotted in the legend)
        :return: the pyplot figure where everything has been plotted
        """
        # line width
        lw = 2
        fig = plt.figure()
        ax = plt.subplot(111)

        # plot the micro ROC curves
        plt.plot(false_positive_rate["micro"], true_positive_rate["micro"],
                 label='Micro-average ROC (AUC = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 linestyle='-.',
                 linewidth=4,
                 solid_capstyle="round")

        # plot the macro ROC curves
        plt.plot(false_positive_rate["macro"], true_positive_rate["macro"],
                 label='Macro-average ROC (AUC = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 linestyle='-.',
                 linewidth=4,
                 solid_capstyle="round")

        # plot the ROC curves per class
        for i in range(self.n_classes):
            class_name = classes[i] if self.n_classes == 2 else "{}".format(i)
            plt.plot(false_positive_rate[i], true_positive_rate[i],
                     lw=lw,
                     label='ROC of class {0} (AUC = {1:0.2f})'
                           ''.format(class_name, roc_auc[i]))

        # configure plot limits, labels and title
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.00])
        plt.xlabel('False positive rate', size=12)
        plt.ylabel('True positive rate', size=12)
        plt.title('Receiver operating characteristic for multi-label classification', size=15)

        # Put a legend below current axis
        ax.legend(
            loc='lower right',
            fancybox=True,
            shadow=True,
            ncol=1,
            prop={'size': 12},
            frameon=True)

        # adjust plot styles (colors, surrounding lines)
        sns.despine()
        colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
        sns.set_palette(colors)

        return fig

    def _compute_ROC(self):
        """
        Compute ROC curve and ROC area for each class.
        The function produces three dictionaries: the false positive
        rate coordinates, true positive rate coordinates and an AUC scores.
        Each dictionary has 3 keys:
            * "micro" : fpr, tpr and AUC for the flattened predictions for all
            classes (i.e. all predictions combined). Weakly represented classes
            thus contribute less to the "micro" curve.
            * "macro" : fpr, tpr and AUC are computed as an average of the ROC
            curves for each of the classes. Thus every class is treated as
            equally important in the "macro" curve.
            * i: one fpr, tpr and AUC for each of the classes (i is the index
            of the class).
        :return: false positive rate, true positive rate, roc_auc dictionaries
        """
        false_positive_rate = dict()
        true_positive_rate = dict()
        roc_auc = dict()

        for i in range(self.n_classes):
            false_positive_rate[i], true_positive_rate[i], _ = roc_curve(
                self.y_expected[:, i], self.y_predicted[:, i])
            roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

        # Compute micro-average ROC curve and ROC area
        false_positive_rate["micro"], true_positive_rate["micro"], _ = \
            roc_curve(self.y_expected.ravel(), self.y_predicted.ravel())
        roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate(
            [false_positive_rate[i]
             for i in range(self.n_classes)]))

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
    path_to_hist_dict = os.path.join(
        data_dir, 'models', 'from_grids_disjoint_classifier',
        'train_history_from_grids_disjoint_classifier_interrupted.pickle')
    with open(path_to_hist_dict, 'rb') as f:
        history_dict = cPickle.load(f)
    print(history_dict.keys())
    val_predictions = history_dict['val_predictions']
    val_targets = history_dict['val_targets']

    val_predictions = np.asarray(val_predictions)
    # TODO: fix this, it's now a sigmoid
    # take the score of predicting 1, since it's a softmax per class and not
    # sigmoid
    val_predictions = np.exp(val_predictions[:, :, :, :, 1])
    # transpose to (epochs x mini_batches x samples in mini_batch x classes)
    val_predictions = np.transpose(val_predictions, (0, 1, 3, 2))
    # reshape to a flattened version, N x num_classes
    val_predictions = np.reshape(val_predictions,
                                 (-1, val_predictions.shape[-1]))

    # transform the shape of the targets in the same way
    val_targets = np.asarray(val_targets)
    val_targets = np.transpose(val_targets, (0, 1, 3, 2))
    val_targets = np.reshape(val_targets, (-1, val_targets.shape[-1]))

    pa = PerformanceAnalyser(n_classes=2, y_expected=val_targets,
                             y_predicted=val_predictions, data_dir=data_dir,
                             model_name="grids_val")
    pa.plot_ROC()
