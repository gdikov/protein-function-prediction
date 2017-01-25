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


class ROCView(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ax, self.fig = self._init_ROC()

    def _init_ROC(self):
        sns.set_style("whitegrid")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_aspect(1)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.axes().set_aspect('equal', 'datalim')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False positive rate', size=12)
        plt.ylabel('True positive rate', size=12)
        plt.title('Receiver operating characteristic', size=15)

        colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        sns.set_palette(colors)

        return ax, fig

    def add_curve(self, predicted, expected, label):
        fpr, tpr, _ = roc_curve(expected, predicted)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC {0} (AUC = {1:0.2f})'.format(label, roc_auc))

    def save_anc_close(self, filename):
        # Put a legend below current axis
        self.ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=1, prop={'size': 12}, frameon=True)
        path_to_fig = os.path.join(self.data_dir, 'figures', filename)
        self.fig.savefig(filename=path_to_fig)


if __name__ == "__main__":
    from protfun.utils.data_utils import load_pickle

    view = ROCView(data_dir=os.path.dirname(__file__))

    # predictions = np.asarray(load_pickle("/home/valor/workspace/DLCV_ProtFun/data/split_by_level4/restricted_multi_64/models/grids_axufeoamvo_2-classes_1-21-2017_16-41/test_predictions.pickle"))[:,0]
    # targets = np.asarray(load_pickle("/home/valor/workspace/DLCV_ProtFun/data/split_by_level4/restricted_multi_64/models/grids_axufeoamvo_2-classes_1-21-2017_16-41/test_targets.pickle"))[:,0]
    # view.add_curve(predictions, targets, "multi channel")
    #
    # predictions = np.asarray(load_pickle("/home/valor/workspace/DLCV_ProtFun/data/split_by_level4/restricted_single_64/models/grids_lbcwfjpraz_2-classes_1-21-2017_18-37/test_predictions.pickle"))[:,0]
    # targets = np.asarray(load_pickle("/home/valor/workspace/DLCV_ProtFun/data/split_by_level4/restricted_single_64/models/grids_lbcwfjpraz_2-classes_1-21-2017_18-37/test_targets.pickle"))[:,0]
    # view.add_curve(predictions, targets, "single channel")
    # view.save_anc_close("ROC_combined_strict_split.png")

    predictions = np.asarray(load_pickle(
        "/home/valor/workspace/DLCV_ProtFun/data/split_on_level3/restricted_multi_64/models/grids_asitycwovd_2-classes_1-22-2017_21-35/test_predictions.pickle"))[
                  :, 0]
    targets = np.asarray(load_pickle(
        "/home/valor/workspace/DLCV_ProtFun/data/split_on_level3/restricted_multi_64/models/grids_asitycwovd_2-classes_1-22-2017_21-35/test_targets.pickle"))[
              :, 0]
    view.add_curve(predictions, targets, "multi channel")

    predictions = np.asarray(load_pickle(
        "/home/valor/workspace/DLCV_ProtFun/data/split_on_level3/restricted_single_64/models/grids_umyfbmdxjg_2-classes_1-21-2017_16-50/test_predictions.pickle"))[
                  :, 0]
    targets = np.asarray(load_pickle(
        "/home/valor/workspace/DLCV_ProtFun/data/split_on_level3/restricted_single_64/models/grids_umyfbmdxjg_2-classes_1-21-2017_16-50/test_targets.pickle"))[
              :, 0]
    view.add_curve(predictions, targets, "single channel")
    view.save_anc_close("ROC_combined_naive_split.png")


