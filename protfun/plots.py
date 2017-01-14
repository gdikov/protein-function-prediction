import os
import colorlog as log
import logging
import cPickle
import numpy as np

from protfun.config import get_config, save_config
from protfun.models import test_enz_from_grids
from protfun.visualizer.progressview import ProgressView

log.basicConfig(level=logging.DEBUG)

root_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
root_config = get_config(root_config_path)
root_models_dir = os.path.join(root_config['data']['dir'], 'models')
model_names = os.listdir(root_models_dir)
model_dirs = [os.path.join(root_models_dir, model_dirname) for model_dirname in model_names]


def create_plots():
    for model_dir, model_name in zip(model_dirs, model_names):
        # create plots for the training history of this model
        with open(os.path.join(model_dir, 'train_history.pickle'), mode='r') as history_file:
            train_history = cPickle.load(history_file)
        view = ProgressView(model_name=model_name, data_dir=model_dir, history_dict=train_history)
        view.save()
        log.info("Saved progress plots for: {}".format(model_name))

        # use the local configuration of this particular model
        local_config_path = os.path.join(model_dir, 'config.yaml')
        local_config = get_config(local_config_path)

        # find the parameters file with the best accuracy scores
        param_files = [f for f in os.listdir(model_dir) if f.startswith("params_") and "meanvalacc" in f]
        epochs = np.array([int(f.split('_')[1][:-2]) for f in param_files], dtype=np.int32)
        best_params_file = param_files[np.argmax(epochs)]

        # test the model and create an ROC curve
        test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="test")
        test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="val")


if __name__ == "__main__":
    create_plots()
    # # import numpy as np
    # model_names = ["grids_classifier_2_classes_1_13_2017_18-26",
    #                "grids_kvhlnusegp_2-classes_1-14-2017_1-21",
    #                "grids_xbkzeqcsnc_2-classes_1-14-2017_1-21",
    #                "grids_classifier_2_classes_1_13_2017_18-54"]
    # for model_name in model_names:
    #     model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
    #     local_config = get_config(os.path.join(model_dir, "config.yaml"))
    #
    #     param_files = [f for f in os.listdir(model_dir) if f.startswith("params_") and "meanvalacc" in f]
    #     epochs = np.array([int(f.split('_')[1][:-2]) for f in param_files], dtype=np.int32)
    #     best_params_file = param_files[np.argmax(epochs)]
    #
    #     test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="test")
    #     test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="val")
