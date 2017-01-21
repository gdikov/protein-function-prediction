import cPickle
import logging

import colorlog as log
import numpy as np
import os

from protfun.config import get_config
from protfun.models import test_enz_from_grids, get_hidden_activations
from protfun.utils import save_pickle, load_pickle
from protfun.visualizer.molview import MoleculeView
from protfun.visualizer.progressview import ProgressView

log.basicConfig(level=logging.DEBUG)

root_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
root_config = get_config(root_config_path)
root_models_dir = os.path.join(root_config['data']['dir'], 'models')
model_names = os.listdir(root_models_dir)
model_dirs = [os.path.join(root_models_dir, model_dirname) for model_dirname in model_names]


def create_performance_plots():
    # import numpy as np
    # model_names = ["grids_eqzchnjmlm_2-classes_1-14-2017_20-38",
    #                "grids_djizkcxnql_2-classes_1-14-2017_20-41",
    #                "grids_dtxaolqeau_2-classes_1-14-2017_20-40",
    #                "grids_fdbwpafdxv_2-classes_1-14-2017_23-22",
    #                "grids_jexjiuroyc_2-classes_1-14-2017_20-38",
    #                "grids_lrrnplamxq_2-classes_1-14-2017_23-21",
    #                "grids_oukazqxksk_2-classes_1-14-2017_20-36"]
    model_names = ["grids_classifier_2_classes_1_13_2017_18-26",
                   "grids_classifier_2_classes_1_13_2017_18-54",
                   "grids_kvhlnusegp_2-classes_1-14-2017_1-21",
                   "grids_xbkzeqcsnc_2-classes_1-14-2017_1-21"]
    for model_name in model_names:
        model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
        local_config = get_config(os.path.join(model_dir, "config.yaml"))

        param_files = [f for f in os.listdir(model_dir) if f.startswith("params_") and "meanvalacc" in f]
        epochs = np.array([int(f.split('_')[1][:-2]) for f in param_files], dtype=np.int32)
        best_params_file = param_files[np.argmax(epochs)]

        test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="test")
        test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="val")


def create_history_plots():
    for model_dir, model_name in zip(model_dirs, model_names):
        # create plots for the training history of this model
        history_file = os.path.join(model_dir, 'train_history.pickle')
        if os.path.exists(history_file):
            with open(os.path.join(model_dir, 'train_history.pickle'), mode='r') as history_file:
                train_history = cPickle.load(history_file)
            view = ProgressView(model_name=model_name, data_dir=model_dir, history_dict=train_history)
            view.save()
            log.info("Saved progress plots for: {}".format(model_name))
        else:
            log.warning("Missing history file for: {}".format(model_name))


def save_hidden_activations():
    model_name = "grids_eqzchnjmlm_2-classes_1-14-2017_20-38"
    model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
    local_config = get_config(os.path.join(model_dir, "config.yaml"))
    local_config["training"]["minibatch_size"] = 1

    param_files = [f for f in os.listdir(model_dir) if f.startswith("params_") and "meanvalacc" in f]
    epochs = np.array([int(f.split('_')[1][:-2]) for f in param_files], dtype=np.int32)
    best_params_file = param_files[np.argmax(epochs)]
    prots, activations = get_hidden_activations(config=local_config, model_name=model_name,
                                                params_file=best_params_file)
    save_pickle(file_path=os.path.join(model_dir, "activations.pickle"),
                data=activations)


def visualize_hidden_activations():
    model_name = "grids_eqzchnjmlm_2-classes_1-14-2017_20-38"
    model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
    activations = load_pickle(file_path=os.path.join(model_dir, "activations.pickle"))

    # for i in range(64):
    viewer = MoleculeView(data_dir=root_config['data']['dir'],
                          data={"potential": None, "density": activations[0][0, 1]},
                          info={"name": "test"})

    viewer.density3d()


if __name__ == "__main__":
    # create_history_plots()
    # create_performance_plots()
    # save_hidden_activations()
    visualize_hidden_activations()
