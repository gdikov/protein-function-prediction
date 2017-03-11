import cPickle
import logging

import colorlog as log
import numpy as np
import os

# os.environ["THEANO_FLAGS"] = "device=gpu0,lib.cnmem=2000,base_compiledir=~/.atcremers11"
# os.environ["THEANO_FLAGS"] = "device=gpu6,lib.cnmem=2500,base_compiledir=~/.tiptop"
from protfun.config import get_config
from protfun.models import test_enz_from_grids, get_hidden_activations
from protfun.utils import save_pickle, load_pickle
from protfun.visualizer.molview import MoleculeView
from protfun.visualizer.progressview import ProgressView
from protfun.visualizer.performance_view import PerformanceAnalyser

log.basicConfig(level=logging.DEBUG)

root_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
root_config = get_config(root_config_path)
root_models_dir = os.path.join(root_config['data']['dir'], 'models')
model_names = os.listdir(root_models_dir)
model_dirs = [os.path.join(root_models_dir, model_dirname) for model_dirname in model_names]


def measure_performance():
    model_names = [
        "grids_cdriafrvod_166-classes_1-21-2017_18-23"
    ]
    for model_name in model_names:
        model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
        local_config = get_config(os.path.join(model_dir, "config.yaml"))

        param_files = [f for f in os.listdir(model_dir) if f.startswith("params_") and "best" in f]
        epochs = np.array([int(f.split('_')[1][:-2]) for f in param_files], dtype=np.int32)
        best_params_file = param_files[np.argmax(epochs)]

        test_enz_from_grids(config=local_config, model_name=model_name,
                            params_file=best_params_file, mode="test")
        # test_enz_from_grids(config=local_config, model_name=model_name, params_file=best_params_file, mode="val")


def create_history_plots(config, model_name, history_filename, checkpoint=None, until=None):
    model_dir = os.path.join(config["data"]["dir"], "models", model_name)
    # create plots for the training history of this model
    history_file = os.path.join(model_dir, history_filename)

    if os.path.exists(history_file):
        with open(os.path.join(model_dir, history_filename), mode='r') as history_file:
            train_history = cPickle.load(history_file)

        if until:
            for key in train_history:
                train_history[key] = train_history[key][:until]
        view = ProgressView(model_name=model_name, data_dir=model_dir, history_dict=train_history)
        view.save(checkpoint=checkpoint)
        log.info("Saved progress plots for: {}".format(model_name))
    else:
        log.warning("Missing history file for: {}".format(model_name))


def create_performance_plots(config, model_name):
    data_dir = config["data"]["dir"]
    path_to_model_dir = os.path.join(data_dir, "models", model_name)
    path_to_predictions = os.path.join(path_to_model_dir, "test_predictions.pickle")
    path_to_targets = os.path.join(path_to_model_dir, "test_targets.pickle")

    model_predictions = load_pickle(path_to_predictions)
    targets = load_pickle(path_to_targets)

    pa = PerformanceAnalyser(n_classes=2, y_expected=targets,
                             y_predicted=model_predictions, data_dir=data_dir,
                             model_name="grids_val")
    pa.plot_ROC()


def save_hidden_activations():
    model_name = "grids_lzjqkixqvb_2-classes_1-24-2017_21-58"
    model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
    local_config = get_config(os.path.join(model_dir, "config.yaml"))
    local_config["training"]["minibatch_size"] = 4

    param_files = [f for f in os.listdir(model_dir) if f.startswith("params_") and "best" in f]
    epochs = np.array([int(f.split('_')[1][:-2]) for f in param_files], dtype=np.int32)
    best_params_file = param_files[np.argmax(epochs)]
    prots, targets, preds, activations = get_hidden_activations(config=local_config,
                                                                model_name=model_name,
                                                                params_file=best_params_file)
    save_pickle(file_path=os.path.join(model_dir, "activations.pickle"), data=activations)
    save_pickle(file_path=os.path.join(model_dir, "activations_targets.pickle"), data=targets)
    save_pickle(file_path=os.path.join(model_dir, "activations_preds.pickle"), data=preds)
    save_pickle(file_path=os.path.join(model_dir, "activations_prots.pickle"), data=prots)


def visualize_hidden_activations():
    import numpy as np
    # model_name = "transferred_model"
    # model_dir = os.path.join(root_config["data"]["dir"], "models", model_name)
    file_path = "/home/valor/workspace/DLCV_ProtFun/data/restricted/processed_multi_128/3V7T/grid.memmap"
    grid = np.memmap(file_path, mode='r',
                     dtype="float32").reshape((1, -1,
                                               128,
                                               128,
                                               128))

    for i in range(21):
        viewer = MoleculeView(data_dir=root_config['data']['dir'],
                              data={"potential": None, "density": grid[0, i]},
                              info={"name": "test"})

        viewer.density3d()


if __name__ == "__main__":
    # model_name = "grids_scjcdzajzw_2-classes_1-25-2017_10-44"
    # model_name = "grids_qvrvatyodl_2-classes_1-24-2017_1-49"
    #
    # history_filename = "train_history_best.pickle"
    #
    # minibatch_size = get_config(os.path.join(os.path.join(root_config["data"]["dir"],
    #                                                       "models", model_name),
    #                                          "config.yaml"))["training"]["minibatch_size"]
    #
    # factor_restricted = 400 // minibatch_size
    # factor_all = 5000 // minibatch_size
    #
    # import re
    # checkpoint = map(int, re.findall(r'\d+', history_filename))[0] * factor_restricted
    # print("checkpoint:", checkpoint)
    # create_history_plots(model_name=model_name,
    #                      history_filename=history_filename,
    #                      checkpoint=checkpoint)
    # create_history_plots()
    # measure_performance()
    # save_hidden_activations()
    visualize_hidden_activations()
    # create_history_plots("grids_umyfbmdxjg_2-classes_1-21-2017_16-50", "train_history_best.pickle", checkpoint=8950,
    #                      until=30000)
