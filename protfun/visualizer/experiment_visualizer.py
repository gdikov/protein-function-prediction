import os
import re
import numpy as np

from protfun.models import get_hidden_activations, get_best_params
from protfun.utils import save_pickle, load_pickle
from protfun.visualizer.molview import MoleculeView
from protfun.visualizer.progressview import ProgressView
from protfun.visualizer.performance_view import PerformanceAnalyser
from protfun.utils.log import get_logger

log = get_logger("experiment_visualizer")


def create_history_plots(config, model_name, checkpoint=None, until=None):
    """
    Creates training history diagrams for a desired model that has already been trained.

    :param config: a config dictionary, containing the contents of the config.yaml for the trained
        model. You can load it from file with protfun.config.get_config(file_path)
    :param model_name: name (model id) of the model to create diagrams for. Corresponds to the name
        of the model directory under <data_dir>/models
    :param checkpoint: (optional) specify a mini-batch at which you want a vertical line visualize
        to represent when the model was check-pointed
    :param until: (optional) restrict the number of mini-batches shown in the progress diagram
    """
    model_dir = os.path.join(config["data"]["dir"], "models", model_name)

    hisotry_files = [f for f in os.listdir(model_dir) if f.startswith("train_history_ep")]
    epochs = np.array([int(re.search(r'\d+', f.split('_')[2]).group()) for f in hisotry_files],
                      dtype=np.int32)
    history_filename = hisotry_files[np.argmax(epochs)]
    # create plots for the training history of this model
    history_file = os.path.join(model_dir, history_filename)

    if os.path.exists(history_file):
        train_history = load_pickle(history_file)

        if until:
            for key in train_history:
                train_history[key] = train_history[key][:until]
        view = ProgressView(model_name=model_name, data_dir=model_dir, history_dict=train_history)
        view.save(checkpoint=checkpoint)
        log.info("Saved progress plots for: {}".format(model_name))
    else:
        log.warning("Missing history file for: {}".format(model_name))


def create_performance_plots(config, model_name, n_classes):
    """
    Create ROC plots for a given model. The model must have already been trained and **TESTED**,
    so that test_predictions.pickle and test_targets.pickle are present in the model's directory.

    :param config: a config dictionary, containing the contents of the config.yaml for the trained
        model. You can load it from file with protfun.config.get_config(file_path)
    :param model_name: name (model id) of the model to create diagrams for. Corresponds to the name
        of the model directory under <data_dir>/models
    :param n_classes: number of different classes the model had to discriminate (classify).
        If bigger than 2, the ROC plot will only contain micro- and macro- average curves over all
        classes.
    """
    data_dir = config["data"]["dir"]
    model_dir = os.path.join(data_dir, "models", model_name)
    path_to_predictions = os.path.join(model_dir, "test_predictions.pickle")
    path_to_targets = os.path.join(model_dir, "test_targets.pickle")

    model_predictions = load_pickle(path_to_predictions)
    targets = load_pickle(path_to_targets)

    pa = PerformanceAnalyser(n_classes=n_classes, y_expected=targets,
                             y_predicted=model_predictions, data_dir=data_dir,
                             model_name=model_name)
    pa.plot_ROC()
    log.info("Saved ROC plots for: {}".format(model_name))


def save_hidden_activations(config, model_name):
    """
    Save activations from the hidden layers of an already trained model for a small set of samples
    from the test set. The activations are saved under the model's directory:
        * activations.pickle - the activations of the hidden layers
        * activations_targets.pickle - the ground truth target classes for the chosen samples
        * activations_preds.pickle - the predictions (scores) for the chosen samples
        * activations_prots.pickle - the protein codes of the chosen samples

    :param config: a config dictionary, containing the contents of the config.yaml for the trained
        model. You can load it from file with protfun.config.get_config(file_path)
    :param model_name: name (model id) of the model to create diagrams for. Corresponds to the name
        of the model directory under <data_dir>/models
    """

    model_dir = os.path.join(config["data"]["dir"], "models", model_name)
    # set a lower mini-batch size to fit into smaller GPU for this run
    config["training"]["minibatch_size"] = 4

    best_params_file = get_best_params(config, model_name)
    prots, targets, preds, activations = get_hidden_activations(config=config,
                                                                model_name=model_name,
                                                                params_file=best_params_file)
    save_pickle(file_path=os.path.join(model_dir, "activations.pickle"), data=activations)
    save_pickle(file_path=os.path.join(model_dir, "activations_targets.pickle"), data=targets)
    save_pickle(file_path=os.path.join(model_dir, "activations_preds.pickle"), data=preds)
    save_pickle(file_path=os.path.join(model_dir, "activations_prots.pickle"), data=prots)
    log.info("Saved hidden activations for: {}".format(model_name))


def visualize_molecule(config, grid_size, mol_grid_filepath):
    """
    Visualize the 3D el. density map of a single protein molecule, from a grid.memmap file.

    :param config: a config dictionary, containing the contents of the config.yaml with which the
        dataset was preprocessed.
    :param grid_size: number of points on each side of the 3D grid (e.g. 64)
    :param mol_grid_filepath: path to the grid.memmap file for this molecule. Usually those are
        to be found under <data_dir>/processed.
    """
    import numpy as np
    grid = np.memmap(mol_grid_filepath, mode='r',
                     dtype="float32").reshape((1, -1, grid_size, grid_size, grid_size))

    for i in range(grid.shape[1]):
        viewer = MoleculeView(data_dir=config['data']['dir'],
                              data={"potential": None, "density": grid[0, i]},
                              info={"name": "test"})
        viewer.density3d()
