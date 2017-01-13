import os
import sys
import numpy as np
import datetime

os.environ["THEANO_FLAGS"] = "device=gpu0,lib.cnmem=0"
sys.setrecursionlimit(10000)
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from protfun.visualizer.performance_view import PerformanceAnalyser
from protfun.data_management.data_feed import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import ModelTrainer
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import basic_convnet, single_trunk_network, dense_network
from protfun.config import get_config

config = get_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))


def train_enz_from_memmaps():
    data_feeder = EnzymesMolDataFeeder(data_dir=config['data']['dir'],
                                       minibatch_size=config['training']['minibatch_size'],
                                       init_samples_per_class=config['training']['init_samples_per_class'],
                                       prediction_depth=config['proteins']['prediction_depth'],
                                       enzyme_classes=config['proteins']['enzyme_trees'])
    current_time = datetime.datetime.now()
    model_name = "molmap_classifier_{}_classes_{}_{}_{}_{}-{}".format(config["proteins"]["n_classes"],
                                                                      current_time.month,
                                                                      current_time.day,
                                                                      current_time.year,
                                                                      current_time.hour,
                                                                      current_time.minute)
    model = MemmapsDisjointClassifier(name=model_name, n_classes=config['proteins']['n_classes'], network=basic_convnet,
                                      minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=config['training']['epochs'])


def _build_enz_feeder_model_trainer():
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'])
    current_time = datetime.datetime.now()
    model_name = "grids_classifier_{}_classes_{}_{}_{}_{}-{}".format(config["proteins"]["n_classes"],
                                                                     current_time.month,
                                                                     current_time.day,
                                                                     current_time.year,
                                                                     current_time.hour,
                                                                     current_time.minute)
    model = GridsDisjointClassifier(name=model_name,
                                    n_classes=config['proteins']['n_classes'],
                                    network=single_trunk_network,
                                    grid_size=64,
                                    minibatch_size=config['training']['minibatch_size'],
                                    learning_rate=config['training']['learning_rate'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder, val_frequency=10)
    return data_feeder, model, trainer


def train_enz_from_grids():
    _, _, trainer = _build_enz_feeder_model_trainer()
    trainer.train(epochs=config['training']['epochs'])


def test_enz_from_grids():
    _, model, trainer = _build_enz_feeder_model_trainer()
    trainer.monitor.load_model(model_name="params_180ep_meanvalacc[|0.849|0.849].npz",
                               network=model.get_output_layers())

    _, _, _, test_predictions, test_targets, proteins = trainer.test()

    # make the shapes to be (N x n_classes)
    test_predictions = np.exp(np.asarray(test_predictions)[:, :, :, 1]).transpose((0, 2, 1)).reshape(
        (-1, config['proteins']['n_classes']))
    test_targets = np.asarray(test_targets).transpose((0, 2, 1)).reshape((-1, config['proteins']['n_classes']))

    # compute the ROC curve
    pa = PerformanceAnalyser(n_classes=config['proteins']['n_classes'], y_expected=test_targets,
                             y_predicted=test_predictions, data_dir=config['data']['dir'], model_name="grids_test")
    pa.plot_ROC()


if __name__ == "__main__":
    # train_enz_from_memmaps()
    train_enz_from_grids()
    # test_enz_from_grids()
    # visualize()
