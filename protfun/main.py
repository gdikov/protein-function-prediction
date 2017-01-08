import os
import numpy as np

os.environ["THEANO_FLAGS"] = "device=gpu2,lib.cnmem=0"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from protfun.visualizer.performance_view import PerformanceAnalyser
from protfun.data_management.data_feed import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import ModelTrainer
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import basic_convnet
from protfun.config import get_config

config = get_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))


def train_enz_from_memmaps():
    data_feeder = EnzymesMolDataFeeder(data_dir=config['data']['dir'],
                                       minibatch_size=config['training']['minibatch_size'],
                                       init_samples_per_class=config['training']['init_samples_per_class'],
                                       prediction_depth=config['proteins']['prediction_depth'],
                                       enzyme_classes=config['proteins']['enzyme_trees'])
    model = MemmapsDisjointClassifier(n_classes=config['proteins']['n_classes'], network=basic_convnet,
                                      minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=config['training']['epochs'])


def _build_enz_feeder_model_trainer():
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'])
    model = GridsDisjointClassifier(n_classes=config['proteins']['n_classes'], network=basic_convnet, grid_size=64,
                                    minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    return data_feeder, model, trainer


def train_enz_from_grids():
    _, _, trainer = _build_enz_feeder_model_trainer()
    trainer.train(epochs=config['training']['epochs'])


def test_enz_from_grids():
    _, model, trainer = _build_enz_feeder_model_trainer()
    trainer.monitor.load_model(model_name="params_160ep_meanvalacc[|0.953|0.964].npz",
                               network=model.get_output_layers())

    _, _, test_predictions, test_targets = trainer.test()

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
    # train_enz_from_grids()
    test_enz_from_grids()
    # visualize()
