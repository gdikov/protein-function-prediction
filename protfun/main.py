import numpy as np
import string, random, datetime, sys, os

os.environ["THEANO_FLAGS"] = "device=gpu0,lib.cnmem=0"
sys.setrecursionlimit(10000)
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from protfun.visualizer.performance_view import PerformanceAnalyser
from protfun.data_management.data_feed import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import ModelTrainer
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import get_network
from protfun.config import get_config, save_config

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = get_config(config_path)


def train_enz_from_memmaps():
    data_feeder = EnzymesMolDataFeeder(data_dir=config['data']['dir'],
                                       minibatch_size=config['training']['minibatch_size'],
                                       init_samples_per_class=config['training']['init_samples_per_class'],
                                       prediction_depth=config['proteins']['prediction_depth'],
                                       enzyme_classes=config['proteins']['enzyme_trees'])
    current_time = datetime.datetime.now()
    suffix = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(10))
    model_name = "molmap-{}_{}-classes_{}-{}-{}_{}-{}".format(suffix, config["proteins"]["n_classes"],
                                                              current_time.month,
                                                              current_time.day,
                                                              current_time.year,
                                                              current_time.hour,
                                                              current_time.minute)
    model = MemmapsDisjointClassifier(name=model_name, n_classes=config['proteins']['n_classes'],
                                      network=get_network(config['training']['network']),
                                      minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=config['training']['epochs'])
    save_config(config, os.path.join(trainer.monitor.get_model_dir(), "config.yaml"))


def _build_enz_feeder_model_trainer(model_name=None):
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'])
    if model_name is None:
        current_time = datetime.datetime.now()
        suffix = ''.join(random.choice(string.ascii_lowercase) for _ in xrange(10))
        model_name = "grids_{}_{}-classes_{}-{}-{}_{}-{}".format(suffix, config["proteins"]["n_classes"],
                                                                 current_time.month,
                                                                 current_time.day,
                                                                 current_time.year,
                                                                 current_time.hour,
                                                                 current_time.minute)
    model = GridsDisjointClassifier(name=model_name,
                                    n_classes=config['proteins']['n_classes'],
                                    network=get_network(config['training']['network']),
                                    grid_size=64,
                                    minibatch_size=config['training']['minibatch_size'],
                                    learning_rate=config['training']['learning_rate'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder, val_frequency=10)
    return data_feeder, model, trainer


def train_enz_from_grids():
    _, _, trainer = _build_enz_feeder_model_trainer()
    trainer.train(epochs=config['training']['epochs'])
    save_config(config, os.path.join(trainer.monitor.get_model_dir(), "config.yaml"))


def test_enz_from_grids(model_name, params_file):
    _, model, trainer = _build_enz_feeder_model_trainer(model_name)
    trainer.monitor.load_model(params_filename=params_file,
                               network=model.get_output_layers())

    _, _, _, test_predictions, test_targets, proteins = trainer.test()

    # make the shapes to be (N x n_classes)
    test_predictions = np.exp(np.asarray(test_predictions)[:, :, :, 1]).transpose((0, 2, 1)).reshape(
        (-1, config['proteins']['n_classes']))
    test_targets = np.asarray(test_targets).transpose((0, 2, 1)).reshape((-1, config['proteins']['n_classes']))

    # compute the ROC curve
    pa = PerformanceAnalyser(n_classes=config['proteins']['n_classes'], y_expected=test_targets,
                             y_predicted=test_predictions, data_dir=trainer.monitor.get_model_dir(),
                             model_name="grids_test")
    pa.plot_ROC()


if __name__ == "__main__":
    # train_enz_from_memmaps()
    train_enz_from_grids()
    # test_enz_from_grids(model_name="grids_classifier_2_classes_1_13_2017_10-7",
    #                     params_file="params_330ep_meanvalacc0.928571.npz")
    # visualize()
