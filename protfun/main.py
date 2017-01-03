import os

os.environ["THEANO_FLAGS"] = "device=gpu2,lib.cnmem=0"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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


def train_enz_from_grids():
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'])
    model = GridsDisjointClassifier(n_classes=config['proteins']['n_classes'], network=basic_convnet, grid_size=64,
                                    minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=config['training']['epochs'])


def test_enz_from_grids():
    data_feeder = EnzymesGridFeeder(data_dir=config['data']['dir'],
                                    minibatch_size=config['training']['minibatch_size'],
                                    init_samples_per_class=config['training']['init_samples_per_class'],
                                    prediction_depth=config['proteins']['prediction_depth'],
                                    enzyme_classes=config['proteins']['enzyme_trees'])
    model = GridsDisjointClassifier(n_classes=config['proteins']['n_classes'], network=basic_convnet, grid_size=64,
                                    minibatch_size=config['training']['minibatch_size'])
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.monitor.load_model(model_name="params_54ep_meanvalacc[ 0.90322578  0.88306451].npz",
                               network=model.get_output_layers())
    trainer.test()


if __name__ == "__main__":
    # train_enz_from_memmaps()
    train_enz_from_grids()
    # visualize()
