import os

os.environ["THEANO_FLAGS"] = "device=gpu2,lib.cnmem=0"
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from protfun.data_management.data_feed import EnzymesMolDataFeeder, EnzymesGridFeeder
from protfun.models import ModelTrainer
from protfun.models import MemmapsDisjointClassifier, GridsDisjointClassifier
from protfun.networks import basic_convnet

data_dir = os.path.join(os.path.dirname(__file__), "../data_new")


def train_enz_from_memmaps():
    data_feeder = EnzymesMolDataFeeder(data_dir=data_dir,
                                       minibatch_size=8,
                                       init_samples_per_class=2000,
                                       prediction_depth=3,
                                       enzyme_classes=['3.5', '4'])
    model = MemmapsDisjointClassifier(n_classes=5, network=basic_convnet, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=1000)


def train_enz_from_grids():
    data_feeder = EnzymesGridFeeder(data_dir=data_dir,
                                    minibatch_size=8,
                                    init_samples_per_class=2000,
                                    prediction_depth=3,
                                    enzyme_classes=['3.5', '4'])
    model = GridsDisjointClassifier(n_classes=5, network=basic_convnet, grid_size=64, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.train(epochs=1000)


def test_enz_from_grids():
    data_feeder = EnzymesGridFeeder(data_dir=data_dir,
                                    minibatch_size=8,
                                    init_samples_per_class=2000,
                                    prediction_depth=3,
                                    enzyme_classes=['3.5', '4'])
    model = GridsDisjointClassifier(n_classes=5, network=basic_convnet, grid_size=64, minibatch_size=8)
    trainer = ModelTrainer(model=model, data_feeder=data_feeder)
    trainer.monitor.load_model(model_name="params_54ep_meanvalacc[ 0.90322578  0.88306451].npz",
                               network=model.get_output_layers())
    trainer.test()


if __name__ == "__main__":
    # train_enz_from_memmaps()
    train_enz_from_grids()
    # visualize()
