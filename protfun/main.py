import sys, os

# os.environ["THEANO_FLAGS"] = "device=gpu0,lib.cnmem=1"
sys.setrecursionlimit(10000)
# enable if you want to profile the forward pass
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from protfun.models import train_enz_from_grids, test_enz_from_grids
from protfun.config import get_config

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = get_config(config_path)
    # train_enz_from_memmaps()
    train_enz_from_grids(config)
    # test_enz_from_grids(model_name="grids_classifier_2_classes_1_13_2017_10-7",
    #                     params_file="params_330ep_meanvalacc0.928571.npz")
    # visualize()
