
import sys
import os
sys.setrecursionlimit(10000)

from protfun.models import train_enz_from_grids
from protfun.config import get_config

def describe_model():
    # current experimental constraints
    available_class_sizes = (2, 166)
    available_grid_sizes = (64, 128)
    available_splits = ('naive', 'strict')
    available_channel_counts = (1, 24)
    available_networks = ('dense_net', 'heavy_regularized_net', 'l2_network', 'regularized_net', 'resnet',
                          'shallow_net', 'small_dense_net', 'standard_network')

    print("Hi there! You are about to launch an experiment the we worked on. "
          "Please, tell me something about your preferences.\n")
    class_num = 0
    while not class_num in available_class_sizes:
        class_num = int(raw_input("\tDo you like to discriminate between 2 or 166 classes:\n"))
    grid_size = 0
    while not grid_size in available_grid_sizes:
        grid_size = int(raw_input("\tDo you like to use a grid size of 64 128 voxels:\n"))
    split_strategy = 'dummy'
    while not split_strategy in available_splits:
        split_strategy = str(raw_input("\tDo you like to use a naive or strict split:\n"))
    channel_count = 0
    while not channel_count in available_channel_counts:
        channel_count = int(raw_input("\tDo you like to use 1 or 24 channels:\n"))
    network_type = 'dummy'
    while not network_type in available_networks:
        network_type = str(raw_input("\tDo you like to use dense_net, heavy_regularized_net, "
                                     "l2_network, regularized_net, resnet, "
                                     "shallow_net, small_dense_net or standard_network:\n"))

    config_name = '_'.joint(['config', str(class_num) + 'class', str(grid_size),
                             split_strategy, str(channel_count) + 'channel.yaml'])

    return config_name, network_type


def run_experiment_from_config(config_filename, network_type):
    config_filepath = os.path.join(os.path.dirname(__file__), config_filename)
    config = get_config(config_filepath)
    train_enz_from_grids(config, model_name=network_type)


def run_experiment():
    config_filename, network_type = describe_model()
    run_experiment_from_config(config_filename, network_type)


if __name__ == "__main__":
    run_experiment()