import sys
import os

sys.setrecursionlimit(10000)

from protfun.models import train_enz_from_grids, test_enz_from_grids, get_best_params
from protfun.config import get_config
from protfun.visualizer.experiment_visualizer import create_history_plots, \
    create_performance_plots, save_hidden_activations


def describe_model():
    # current experimental constraints
    available_choices = (1, 2)
    available_class_sizes = (2, 166)
    available_grid_sizes = (64, 128)
    available_splits = ('naive', 'strict')
    available_channel_counts = (1, 24)

    print("\n\tHi there! You are about to launch a 3D protein function prediction experiment on "
          "the EC enzymes database. Please, tell me something about your preferences.\n")
    choice = 'dummy'
    while not choice in available_choices:
        choice = int(
            raw_input("\tWould you like to: (type in 1 or 2)\n"
                      "\t1. reproduce one of our existing experiments\n"
                      "\t2. run with your own unique configuration\n"))
    if choice == 2:
        config_path = 'dummy'
        while not os.path.exists(config_path):
            config_path = str(raw_input("\tPlease provide the absolute path to the config.yaml "
                                        "file which describes your desired experiment:\n"))
    else:
        print("\tWe will now use one of the existing experiment configurations. We still need some "
              "details from you to select the right one.\n")
        class_num = 0
        while not class_num in available_class_sizes:
            class_num = int(
                raw_input("\tWould you like to discriminate between 2 classes (3.4.21 vs 3.4.24) "
                          "or 166 classes (whole EC database): "))
        grid_size = 0
        while not grid_size in available_grid_sizes:
            grid_size = int(raw_input("\tWould you like to use a grid size of 64 or 128 voxels: "))
        split_strategy = 'dummy'
        while not split_strategy in available_splits:
            split_strategy = str(raw_input("\tWould you like to use a naive or a strict split: "))
        channel_count = 0
        while not channel_count in available_channel_counts:
            channel_count = int(raw_input("\tWould you like to use 1 or 24 input channels: "))

        config_name = '_'.join(['config', str(class_num) + 'class', str(grid_size),
                                split_strategy, str(channel_count) + 'channel.yaml'])
        config_path = os.path.join(os.path.dirname(__file__), 'experiments', config_name)

    return config_path


def run_experiment(visualize_results=True):
    config_filepath = describe_model()
    config = get_config(config_filepath)
    model_id = train_enz_from_grids(config,
                                    force_download=True,
                                    force_memmaps=True,
                                    force_grids=True,
                                    force_split=True)

    best_params_file = get_best_params(config, model_id)
    test_enz_from_grids(config, model_id, best_params_file, mode='test')

    if visualize_results:
        create_history_plots(config, model_id)
        create_performance_plots(config, model_id, n_classes=config["proteins"]["n_classes"])
        save_hidden_activations(config, model_id)


if __name__ == "__main__":
    run_experiment()
