import os
import lasagne
from nolearn.lasagne.visualize import draw_to_file


class NetworkView(object):
    """
    Visualize network filters and activations
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def save_network_graph(self, network, filename):
        file_path = os.path.join(self.data_dir, filename)
        layers_debug = lasagne.layers.get_all_layers(network)
        draw_to_file(layers_debug, file_path)
