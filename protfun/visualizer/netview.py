class NetworkView(object):
    """
    Visualize the neural network layers in a graph.
    """

    def __init__(self, data_dir):
        """
        :param data_dir: data directory in which the network graph will be
        saved.
        """
        self.data_dir = data_dir

    def save_network_graph(self, network, filename):
        """
        Creates and saves the network graph image.
        :param network: the last lasagne layer(s) of a network
        :param filename: name of the image file to be saved in data_dir
        """
        import os
        import lasagne
        import matplotlib

        matplotlib.use('Agg')
        from nolearn.lasagne.visualize import draw_to_file

        file_path = os.path.join(self.data_dir, filename)
        layers = lasagne.layers.get_all_layers(network)
        draw_to_file(layers, file_path)
