from protfun.networks.basic_convnet import basic_convnet
from protfun.networks.single_trunk_network import single_trunk_network
from protfun.networks.dense_net import dense_network
from protfun.networks.resnet import resnet
from protfun.networks.shallow_net import shallow_network

networks = {
    "basic_convnet": basic_convnet,
    "single_trunk_network": single_trunk_network,
    "dense_network": dense_network,
    "resnet": resnet,
    "shallow_network": shallow_network,
}


def get_network(network_name):
    return networks[network_name]
