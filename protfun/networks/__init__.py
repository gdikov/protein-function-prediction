from protfun.networks.single_trunk_network import single_trunk_network
from protfun.networks.dense_net import dense_network
from protfun.networks.resnet import resnet
from protfun.networks.shallow_net import shallow_network
from protfun.networks.regularized_net import regularized_net

networks = {
    "single_trunk_network": single_trunk_network,
    "dense_network": dense_network,
    "resnet": resnet,
    "shallow_network": shallow_network,
    "regularized_network": regularized_net
}


def get_network(network_name):
    return networks[network_name]
