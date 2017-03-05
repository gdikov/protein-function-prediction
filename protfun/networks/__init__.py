from protfun.networks.standard_network import standard_network
from protfun.networks.dense_net import dense_network
from protfun.networks.small_dense_net import small_dense_network
from protfun.networks.resnet import resnet
from protfun.networks.shallow_net import shallow_network
from protfun.networks.regularized_net import regularized_net
from protfun.networks.heavy_regularized_net import heavy_regularized_net
from protfun.networks.l2_network import l2_network

networks = {
    "standard_network": standard_network,
    "dense_network": dense_network,
    "small_dense_network": small_dense_network,
    "resnet": resnet,
    "shallow_network": shallow_network,
    "regularized_network": regularized_net,
    "heavy_regularized_network": heavy_regularized_net,
    "l2_network": l2_network
}


def get_network(network_name):
    return networks[network_name]
