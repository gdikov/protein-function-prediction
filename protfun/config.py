import yaml
from yaml import load


def get_config(filename):
    with open(filename, 'r') as stream:
        content = load(stream)
    return content


def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
