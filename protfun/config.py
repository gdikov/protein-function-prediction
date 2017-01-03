from yaml import load


def get_config(filename):
    with open(filename, 'r') as stream:
        content = load(stream)
    return content
