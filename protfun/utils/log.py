import logging
from colorlog import ColoredFormatter


def setup_logger(module_name):
    """
    Return a logger with a default ColoredFormatter for the given module name.
    :param module_name: name of the module where the logger will be used
    :return a logger object
    """
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(name)s: %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger(module_name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger