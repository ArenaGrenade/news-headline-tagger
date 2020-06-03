import json
from dotmap import DotMap
import os
import time


def getConfigFromJSON(json_file):
    """
    Parse the configuration form a json file.
    :param json_file: path to the json configuration file
    :return: config(namespace) and config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using DotMap
    config = DotMap(config_dict)

    return config, config_dict


def processConfig(json_file):
    """
    this function is in fact a wrapper over get_config_from_json. It also logs to the tensor-board and checkpoint.
    :param json_file: a path to the json config file
    :return: configuration as a namespace
    """
    config, _ = getConfigFromJSON(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments",
                                                        time.strftime("%Y-%m-%d/", time.localtime()),
                                                        config.exp.name,
                                                        "logs/"
                                                        )
    config.callbacks.checkpoint_dir = os.path.join("experiments",
                                                   time.strftime("%Y-%m-%d/", time.localtime()),
                                                   config.exp.name,
                                                   "checkpoints/"
                                                   )
    return config
