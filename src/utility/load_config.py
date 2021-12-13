import os
import yaml

CONFIG_PATH = "config/"


def load_config(config_name: str):
    """Loads configuration file.
    """

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config
