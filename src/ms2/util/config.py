### ~~~ GLOBAL IMPORTS ~~~ ###
from pathlib import Path
from typing import TypeVar
import yaml


### ~~~ LOCAL IMPORTS ~~~ ###
from .paths import PipelineStep, config_path

### ~~~ STATE MANAGEMENT ~~~ ###
T = TypeVar("T")


def load_pipeline_config(step: PipelineStep) -> dict:
    """
    This function loads a pipeline config from YAML.
    Args:
        step (PipelineStep): The pipeline step enum value.
    Returns:
        dict: The loaded config dictionary.
    """
    ### resolve config path ###
    path: Path = config_path[step]

    ### parse yaml file ###
    with path.open("r", encoding="utf-8") as file:
        loaded: object = yaml.safe_load(file)

    ### normalize empty documents ###
    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a mapping in config file: {path}")

    return loaded


def get_config_value(
    config: dict,
    section: str,
    key: str,
    default: T,
) -> T:
    """
    This function gets a config value with a default fallback.
    Args:
        config (dict): The loaded config dictionary.
        section (str): The config section name.
        key (str): The config key name.
        default (object): The default value.
    Returns:
        object: The config value.
    """
    return config.get(section, {}).get(key, default)
