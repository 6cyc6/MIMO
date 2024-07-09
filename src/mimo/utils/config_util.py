import os
import yaml
import numpy as np
import marshmallow

from pathlib import Path
from copy import deepcopy
from datetime import datetime
from marshmallow_dataclass import class_schema
from dataclasses import field
# from deprecation import deprecated
from typing import TypeVar, Dict, Union, Type, Any
T = TypeVar('T')


def load_dict_from_yaml(filename: Union[str, Path]):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def save_to_yaml(data: Union[list, dict], filename: Union[str, Path], flush: bool = False, default_flow_style: bool = None):
    """
    Args:
        data:
        filename:
        flush:
        default_flow_style: True: the whole structure has a enclosed {}; False: expand the list; None: compact format
    """
    with open(filename, 'w') as outfile:
        yaml.safe_dump(
            data, outfile, encoding='utf-8', allow_unicode=True,
            default_flow_style=default_flow_style, sort_keys=False
        )
        if flush:
            outfile.flush()


def load_dataclass_from_yaml(dataclass_type: Type[T], filename: Union[Path, str], include_unknown: bool = False) -> T:
    filename = Path(filename)
    if not filename.exists():
        raise FileExistsError(f"{filename} does not exist")
    loaded = load_dict_from_yaml(filename)
    return load_dataclass_from_dict(dataclass_type, loaded, include_unknown)


def load_dataclass_from_dict(dataclass_type: Type[T], dic: Dict, include_unknown: bool = False) -> T:
    schema = class_schema(dataclass_type)
    include_schema = marshmallow.INCLUDE if include_unknown else marshmallow.EXCLUDE
    return schema().load(dic, unknown=include_schema)


def dump_data_to_dict(dataclass_type: T, data):
    schema = class_schema(dataclass_type)
    return schema().dump(data)


def dump_data_to_yaml(
        dataclass_type: T,
        data: Any,
        filename: Union[str, Path]
):
    save_to_yaml(dump_data_to_dict(dataclass_type, data), filename, default_flow_style=False)


def load_dataclass(
        dataclass_type: Type[T],
        config: Union[str, Path, Dict, T],
        include_unknown: bool = False
) -> T:
    if config is None:
        return dataclass_type()
    elif isinstance(config, str) or isinstance(config, Path):
        return load_dataclass_from_yaml(dataclass_type, config, include_unknown)
    elif isinstance(config, Dict):
        return load_dataclass_from_dict(dataclass_type, config, include_unknown)
    elif isinstance(config, dataclass_type):
        return config
    else:
        raise TypeError(f"type of config {type(config)} is not supported. supported types include: \n"
                        f"-- 1) path (str, Path) to the configuration yaml.\n"
                        f"-- 2) Dictionary of the configuration. \n"
                        f"-- 3) The configuration instance of dataclass_type.")


def default_field(obj: Any):
    return field(default_factory=lambda: deepcopy(obj))


def get_datetime_string(time_format: str = "%Y%m%d_%H%M%S"):
    return datetime.now().strftime(time_format)
