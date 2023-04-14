import yaml
import argparse
from typing import Dict

import torch
import torch.nn as nn
import importlib

def process_value(value):
    if isinstance(value, str) and value.startswith("torch."):
        return import_class(f"{value}")
    return value

def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_yaml(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
        if "class_path" in config:
            config["class"] = import_class(config["class_path"])
        
        # Process the 'init_args' dictionary
        init_args = config.get("init_args", {})
        for key, value in init_args.items():
            init_args[key] = process_value(value)

        return config

def parse_configs(args) -> Dict[str, Dict]:
    configs = {}

    configs["general"] = parse_yaml(args.general)
    configs["model"] = parse_yaml(args.model)
    configs["data"] = parse_yaml(args.data)
    configs["strategy"] = parse_yaml(args.strategy)

    return configs


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--general", type=str, required=True, help="Path to the general config file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model config file."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the data config file."
    )
    parser.add_argument(
        "--strategy", type=str, required=True, help="Path to the strategy config file."
    )

    return parser.parse_args()


# if __name__ == "__main__":
#     args = get_args()
#     configs = parse_configs(args)

#     model_class = configs["model"]["class"]
#     init_args = configs["model"]["init_args"]
#     model = model_class(**init_args)
#     print(model)
