import argparse
import json
import os

from src.training import Trainer
from src.factories import get_dataset, get_model


def deep_merge(dict1, dict2):
    """
    Recursively merges two dictionaries. dict2 values override dict1 values.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, loaded_paths=None) -> dict:
    """
    Loads a JSON configuration file, handling inheritance from other configs.
    """
    if loaded_paths is None:
        loaded_paths = set()

    config_path = os.path.abspath(config_path)
    if config_path in loaded_paths:
        return {}  # Avoid circular dependencies
    loaded_paths.add(config_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    if "extends" in config:
        extends_paths = config["extends"]
        if isinstance(extends_paths, str):
            extends_paths = [extends_paths]

        base_config = {}
        for extends_path in extends_paths:
            full_extends_path = os.path.join(os.path.dirname(config_path), extends_path)
            extended_config = load_config(full_extends_path, loaded_paths)
            base_config = deep_merge(base_config, extended_config)

        config = deep_merge(base_config, config)
        del config["extends"]

    return config


def main(config_path):
    """
    Main function to run an experiment.
    """
    # 1. Load configuration
    config = load_config(config_path)

    # 2. Select and instantiate dataset
    train_ds, test_ds = get_dataset(config)

    # 3. Select and instantiate model
    model = get_model(config)

    # 4. Instantiate and run the trainer
    trainer = Trainer(model, train_ds, test_ds, config)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training experiment.")
    parser.add_argument("config", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    main(args.config)
