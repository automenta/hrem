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


def load_config(config_path: str) -> dict:
    """
    Loads a JSON configuration file, handling inheritance from a default config.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if "extends" in config:
        base_config_path = os.path.join(os.path.dirname(config_path), config["extends"])
        base_config = load_config(base_config_path)
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
