import argparse

from pyhocon import ConfigFactory

from src.training import Trainer
from src.factories import get_dataset, get_model


def load_config(config_path: str) -> dict:
    """
    Loads a HOCON configuration file.
    """
    config = ConfigFactory.parse_file(config_path)
    return config.as_plain_ordered_dict()


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
