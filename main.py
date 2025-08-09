import argparse
import json
import torch.nn as nn

from src.models import MLP, HRM, HREM, LSTM
from src.datasets import (
    ReverseDataset,
    TinyShakespeareDataset,
    CopyTaskDataset,
    AssociativeRecallDataset,
    GymnasiumDataset,
)
from src.training import Trainer

# Registry for models
MODEL_REGISTRY = {"mlp": MLP, "hrm": HRM, "hrem": HREM, "lstm": LSTM}

# Registry for datasets
DATASET_REGISTRY = {
    "reverse": ReverseDataset,
    "tiny_shakespeare": TinyShakespeareDataset,
    "copy": CopyTaskDataset,
    "associative_recall": AssociativeRecallDataset,
    "gymnasium": GymnasiumDataset,
}


def get_dataset(config: dict):
    """
    Instantiates the dataset based on the configuration.
    """
    dataset_name = config["dataset"]["name"]
    dataset_params = config["dataset"]["params"]
    dataset_class = DATASET_REGISTRY.get(dataset_name)

    if not dataset_class:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "reverse":
        train_ds = dataset_class(
            dataset_params["train_size"], dataset_params["seq_len"]
        )
        test_ds = dataset_class(
            dataset_params["test_size"], dataset_params["seq_len"]
        )
        # Set the feature dimension for recurrent models, but MLP will need to override this
        config["model"]["params"]["input_size_per_step"] = 1
    elif dataset_name == "tiny_shakespeare":
        train_ds = dataset_class(seq_length=dataset_params["seq_length"], split="train")
        test_ds = dataset_class(seq_length=dataset_params["seq_length"], split="test")
        config["model"]["params"]["vocab_size"] = train_ds.vocab_size
        config["model"]["params"]["output_size"] = train_ds.vocab_size
    elif dataset_name == "copy":
        train_ds = dataset_class(
            size=dataset_params["train_size"],
            seq_len=dataset_params["seq_len"],
            vec_len=dataset_params["vec_len"],
        )
        test_ds = dataset_class(
            size=dataset_params["test_size"],
            seq_len=dataset_params["seq_len"],
            vec_len=dataset_params["vec_len"],
        )
        config["model"]["params"]["input_size"] = dataset_params["vec_len"]
    elif dataset_name == "associative_recall":
        train_ds = dataset_class(
            size=dataset_params["train_size"],
            item_range=tuple(dataset_params["item_range"]),
            vec_len=dataset_params["vec_len"],
        )
        test_ds = dataset_class(
            size=dataset_params["test_size"],
            item_range=tuple(dataset_params["item_range"]),
            vec_len=dataset_params["vec_len"],
        )
        config["model"]["params"]["input_size"] = dataset_params["vec_len"]
    elif dataset_name == "gymnasium":
        train_ds = dataset_class(env_name=dataset_params["env_name"])
        test_ds = dataset_class(env_name=dataset_params["env_name"])
        config["model"]["params"]["input_size"] = train_ds.obs_space.shape[0]
    else:
        raise ValueError(f"Instantiation logic for {dataset_name} not implemented.")

    return train_ds, test_ds


def get_model(config: dict):
    """
    Instantiates the model based on the configuration.
    """
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]
    model_class = MODEL_REGISTRY.get(model_name)

    if not model_class:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "mlp":
        dataset_name = config["dataset"]["name"]
        dataset_params = config["dataset"]["params"]
        if dataset_name == "copy":
            model_params["input_size"] = (
                dataset_params["seq_len"] * 2 + 1
            ) * dataset_params["vec_len"]
            model_params["output_size"] = model_params["input_size"]
        elif dataset_name == "associative_recall":
            model_params["input_size"] = (
                dataset_params["item_range"][1] + 2
            ) * dataset_params["vec_len"]
            model_params["output_size"] = model_params["input_size"]
        elif dataset_name == "reverse":
            model_params["input_size"] = dataset_params["seq_len"]
            model_params["output_size"] = dataset_params["seq_len"]

        model = MLP(**model_params)
    elif model_name in ["hrm", "hrem"]:
        model = model_class(config_dict=model_params)
    elif model_name == "lstm":
        model = model_class(**model_params)
    else:
        model = model_class(**model_params)

    return model


def main(config_path):
    """
    Main function to run an experiment.
    """
    # 1. Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

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
