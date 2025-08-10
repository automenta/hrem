from src.models import MLP, HRM, HREM, LSTM, RNN, Transformer
from src.datasets import (
    ReverseDataset,
    TinyShakespeareDataset,
    CopyTaskDataset,
    AssociativeRecallDataset,
    GymnasiumDataset,
)

# Registry for models
MODEL_REGISTRY = {
    "mlp": MLP,
    "hrm": HRM,
    "hrem": HREM,
    "lstm": LSTM,
    "rnn": RNN,
    "transformer": Transformer,
}

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
    dataset_params = config["dataset"].get("params", {})
    dataset_class = DATASET_REGISTRY.get(dataset_name)

    if not dataset_class:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Dataset-specific instantiation logic
    if dataset_name == "reverse":
        train_ds = dataset_class(
            size=dataset_params["train_size"], seq_len=dataset_params["seq_len"]
        )
        test_ds = dataset_class(
            size=dataset_params["test_size"], seq_len=dataset_params["seq_len"]
        )
    elif dataset_name == "tiny_shakespeare":
        train_ds = dataset_class(
            seq_length=dataset_params["seq_length"], split="train"
        )
        test_ds = dataset_class(seq_length=dataset_params["seq_length"], split="test")
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
    elif dataset_name == "gymnasium":
        # Gymnasium environments are usually not split into train/test
        train_ds = dataset_class(env_name=dataset_params["env_name"])
        test_ds = dataset_class(env_name=dataset_params["env_name"])
    else:
        # Fallback for simple datasets
        train_ds = dataset_class(**dataset_params)
        test_ds = dataset_class(**dataset_params)

    # Get model config updates from the dataset
    model_name = config["model"]["name"]
    model_config_updates = train_ds.get_model_config_updates(model_name)
    config["model"]["params"].update(model_config_updates)

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

    # The logic for calculating input/output sizes is now handled by the
    # dataset classes, so the factory can be much simpler.
    if model_name in ["hrm", "hrem"]:
        model = model_class(config_dict=model_params)
    else:
        model = model_class(**model_params)

    return model
