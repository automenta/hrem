import argparse
import copy
from functools import partial

import optuna
from pyhocon import ConfigFactory

from src.training import Trainer
from src.factories import get_dataset, get_model


def load_config(config_path: str) -> dict:
    """
    Loads a HOCON configuration file.
    """
    config = ConfigFactory.parse_file(config_path)
    return config.as_plain_ordered_dict()


def set_nested_value(d, key_path, value):
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def objective(trial, base_config):
    """
    The objective function for Optuna to optimize.
    """
    # Create a deep copy of the base config for this trial
    trial_config = copy.deepcopy(base_config)

    # 1. Suggest hyperparameters
    search_config = trial_config["search"]["params"]
    for param_path, search_params in search_config.items():
        param_type = search_params["type"]
        args = search_params.get("args", [])
        kwargs = search_params.get("kwargs", {})

        suggest_method = getattr(trial, f"suggest_{param_type}")
        value = suggest_method(param_path, *args, **kwargs)

        set_nested_value(trial_config, param_path, value)

    # Update experiment name for logging
    trial_config["experiment_name"] = (
        f"{base_config['experiment_name']}_trial_{trial.number}"
    )

    # 2. Setup and run the training
    try:
        train_ds, test_ds = get_dataset(trial_config)
        model = get_model(trial_config)
        trainer = Trainer(model, train_ds, test_ds, trial_config)
        final_metrics = trainer.run(trial)

        # Return the metric to optimize
        metric = base_config["search"]["metric"]
        return final_metrics[metric]

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Prune trial if it fails
        raise optuna.exceptions.TrialPruned()


def main(config_path):
    """
    Main function to run an experiment or a search.
    """
    # 1. Load configuration
    config = load_config(config_path)

    # 2. Check if this is a search or a single run
    if "search" in config:
        # This is a hyperparameter search
        search_settings = config.get("search", {})

        if not search_settings:
            print("Error: 'search' section not found in the configuration file.")
            return

        study = optuna.create_study(direction=search_settings.get("direction", "minimize"))
        obj_fn = partial(objective, base_config=config)
        study.optimize(obj_fn, n_trials=search_settings.get("n_trials", 20))

        print("\n--- Hyperparameter Search Complete ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value ({search_settings['metric']}): {trial.value:.4f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    else:
        # This is a single experiment run
        train_ds, test_ds = get_dataset(config)
        model = get_model(config)
        trainer = Trainer(model, train_ds, test_ds, config)
        trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training experiment or a hyperparameter search.")
    parser.add_argument("config", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    main(args.config)
