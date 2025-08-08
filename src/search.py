import optuna
import torch
from src.models import MLP # Assuming a factory or dict might be better later
from src.datasets import ReverseDataset
from src.training import Trainer

def objective(trial):
    """
    The objective function for Optuna to optimize.
    A trial consists of:
    1. Suggesting a set of hyperparameters.
    2. Building and training a model with them.
    3. Returning the performance metric.
    """
    # 1. Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)

    # For this example, we'll hardcode the model and dataset
    # but this could be parameterized in a more advanced setup.
    seq_len = 64

    # 2. Create the configuration for this trial
    config = {
        "experiment_name": f"hparam_search_trial_{trial.number}",
        "model": {
            "name": "mlp",
            "params": {
                "input_size": seq_len,
                "hidden_size": hidden_size,
                "output_size": seq_len,
            },
        },
        "dataset": {
            "name": "reverse",
            "params": {"seq_len": seq_len, "train_size": 1000, "test_size": 200},
        },
        "training": {
            "epochs": 10,  # Use fewer epochs for a faster search
            "batch_size": 32,
            "learning_rate": lr,
            "loss": "bce",
        },
    }

    # 3. Setup and run the training
    try:
        model = MLP(**config["model"]["params"])
        train_ds = ReverseDataset(
            config["dataset"]["params"]["train_size"],
            config["dataset"]["params"]["seq_len"],
        )
        test_ds = ReverseDataset(
            config["dataset"]["params"]["test_size"],
            config["dataset"]["params"]["seq_len"],
        )

        trainer = Trainer(model, train_ds, test_ds, config)
        results = trainer.run()

        # Return the metric to optimize
        return results['test_acc']

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a value indicating failure, e.g., 0.0 or raise optuna.TrialPruned()
        return 0.0


if __name__ == "__main__":
    # Create a study object and specify the direction is to maximize the metric.
    study = optuna.create_study(direction="maximize")

    # Start the optimization. Optuna will call the objective function n_trials times.
    study.optimize(objective, n_trials=20)

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Test Accuracy): {trial.value:.4f}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
