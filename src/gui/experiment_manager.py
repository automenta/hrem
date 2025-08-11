import json
import os
import shutil
from datetime import datetime

from .constants import (
    ARCHIVE_DIR,
    RESULTS_DIR,
    STATUS_COMPLETED,
    STATUS_RUNNING,
)
from .process_manager import BaseProcessManager


class ExperimentManager(BaseProcessManager):
    """
    Handles the logic for launching, monitoring, and loading experiment results.
    Inherits process management from BaseProcessManager.
    """

    def __init__(self):
        super().__init__()
        self.RESULTS_DIR = RESULTS_DIR

    def launch_experiment(self, config_path: str):
        """
        Launches an experiment in a new process.
        """
        return self.launch_process(config_path, "main.py")

    def stop_experiment(self, exp_name: str) -> bool:
        """
        Stops a running experiment process.
        """
        return self.stop_process(exp_name)

    def get_experiment_statuses(self) -> dict:
        """
        Checks the status of all experiments, including those on disk.
        """
        # Get statuses of running/finished processes from the parent
        statuses = super().get_statuses()

        # Augment with experiments that are on disk but not tracked as processes
        if os.path.exists(RESULTS_DIR):
            for exp_name in os.listdir(RESULTS_DIR):
                if (
                    os.path.isdir(os.path.join(RESULTS_DIR, exp_name))
                    and exp_name not in statuses
                ):
                    statuses[exp_name] = STATUS_COMPLETED

        return statuses

    def load_experiment_results(self, exp_name, base_dir=None):
        """
        Loads results for an experiment from a given base directory.
        Returns a tuple: (data, error_message).
        """
        if base_dir is None:
            base_dir = self.RESULTS_DIR

        live_file = os.path.join(base_dir, exp_name, "live.json")
        results_file = os.path.join(base_dir, exp_name, "results.json")

        def _load_json(path):
            try:
                with open(path, "r") as f:
                    return json.load(f), None
            except json.JSONDecodeError:
                return None, f"Error: Corrupted JSON file at {path}"
            except IOError:
                return None, f"Error: Could not read file at {path}"

        if os.path.exists(live_file):
            data, err = _load_json(live_file)
            if err:
                return None, err
            # Live file has a different structure
            return data.get("full_results", {}), None

        if os.path.exists(results_file):
            return _load_json(results_file)

        return None, None  # No results found, but not an error

    def load_experiment_config(self, exp_name, base_dir=None):
        """
        Loads the config for a given experiment from a given base directory.
        Returns a tuple: (config_data, error_message).
        """
        if base_dir is None:
            base_dir = self.RESULTS_DIR
        config_file = os.path.join(base_dir, exp_name, "config.json")
        if not os.path.exists(config_file):
            return None, "Config file not found."

        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
                return config_data, None
        except (json.JSONDecodeError, IOError) as e:
            return None, f"Error reading config: {e}"

    def get_experiments_data(self) -> list:
        """
        Gathers comprehensive data for all experiments.
        """
        experiments_data = []
        statuses = self.get_experiment_statuses()

        for exp_name, status in statuses.items():
            config, _ = self.load_experiment_config(exp_name)
            results, _ = self.load_experiment_results(exp_name)

            # --- Extract data with defaults ---
            model_name = "N/A"
            dataset_name = "N/A"
            learning_rate = "N/A"
            if config:
                model_name = config.get("model", {}).get("name", "N/A")
                dataset_name = config.get("dataset", {}).get("name", "N/A")
                learning_rate = config.get("training", {}).get("learning_rate", "N/A")

            final_loss = "N/A"
            if results and "test_loss" in results and results["test_loss"]:
                final_loss = f"{results['test_loss'][-1]:.4f}"

            creation_time = "N/A"
            try:
                exp_path = os.path.join(RESULTS_DIR, exp_name)
                timestamp = os.path.getctime(exp_path)
                creation_time = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except FileNotFoundError:
                pass # Exp might not have a directory yet

            experiments_data.append(
                {
                    "name": exp_name,
                    "status": status,
                    "model": model_name,
                    "dataset": dataset_name,
                    "lr": learning_rate,
                    "final_loss": final_loss,
                    "created": creation_time,
                }
            )

        return experiments_data

    def archive_experiment(self, exp_name: str) -> (bool, str):
        """
        Moves an experiment's directory to the archive folder.
        Returns a tuple (success, message).
        """
        source_path = os.path.join(RESULTS_DIR, exp_name)
        dest_path = os.path.join(ARCHIVE_DIR, exp_name)

        if not os.path.exists(source_path):
            return False, f"Error: Experiment directory not found at {source_path}"

        try:
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            shutil.move(source_path, dest_path)
            return True, f"Experiment {exp_name} archived successfully."
        except Exception as e:
            return False, f"Error archiving experiment: {e}"

    def get_archived_experiments_data(self) -> list:
        """
        Gathers comprehensive data for all archived experiments.
        """
        archived_experiments = []
        if not os.path.exists(ARCHIVE_DIR):
            return archived_experiments

        for exp_name in os.listdir(ARCHIVE_DIR):
            exp_path = os.path.join(ARCHIVE_DIR, exp_name)
            if os.path.isdir(exp_path):
                config, _ = self.load_experiment_config(exp_name, base_dir=ARCHIVE_DIR)

                model_name = config.get("model", {}).get("name", "N/A") if config else "N/A"
                dataset_name = config.get("dataset", {}).get("name", "N/A") if config else "N/A"

                creation_time = "N/A"
                try:
                    timestamp = os.path.getctime(exp_path)
                    creation_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                except FileNotFoundError:
                    pass

                archived_experiments.append({
                    "name": exp_name,
                    "model": model_name,
                    "dataset": dataset_name,
                    "created": creation_time,
                })
        return archived_experiments

    def restore_experiment(self, exp_name: str) -> (bool, str):
        """
        Moves an experiment's directory from the archive back to the main results folder.
        """
        source_path = os.path.join(ARCHIVE_DIR, exp_name)
        dest_path = os.path.join(RESULTS_DIR, exp_name)

        if not os.path.exists(source_path):
            return False, f"Error: Archived experiment not found at {source_path}"

        try:
            shutil.move(source_path, dest_path)
            return True, f"Experiment {exp_name} restored successfully."
        except Exception as e:
            return False, f"Error restoring experiment: {e}"

    def delete_experiment_permanently(self, exp_name: str) -> (bool, str):
        """
        Permanently deletes an experiment's directory from the archive.
        """
        exp_path = os.path.join(ARCHIVE_DIR, exp_name)
        if not os.path.exists(exp_path):
            return False, f"Error: Archived experiment not found at {exp_path}"

        try:
            shutil.rmtree(exp_path)
            return True, f"Experiment {exp_name} permanently deleted."
        except Exception as e:
            return False, f"Error deleting experiment: {e}"

    def rename_experiment(self, old_name: str, new_name: str) -> (bool, str):
        """
        Renames an experiment's directory.
        """
        if not old_name or not new_name:
            return False, "Error: Experiment names cannot be empty."
        if old_name == new_name:
            return False, "Error: New name is the same as the old name."

        statuses = self.get_experiment_statuses()
        if statuses.get(old_name) == STATUS_RUNNING:
            return False, f"Error: Cannot rename a running experiment: {old_name}"

        old_path = os.path.join(self.RESULTS_DIR, old_name)
        new_path = os.path.join(self.RESULTS_DIR, new_name)

        if not os.path.exists(old_path):
            return False, f"Error: Experiment to rename not found at {old_path}"
        if os.path.exists(new_path):
            return False, f"Error: An experiment with the name {new_name} already exists."

        try:
            os.rename(old_path, new_path)
            return True, f"Experiment '{old_name}' renamed to '{new_name}' successfully."
        except Exception as e:
            return False, f"Error renaming experiment: {e}"

    def clone_experiment(self, original_name: str, new_name: str) -> (bool, str):
        """
        Clones an experiment by copying its config to a new experiment directory.
        """
        if not original_name or not new_name:
            return False, "Error: Experiment names cannot be empty."
        if original_name == new_name:
            return False, "Error: New name is the same as the original name."

        original_path = os.path.join(self.RESULTS_DIR, original_name)
        new_path = os.path.join(self.RESULTS_DIR, new_name)

        if not os.path.exists(original_path):
            return False, f"Error: Experiment to clone not found at {original_path}"
        if os.path.exists(new_path):
            return False, f"Error: An experiment with the name {new_name} already exists."

        original_config_path = os.path.join(original_path, "config.json")
        if not os.path.exists(original_config_path):
            return False, f"Error: Config file not found for experiment {original_name}"

        try:
            os.makedirs(new_path)
            new_config_path = os.path.join(new_path, "config.json")
            shutil.copy(original_config_path, new_config_path)
            # Also update the experiment name within the new config
            config, err = self.load_experiment_config(new_name)
            if config and not err:
                config["experiment_name"] = new_name
                with open(new_config_path, "w") as f:
                    json.dump(config, f, indent=4)

            return True, f"Experiment '{original_name}' cloned to '{new_name}' successfully."
        except Exception as e:
            return False, f"Error cloning experiment: {e}"
