import json
import os
import shutil
import uuid
from datetime import datetime

from pyhocon import ConfigFactory, ConfigTree

from .constants import (
    ARCHIVE_DIR,
    BASE_MODELS_DIR,
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

    def stop_experiment(self, exp_name: str, force: bool = False) -> bool:
        """
        Stops a running experiment process.

        Args:
            exp_name (str): The name of the experiment to stop.
            force (bool): If True, forces the process to kill.

        Returns:
            bool: True if the signal was sent, False otherwise.
        """
        return self.stop_process(exp_name, force=force)

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
            parent = "N/A"
            num_params = "N/A"
            epoch_time = "N/A"
            race_id = "N/A"
            is_baseline_for = None
            exp_type = "Single"
            if config:
                if "search" in config:
                    exp_type = "Search"
                elif config.get("is_baseline_for") or config.get("race_id"):
                    # This is a bit simplistic, might need refinement
                    exp_type = "Race"

                model_name = config.get("model", {}).get("name", "N/A")
                dataset_name = config.get("dataset", {}).get("name", "N/A")
                learning_rate = config.get("training", {}).get("learning_rate", "N/A")
                parent = config.get("parent_experiment", "N/A")
                num_params = config.get("model_num_parameters", "N/A")
                epoch_time = config.get("avg_epoch_time_s", "N/A")
                race_id = config.get("race_id", "N/A")
                is_baseline_for = config.get("is_baseline_for")

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
                pass  # Exp might not have a directory yet

            experiments_data.append(
                {
                    "name": exp_name,
                    "status": status,
                    "model": model_name,
                    "dataset": dataset_name,
                    "lr": learning_rate,
                    "final_loss": final_loss,
                    "created": creation_time,
                    "parent": parent,
                    "params": num_params,
                    "epoch_time": epoch_time,
                    "race_id": race_id,
                    "is_baseline_for": is_baseline_for,
                    "type": exp_type,
                }
            )

        return experiments_data

    def get_completed_races(self) -> list:
        """
        Gathers data for completed races, determines winners, and returns a summary.
        """
        all_experiments = self.get_experiments_data()
        races = {}
        for exp in all_experiments:
            race_id = exp.get("race_id")
            if race_id and race_id != "N/A":
                if race_id not in races:
                    races[race_id] = {
                        "participants": [],
                        "is_complete": True,
                        "completed_at": "N/A",
                        "race_id": race_id,
                    }
                races[race_id]["participants"].append(exp)
                if exp["status"] == STATUS_RUNNING:
                    races[race_id]["is_complete"] = False

        completed_races = []
        for race_id, race_data in races.items():
            if not race_data["is_complete"]:
                continue

            winner_name = "N/A"
            lowest_loss = float("inf")
            participant_names = []
            latest_date = None

            for p in race_data["participants"]:
                participant_names.append(p["name"])
                try:
                    p_loss = float(p["final_loss"])
                    if p_loss < lowest_loss:
                        lowest_loss = p_loss
                        winner_name = p["name"]
                except (ValueError, TypeError):
                    continue # Ignore if loss is not a valid float

                try:
                    p_date = datetime.strptime(p["created"], "%Y-%m-%d %H:%M")
                    if latest_date is None or p_date > latest_date:
                        latest_date = p_date
                except (ValueError, TypeError):
                    continue

            race_data["winner"] = winner_name
            race_data["participants"] = participant_names
            if latest_date:
                race_data["completed_at"] = latest_date.strftime("%Y-%m-%d %H:%M")

            completed_races.append(race_data)

        return completed_races


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

                model_name = (
                    config.get("model", {}).get("name", "N/A") if config else "N/A"
                )
                dataset_name = (
                    config.get("dataset", {}).get("name", "N/A") if config else "N/A"
                )

                creation_time = "N/A"
                try:
                    timestamp = os.path.getctime(exp_path)
                    creation_time = datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                except FileNotFoundError:
                    pass

                archived_experiments.append(
                    {
                        "name": exp_name,
                        "model": model_name,
                        "dataset": dataset_name,
                        "created": creation_time,
                    }
                )
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

    def delete_race(self, race_id: str) -> (bool, str):
        """
        Permanently deletes all experiments associated with a given race_id.
        """
        if not race_id:
            return False, "Error: race_id cannot be empty."

        experiments_to_delete = []
        all_experiments = self.get_experiments_data()
        for exp in all_experiments:
            if exp.get("race_id") == race_id:
                experiments_to_delete.append(exp)

        if not experiments_to_delete:
            return False, f"No experiments found for race_id: {race_id}"

        # Check for running experiments before deleting
        for exp in experiments_to_delete:
            if exp["status"] == STATUS_RUNNING:
                return (
                    False,
                    f"Cannot delete race: Experiment '{exp['name']}' is still running.",
                )

        # Proceed with deletion
        deleted_count = 0
        for exp in experiments_to_delete:
            exp_path = os.path.join(self.RESULTS_DIR, exp["name"])
            if os.path.exists(exp_path):
                try:
                    shutil.rmtree(exp_path)
                    deleted_count += 1
                except Exception as e:
                    return (
                        False,
                        f"Error deleting experiment '{exp['name']}': {e}",
                    )

        return True, f"Successfully deleted {deleted_count} experiments for race '{race_id}'."

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
            return (
                False,
                f"Error: An experiment with the name {new_name} already exists.",
            )

        try:
            os.rename(old_path, new_path)
            return (
                True,
                f"Experiment '{old_name}' renamed to '{new_name}' successfully.",
            )
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
            return (
                False,
                f"Error: An experiment with the name {new_name} already exists.",
            )

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
                # --- Add parent experiment tracking ---
                config["parent_experiment"] = original_name
                with open(new_config_path, "w") as f:
                    json.dump(config, f, indent=4)

            return (
                True,
                f"Experiment '{original_name}' cloned to '{new_name}' successfully.",
            )
        except Exception as e:
            return False, f"Error cloning experiment: {e}"

    def get_experiment_graph(self):
        """
        Builds a graph structure of experiments based on parent-child relationships.

        Returns:
            dict: A dictionary containing 'nodes', 'edges', and 'roots'.
                  'nodes': {exp_name: {data}}
                  'edges': [(parent_name, child_name)]
                  'roots': [exp_name]
        """
        experiments = self.get_experiments_data()
        nodes = {exp["name"]: exp for exp in experiments}
        edges = []

        all_children = set()

        for exp_name, exp_data in nodes.items():
            parent = exp_data.get("parent")
            if parent and parent != "N/A" and parent in nodes:
                edges.append((parent, exp_name))
                all_children.add(exp_name)

        roots = [name for name in nodes if name not in all_children]

        return {"nodes": nodes, "edges": edges, "roots": roots}

    def launch_experiment_race(self, launch_info: dict):
        """
        Launches a "race" of experiments: a challenger against multiple baselines.
        """
        race_id = str(uuid.uuid4())[:8]
        base_name = launch_info["base_name"]
        challenger_config_path = launch_info["challenger_config"]
        baseline_models = launch_info["baselines"]
        notes = launch_info.get("notes")

        # 1. Load challenger config
        try:
            challenger_config = ConfigFactory.parse_file(challenger_config_path)
        except Exception as e:
            return False, f"Failed to load challenger config: {e}"

        # 2. Prepare and launch challenger experiment
        challenger_exp_name = f"{base_name}_challenger"
        challenger_config["experiment_name"] = challenger_exp_name
        challenger_config["race_id"] = race_id
        if notes:
            challenger_config["notes"] = notes
        self._prepare_and_launch_exp(challenger_exp_name, challenger_config, "main.py")

        # 3. Prepare and launch baseline experiments
        for baseline_model_name in baseline_models:
            baseline_exp_name = f"{base_name}_baseline_{baseline_model_name}"
            try:
                # Create a new config for the baseline
                baseline_config = ConfigTree()
                baseline_config.put("training", challenger_config.get("training"))
                baseline_config.put("dataset", challenger_config.get("dataset"))

                # Load the base model config
                base_model_config_path = os.path.join(
                    BASE_MODELS_DIR, f"{baseline_model_name}.json"
                )
                base_model_config = ConfigFactory.parse_file(base_model_config_path)
                baseline_config.put("model", base_model_config)

                # Add metadata
                baseline_config.put("experiment_name", baseline_exp_name)
                baseline_config.put("race_id", race_id)
                baseline_config.put("is_baseline_for", challenger_exp_name)

                self._prepare_and_launch_exp(
                    baseline_exp_name, baseline_config, "main.py"
                )

            except Exception as e:
                print(f"Failed to create/launch baseline {baseline_model_name}: {e}")
                continue  # Continue to the next baseline

        return True, f"Experiment race '{base_name}' launched successfully."

    def _prepare_and_launch_exp(self, exp_name: str, config: dict, script_to_run: str):
        """
        Helper to create an experiment directory, save the config, and launch it.
        """
        exp_dir = os.path.join(self.RESULTS_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        config_path = os.path.join(exp_dir, "config.json")

        # Ensure config is a plain dict for JSON serialization
        if isinstance(config, ConfigTree):
            config_dict = config.as_plain_ordered_dict()
        else:
            config_dict = config

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

        return self.launch_process(config_path, script_to_run)

    def launch_experiment_from_config(self, config: dict, exp_name: str) -> (bool, str):
        """
        Launches a single experiment directly from a config dictionary.
        """
        if not exp_name:
            return False, "Experiment name cannot be empty."

        exp_dir = os.path.join(self.RESULTS_DIR, exp_name)
        if os.path.exists(exp_dir):
            return False, f"Experiment '{exp_name}' already exists."

        try:
            self._prepare_and_launch_exp(exp_name, config, "main.py")
            return True, f"Successfully launched {exp_name}."
        except Exception as e:
            return False, f"Failed to launch {exp_name}: {e}"

    def launch_hyperparameter_search(self, config: dict, exp_name: str) -> (bool, str):
        """
        Launches a hyperparameter search directly from a config dictionary.
        """
        if not exp_name:
            return False, "Experiment name cannot be empty."

        exp_dir = os.path.join(self.RESULTS_DIR, exp_name)
        if os.path.exists(exp_dir):
            return False, f"Search '{exp_name}' already exists."

        try:
            self._prepare_and_launch_exp(exp_name, config, "search.py")
            return True, f"Successfully launched search {exp_name}."
        except Exception as e:
            return False, f"Failed to launch search {exp_name}: {e}"

    def get_plottable_metrics(self):
        """
        Returns a list of metrics that can be used for plotting in the analysis view.
        This includes a mix of flattened config keys and result metrics.
        """
        # This can be expanded or made dynamic in the future
        return sorted(
            [
                "results.final_loss",
                "results.params",
                "results.epoch_time",
                "config.training.learning_rate",
                "config.training.batch_size",
                "config.model.params.hidden_dim",
                "config.model.params.n_layers",
            ]
        )

    def get_log_contents(self, exp_name: str) -> str:
        """
        Retrieves the log content for a given experiment.
        """
        return self.read_log_file(exp_name)

    def get_log_path(self, exp_name: str) -> str:
        """
        Returns the path to the log file for a given experiment.
        """
        return os.path.join(self.RESULTS_DIR, exp_name, "output.log")
