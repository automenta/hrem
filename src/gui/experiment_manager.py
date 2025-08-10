import json
import os

from .constants import STATUS_COMPLETED, STATUS_RUNNING
from .process_manager import BaseProcessManager

# --- Constants ---
RESULTS_DIR = "results"


class ExperimentManager(BaseProcessManager):
    """
    Handles the logic for launching, monitoring, and loading experiment results.
    Inherits process management from BaseProcessManager.
    """

    def __init__(self):
        super().__init__()

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

    def load_experiment_results(self, exp_name):
        """
        Loads results for an experiment.
        Returns a tuple: (data, error_message).
        """
        live_file = os.path.join(RESULTS_DIR, exp_name, "live.json")
        results_file = os.path.join(RESULTS_DIR, exp_name, "results.json")

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

    def load_experiment_config(self, exp_name):
        """
        Loads the config for a given experiment.
        Returns a tuple: (config_data, error_message).
        """
        config_file = os.path.join(RESULTS_DIR, exp_name, "config.json")
        if not os.path.exists(config_file):
            return None, "Config file not found."

        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
                return config_data, None
        except (json.JSONDecodeError, IOError) as e:
            return None, f"Error reading config: {e}"
