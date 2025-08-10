import json
import os
import subprocess
import sys

# --- Constants ---
RESULTS_DIR = "results"

# Experiment Statuses
STATUS_RUNNING = "Running"
STATUS_COMPLETED = "Completed"
STATUS_FAILED = "Failed"
STATUS_UNKNOWN = "Unknown"


class ExperimentManager:
    """
    Handles the logic for launching, monitoring, and loading experiment results.
    """

    def __init__(self):
        self.processes = {}  # Tracks running subprocesses: {exp_name: Popen_obj}

    def launch_experiment(self, config_path):
        """
        Launches an experiment in a new process and tracks it.
        """
        # Extract a unique name for the experiment from the config path
        base_name = os.path.basename(config_path)
        exp_name = base_name.replace(".json", "")

        process = subprocess.Popen(
            [sys.executable, "main.py", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes[exp_name] = process

    def stop_experiment(self, exp_name):
        """
        Stops a running experiment process.
        """
        if exp_name in self.processes:
            self.processes[exp_name].terminate()
            return True
        return False

    def get_experiment_statuses(self):
        """
        Checks the status of all experiments and returns a dictionary.
        """
        statuses = {}

        # First, get all experiment directories on disk
        if os.path.exists(RESULTS_DIR):
            for exp_name in os.listdir(RESULTS_DIR):
                if os.path.isdir(os.path.join(RESULTS_DIR, exp_name)):
                    # Default status for experiments on disk is Completed
                    statuses[exp_name] = STATUS_COMPLETED

        # Now, check the tracked processes for more accurate statuses
        finished_processes = []
        for exp_name, process in self.processes.items():
            return_code = process.poll()
            if return_code is None:
                statuses[exp_name] = STATUS_RUNNING
            else:
                finished_processes.append(exp_name)
                if return_code == 0:
                    statuses[exp_name] = STATUS_COMPLETED
                else:
                    statuses[exp_name] = STATUS_FAILED

        # Clean up finished processes from the tracking dict
        for exp_name in finished_processes:
            del self.processes[exp_name]

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
