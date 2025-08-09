import json
import os
from collections import defaultdict

class TrainingLogger:
    """
    Handles logging of training and evaluation results.
    """
    def __init__(self, config: dict):
        self.config = config
        self.experiment_name = config['experiment_name']
        self.results_dir = f"results/{self.experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.results = defaultdict(list)
        self._save_config()

    def _save_config(self):
        """Saves the configuration file to the results directory."""
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

    def log(self, metrics: dict, step: int):
        """
        Logs a dictionary of metrics.

        Args:
            metrics (dict): A dictionary of metric names to values.
            step (int): The current step (e.g., epoch number).
        """
        for key, value in metrics.items():
            self.results[key].append(value)
        self.results['step'].append(step)

        # Optionally print to console
        print(f"Step {step}: {', '.join([f'{k} = {v:.4f}' for k, v in metrics.items()])}")

        # --- Live logging for GUI ---
        self._save_live_results(metrics, step)

    def _save_live_results(self, metrics: dict, step: int):
        """Saves the latest metrics to a live JSON file for the GUI."""
        live_data = {
            'current_step': step,
            'latest_metrics': metrics,
            'full_results': self.results
        }
        with open(os.path.join(self.results_dir, 'live.json'), 'w') as f:
            json.dump(live_data, f, indent=2)

    def save_results(self):
        """Saves the final logged results to a JSON file."""
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_final_metrics(self) -> dict:
        """Returns the final value for each logged metric."""
        return {key: values[-1] for key, values in self.results.items() if values}
