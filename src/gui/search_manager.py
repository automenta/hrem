import os
import subprocess
import sys


class SearchManager:
    """
    Handles the logic for launching and monitoring hyperparameter search studies.
    """

    def __init__(self):
        self.processes = {}  # Tracks running search processes

    def launch_search(self, config_path):
        """
        Launches a hyperparameter search in a new process.
        """
        base_name = os.path.basename(config_path)
        search_name = base_name.replace(".json", "")

        process = subprocess.Popen(
            [sys.executable, "src/search.py", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes[search_name] = process
        return search_name

    def stop_search(self, search_name):
        """
        Stops a running search process.
        """
        if search_name in self.processes:
            self.processes[search_name].terminate()
            return True
        return False

    def get_search_statuses(self):
        """
        Checks the status of all running searches.
        """
        statuses = {}
        finished_processes = []
        for search_name, process in self.processes.items():
            if process.poll() is None:
                statuses[search_name] = "Running"
            else:
                finished_processes.append(search_name)
                if process.returncode == 0:
                    statuses[search_name] = "Completed"
                else:
                    statuses[search_name] = "Failed"

        for search_name in finished_processes:
            del self.processes[search_name]

        return statuses
