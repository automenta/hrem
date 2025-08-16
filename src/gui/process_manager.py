import fcntl
import os
import subprocess
import sys
from .constants import (
    RESULTS_DIR,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
)


class BaseProcessManager:
    """
    A base class for managing subprocesses.
    Handles launching, stopping, and tracking statuses of processes.
    """

    def __init__(self):
        self.processes = {}  # Tracks running subprocesses: {name: (process, log_path)}

    def _get_name_from_config(self, config_path: str) -> str:
        """
        Extracts the experiment name from the config file path.
        The name is assumed to be the directory containing the config file.
        e.g., "results/my-exp/config.json" -> "my-exp"
        """
        return os.path.basename(os.path.dirname(config_path))

    def launch_process(self, config_path: str, script_path: str):
        """
        Launches a script in a new process and tracks it.
        Redirects stdout/stderr to a log file.
        """
        name = self._get_name_from_config(config_path)

        # Ensure the results directory for the experiment exists
        log_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "output.log")

        process = subprocess.Popen(
            [sys.executable, script_path, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        )

        # Set stdout to be non-blocking
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self.processes[name] = (process, log_path)
        return name

    def stop_process(self, name: str, force: bool = False) -> bool:
        """
        Stops a running process by its name.

        Args:
            name (str): The name of the process to stop.
            force (bool): If True, sends a kill signal (SIGKILL).
                          If False, sends a terminate signal (SIGTERM).

        Returns:
            bool: True if the signal was sent, False otherwise.
        """
        if name in self.processes:
            process, _ = self.processes[name]
            if force:
                process.kill()
            else:
                process.terminate()
            return True
        return False

    def get_statuses(self) -> dict:
        """
        Checks the status of all tracked processes.
        """
        statuses = {}
        finished_processes = []
        for name, (process, _log_path) in self.processes.items():
            if process.poll() is None:
                statuses[name] = STATUS_RUNNING
                continue

            finished_processes.append(name)
            if process.returncode == 0:
                statuses[name] = STATUS_COMPLETED
            else:
                statuses[name] = STATUS_FAILED

        # Clean up finished processes from the tracking dict
        for name in finished_processes:
            # Before deleting, do one last log update to catch any final output
            self.update_log_files()
            del self.processes[name]

        return statuses

    def update_log_files(self):
        """
        Iterates through running processes, reads their output,
        and appends it to log files.
        """
        for name, (process, log_path) in self.processes.items():
            try:
                output = process.stdout.read()
                if output:
                    with open(log_path, "ab") as f:
                        f.write(output)
            except (IOError, TypeError, ValueError):
                # Process might have finished, or other reading errors
                pass

    def read_log_file(self, name: str) -> str:
        """
        Reads the entire contents of a log file for a given experiment.
        """
        log_path = os.path.join(RESULTS_DIR, name, "output.log")
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    return f.read()
            except IOError:
                return f"Error: Could not read log file at {log_path}"
        return ""  # Return empty string if no log file
