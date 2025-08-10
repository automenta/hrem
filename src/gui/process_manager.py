import fcntl
import os
import subprocess
import sys
from .constants import STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED


class BaseProcessManager:
    """
    A base class for managing subprocesses.
    Handles launching, stopping, and tracking statuses of processes.
    """
    def __init__(self):
        self.processes = {}  # Tracks running subprocesses: {name: (process, stdout_path)}

    def _get_name_from_config(self, config_path: str) -> str:
        """
        Extracts a unique name from the config file path.
        e.g., "configs/copy_hrem_baseline.json" -> "copy_hrem_baseline"
        """
        base_name = os.path.basename(config_path)
        return base_name.replace(".json", "").replace(".conf", "")

    def launch_process(self, config_path: str, script_path: str):
        """
        Launches a script in a new process and tracks it.
        """
        name = self._get_name_from_config(config_path)
        process = subprocess.Popen(
            [sys.executable, script_path, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        )

        # Set stdout to be non-blocking
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self.processes[name] = process
        return name

    def stop_process(self, name: str) -> bool:
        """
        Stops a running process by its name.
        """
        if name in self.processes:
            self.processes[name].terminate()
            return True
        return False

    def get_statuses(self) -> dict:
        """
        Checks the status of all tracked processes.
        """
        statuses = {}
        finished_processes = []
        for name, process in self.processes.items():
            return_code = process.poll()
            if return_code is None:
                statuses[name] = STATUS_RUNNING
            else:
                finished_processes.append(name)
                if return_code == 0:
                    statuses[name] = STATUS_COMPLETED
                else:
                    statuses[name] = STATUS_FAILED

        # Clean up finished processes from the tracking dict
        for name in finished_processes:
            del self.processes[name]

        return statuses

    def get_process_output(self, name: str) -> str:
        """
        Reads any available output from the process's stdout.
        """
        if name not in self.processes:
            return ""

        try:
            output = self.processes[name].stdout.read()
            return output.decode("utf-8") if output else ""
        except (TypeError, ValueError):
            return "" # No output
