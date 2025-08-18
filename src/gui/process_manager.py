import os
import signal
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
        self.processes = {}  # Tracks running subprocesses: {name: (process, log_path, pid)}
        self._recover_orphaned_processes()

    def _get_pid_file_path(self, name: str) -> str:
        """Returns the canonical path for a process's PID file."""
        return os.path.join(RESULTS_DIR, name, ".pid")

    def _recover_orphaned_processes(self):
        """
        Scans the results directory for .pid files and "adopts" any
        processes that were orphaned from a previous GUI session.
        """
        if not os.path.exists(RESULTS_DIR):
            return

        for exp_name in os.listdir(RESULTS_DIR):
            pid_file = self._get_pid_file_path(exp_name)
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())
                except (IOError, ValueError):
                    # Invalid PID file, clean it up.
                    os.remove(pid_file)
                    continue

                if self._is_pid_running(pid):
                    # This is an orphan. Adopt it.
                    # We don't have the Popen object, so we store None.
                    # We also don't have the log_path in this context, but we can reconstruct it.
                    log_path = os.path.join(RESULTS_DIR, exp_name, "output.log")
                    self.processes[exp_name] = (None, log_path, pid)
                else:
                    # The process is not running, so the PID file is stale. Clean it up.
                    os.remove(pid_file)

    def _is_pid_running(self, pid: int) -> bool:
        """
        Checks if a process with the given PID is currently running.
        This is a cross-platform implementation.
        """
        if pid <= 0:
            return False

        if sys.platform == "win32":
            # Use the 'tasklist' command to check for the process
            try:
                output = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    stderr=subprocess.STDOUT,
                )
                return str(pid) in str(output)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # CalledProcessError means tasklist ran but found no such process.
                # FileNotFoundError means tasklist command doesn't exist (highly unlikely).
                return False
        else:
            # On POSIX, we can use os.kill with signal 0 as a check.
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                # This is the expected error if the process does not exist.
                return False
            except PermissionError:
                # The process exists, but we don't have permission to signal it.
                # This still means it's running.
                return True
            except OSError:
                 # Other OS-level errors can happen, safer to assume it's not running.
                return False
            else:
                return True

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

        # Set stdout to be non-blocking on Unix-like systems
        if sys.platform != "win32":
            import fcntl
            fd = process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # Write the PID file
        pid_file_path = self._get_pid_file_path(name)
        try:
            with open(pid_file_path, "w") as f:
                f.write(str(process.pid))
        except IOError as e:
            # If we can't write the PID file, we should kill the process
            # to avoid orphaning it without a way to track it.
            process.kill()
            raise IOError(f"Could not write PID file to {pid_file_path}: {e}") from e

        self.processes[name] = (process, log_path, process.pid)
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
            process, _log_path, pid = self.processes[name]
            if process: # It's a process we launched directly
                if force:
                    process.kill()
                else:
                    process.terminate()
                return True
            elif pid: # It's a recovered orphan, we only have the PID
                try:
                    # Use signal.SIGTERM for graceful, signal.SIGKILL for force
                    sig = signal.SIGKILL if force else signal.SIGTERM
                    os.kill(pid, sig)
                    return True
                except ProcessLookupError:
                    # The process died before we could kill it
                    return False
                except OSError:
                     # Other OS-level errors (e.g., permissions)
                    return False
        return False

    def get_statuses(self) -> dict:
        """
        Checks the status of all tracked processes.
        """
        statuses = {}
        finished_processes = []
        for name, (process, _log_path, pid) in self.processes.items():
            # If we have a Popen object, we can poll it.
            if process:
                if process.poll() is None:
                    statuses[name] = STATUS_RUNNING
                    continue

                # Process has finished
                finished_processes.append(name)
                if process.returncode == 0:
                    statuses[name] = STATUS_COMPLETED
                else:
                    statuses[name] = STATUS_FAILED
            # If it's a recovered process, we check if the PID is still alive.
            elif pid:
                if self._is_pid_running(pid):
                    statuses[name] = STATUS_RUNNING
                else:
                    # We can't know the exit code, so we assume it completed
                    # successfully if it's gone.
                    finished_processes.append(name)
                    statuses[name] = STATUS_COMPLETED


        # Clean up finished processes from the tracking dict
        for name in finished_processes:
            # Before deleting, do one last log update to catch any final output
            self.update_log_files()
            if name in self.processes:
                del self.processes[name]

            # Clean up the PID file
            pid_file = self._get_pid_file_path(name)
            if os.path.exists(pid_file):
                try:
                    os.remove(pid_file)
                except OSError:
                    # Log this error? For now, we'll just ignore it.
                    pass

        return statuses

    def update_log_files(self):
        """
        Iterates through running processes, reads their output,
        and appends it to log files.
        """
        for name, (process, log_path, pid) in self.processes.items():
            if not process: # It's a recovered process, we can't read stdout
                continue
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
