from .process_manager import BaseProcessManager


class SearchManager(BaseProcessManager):
    """
    Handles the logic for launching and monitoring hyperparameter search studies.
    Inherits process management from BaseProcessManager.
    """

    def __init__(self):
        super().__init__()

    def launch_search(self, config_path: str):
        """
        Launches a hyperparameter search in a new process.
        """
        return self.launch_process(config_path, "src/search.py")

    def stop_search(self, search_name: str) -> bool:
        """
        Stops a running search process.
        """
        return self.stop_process(search_name)

    def get_search_statuses(self) -> dict:
        """
        Checks the status of all running searches.
        """
        return self.get_statuses()

    def get_search_output(self, search_name: str) -> str:
        """
        Gets the latest stdout from a running search process.
        """
        return self.get_process_output(search_name)
