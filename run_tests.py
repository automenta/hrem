import pytest
import sys
import os

if __name__ == "__main__":
    # Add the project root to the python path
    os.environ["PYTHONPATH"] = "."
    sys.exit(pytest.main(["-v"]))
