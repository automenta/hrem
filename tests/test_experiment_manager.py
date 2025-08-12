import os
import json
import pytest
from unittest.mock import patch

from src.gui.experiment_manager import ExperimentManager


# Fixture to create a temporary directory structure for testing
@pytest.fixture
def temp_experiment_dirs(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # Create a dummy experiment
    exp1_dir = results_dir / "exp1"
    exp1_dir.mkdir()
    with open(exp1_dir / "config.json", "w") as f:
        json.dump(
            {"model": {"name": "test_model"}, "dataset": {"name": "test_dataset"}}, f
        )
    with open(exp1_dir / "results.json", "w") as f:
        json.dump({"test_loss": [0.1, 0.05]}, f)

    # Create a dummy archived experiment
    arch_exp1_dir = archive_dir / "arch_exp1"
    arch_exp1_dir.mkdir()
    with open(arch_exp1_dir / "config.json", "w") as f:
        json.dump(
            {"model": {"name": "arch_model"}, "dataset": {"name": "arch_dataset"}}, f
        )

    with patch("src.gui.experiment_manager.RESULTS_DIR", str(results_dir)), patch(
        "src.gui.experiment_manager.ARCHIVE_DIR", str(archive_dir)
    ):
        yield str(results_dir), str(archive_dir)


def test_load_experiment_config(temp_experiment_dirs):
    results_dir, _ = temp_experiment_dirs
    manager = ExperimentManager()

    config, err = manager.load_experiment_config("exp1")
    assert err is None
    assert config["model"]["name"] == "test_model"

    # Test loading from a specific base directory
    config, err = manager.load_experiment_config("exp1", base_dir=results_dir)
    assert err is None
    assert config["model"]["name"] == "test_model"


def test_load_experiment_results(temp_experiment_dirs):
    results_dir, _ = temp_experiment_dirs
    manager = ExperimentManager()

    results, err = manager.load_experiment_results("exp1")
    assert err is None
    assert results["test_loss"] == [0.1, 0.05]

    results, err = manager.load_experiment_results("exp1", base_dir=results_dir)
    assert err is None
    assert results["test_loss"] == [0.1, 0.05]


def test_get_experiments_data(temp_experiment_dirs):
    manager = ExperimentManager()
    data = manager.get_experiments_data()
    assert len(data) == 1
    assert data[0]["name"] == "exp1"
    assert data[0]["model"] == "test_model"


def test_get_archived_experiments_data(temp_experiment_dirs):
    manager = ExperimentManager()
    data = manager.get_archived_experiments_data()
    assert len(data) == 1
    assert data[0]["name"] == "arch_exp1"
    assert data[0]["model"] == "arch_model"
    # This proves the refactoring works, as it's reading from the archive dir


def test_archive_and_restore_experiment(temp_experiment_dirs):
    manager = ExperimentManager()

    # Check initial state
    assert len(manager.get_experiments_data()) == 1
    assert len(manager.get_archived_experiments_data()) == 1

    # Archive
    success, msg = manager.archive_experiment("exp1")
    assert success
    assert len(manager.get_experiments_data()) == 0
    assert len(manager.get_archived_experiments_data()) == 2

    # Restore
    success, msg = manager.restore_experiment("exp1")
    assert success
    assert len(manager.get_experiments_data()) == 1
    assert len(manager.get_archived_experiments_data()) == 1


def test_delete_experiment_permanently(temp_experiment_dirs):
    manager = ExperimentManager()

    # Check initial state
    assert len(manager.get_archived_experiments_data()) == 1

    # Delete
    success, msg = manager.delete_experiment_permanently("arch_exp1")
    assert success
    assert len(manager.get_archived_experiments_data()) == 0


def test_rename_experiment(temp_experiment_dirs):
    results_dir, _ = temp_experiment_dirs
    manager = ExperimentManager()

    # Test successful rename
    success, msg = manager.rename_experiment("exp1", "exp1_renamed")
    assert success
    assert os.path.exists(os.path.join(results_dir, "exp1_renamed"))
    assert not os.path.exists(os.path.join(results_dir, "exp1"))

    # Test renaming non-existent experiment
    success, msg = manager.rename_experiment("nonexistent", "new_name")
    assert not success

    # Test renaming to an existing name
    os.mkdir(os.path.join(results_dir, "exp2"))
    success, msg = manager.rename_experiment("exp1_renamed", "exp2")
    assert not success


def test_clone_experiment(temp_experiment_dirs):
    results_dir, _ = temp_experiment_dirs
    manager = ExperimentManager()

    # Test successful clone
    success, msg = manager.clone_experiment("exp1", "exp1_cloned")
    assert success
    cloned_path = os.path.join(results_dir, "exp1_cloned")
    assert os.path.exists(cloned_path)
    assert os.path.exists(os.path.join(cloned_path, "config.json"))

    # Verify the experiment name in the cloned config
    config, err = manager.load_experiment_config("exp1_cloned")
    assert err is None
    assert config["experiment_name"] == "exp1_cloned"

    # Test cloning to an existing name
    success, msg = manager.clone_experiment("exp1", "exp1_cloned")
    assert not success

    # Test cloning a non-existent experiment
    success, msg = manager.clone_experiment("nonexistent", "new_clone")
    assert not success
