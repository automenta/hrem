import os
import json
import shutil
import pytest
from unittest.mock import patch

from src.gui.experiment_manager import ExperimentManager


# Fixture to create a temporary directory for integration tests
@pytest.fixture
def temp_integration_env(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # Create dummy config files
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    base_models_dir = configs_dir / "base" / "models"
    base_models_dir.mkdir(parents=True)

    # A simple config for a single run
    single_run_config = {
        "experiment_name": "test_single_run",
        "model": {"name": "test_model", "params": {"layer_sizes": [10, 5]}},
        "dataset": {"name": "test_dataset", "params": {"n_samples": 100}},
        "training": {"epochs": 1, "batch_size": 10},
    }
    single_run_config_path = configs_dir / "single_run.json"
    with open(single_run_config_path, "w") as f:
        json.dump(single_run_config, f)

    # A challenger config for a race
    challenger_config = {
        "experiment_name": "test_challenger",
        "model": {"name": "challenger_model", "params": {"layer_sizes": [20, 10]}},
        "dataset": {"name": "test_dataset", "params": {"n_samples": 100}},
        "training": {"epochs": 1, "batch_size": 10},
    }
    challenger_config_path = configs_dir / "challenger.json"
    with open(challenger_config_path, "w") as f:
        json.dump(challenger_config, f)

    # A baseline model config
    baseline_model_config = {"name": "baseline_model", "params": {"hidden_size": 15}}
    baseline_model_config_path = base_models_dir / "baseline.json"
    with open(baseline_model_config_path, "w") as f:
        json.dump(baseline_model_config, f)

    # A base dataset config
    base_datasets_dir = configs_dir / "base" / "datasets"
    base_datasets_dir.mkdir(parents=True)
    dataset_config = {"name": "test_dataset", "params": {"n_samples": 200}}
    dataset_config_path = base_datasets_dir / "test_dataset.json"
    with open(dataset_config_path, "w") as f:
        json.dump(dataset_config, f)

    patch1 = patch("src.gui.experiment_manager.RESULTS_DIR", str(results_dir))
    patch2 = patch("src.gui.experiment_manager.ARCHIVE_DIR", str(archive_dir))
    patch3 = patch(
        "src.gui.experiment_manager.BASE_MODELS_DIR", str(base_models_dir)
    )
    patch4 = patch(
        "src.gui.experiment_manager.BASE_DATASETS_DIR", str(base_datasets_dir)
    )
    with patch1, patch2, patch3, patch4:
        yield {
            "results_dir": results_dir,
            "archive_dir": archive_dir,
            "configs_dir": configs_dir,
            "single_run_config_path": single_run_config_path,
            "challenger_config_path": challenger_config_path,
        }

    # Clean up created directories
    shutil.rmtree(results_dir)
    shutil.rmtree(archive_dir)
    shutil.rmtree(configs_dir)


@patch("src.gui.process_manager.subprocess.Popen")
def test_full_experiment_lifecycle(mock_popen, temp_integration_env):
    """
    Tests the full lifecycle of experiments: launch, check status, archive, restore, rename.
    """
    # Configure the mock to return a process with a valid stdout
    mock_process = mock_popen.return_value
    mock_process.stdout.fileno.return_value = 1
    mock_process.poll.return_value = 0  # Simulate completed process
    mock_process.returncode = 0  # Explicitly set the returncode

    manager = ExperimentManager()

    # --- 1. Launch a single experiment ---
    with open(temp_integration_env["single_run_config_path"], "r") as f:
        single_run_config = json.load(f)
    success, msg = manager.launch_experiment_from_config(
        single_run_config, "single_exp_1"
    )
    assert success, f"Failed to launch single experiment: {msg}"

    # --- 2. Launch a challenge/race ---
    launch_info = {
        "challenger_config": str(temp_integration_env["challenger_config_path"]),
        "base_name": "my_race",
            "standard_baselines": ["baseline"],
        "dataset": "test_dataset",  # Added dataset for the race
    }
    success, msg = manager.launch_experiment_race(launch_info)
    assert success, f"Failed to launch experiment race: {msg}"

    # Verify that the processes were "launched"
    assert (
        mock_popen.call_count == 3
    )  # single_exp_1, my_race_challenger, my_race_baseline_baseline

    # --- 3. Check experiment status ---
    # Manually create dummy result files to simulate completion
    exp_dir = os.path.join(temp_integration_env["results_dir"], "single_exp_1")
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({"test_loss": [0.1]}, f)

    manager.update_log_files()
    statuses = manager.get_experiment_statuses()
    # With the poll() method mocked, the processes will appear as completed.
    assert statuses["single_exp_1"] == "Completed"
    assert statuses["my_race_challenger"] == "Completed"

    # --- 4. Archive an experiment ---
    success, msg = manager.archive_experiment("single_exp_1")
    assert success, f"Failed to archive experiment: {msg}"
    assert not os.path.exists(temp_integration_env["results_dir"] / "single_exp_1")
    assert os.path.exists(temp_integration_env["archive_dir"] / "single_exp_1")

    # --- 5. Restore an experiment ---
    success, msg = manager.restore_experiment("single_exp_1")
    assert success, f"Failed to restore experiment: {msg}"
    assert os.path.exists(temp_integration_env["results_dir"] / "single_exp_1")
    assert not os.path.exists(temp_integration_env["archive_dir"] / "single_exp_1")

    # --- 6. Rename an experiment ---
    success, msg = manager.rename_experiment("single_exp_1", "single_exp_renamed")
    assert success, f"Failed to rename experiment: {msg}"
    assert not os.path.exists(temp_integration_env["results_dir"] / "single_exp_1")
    assert os.path.exists(temp_integration_env["results_dir"] / "single_exp_renamed")


@patch("src.gui.process_manager.subprocess.Popen")
def test_experiment_manager_failures(mock_popen, temp_integration_env):
    """
    Tests various failure modes of the ExperimentManager.
    """
    mock_process = mock_popen.return_value
    mock_process.poll.return_value = 0  # Simulate completed process

    manager = ExperimentManager()

    # --- 1. Launch with invalid config (missing required keys) ---
    invalid_config = {"model": {"name": "test_model"}}
    success, msg = manager.launch_experiment_from_config(invalid_config, "invalid_exp")
    # This should ideally fail, and the current implementation correctly does.
    # We assert that the launch fails as expected.
    assert not success

    # --- 2. Rename to existing name ---
    with open(temp_integration_env["single_run_config_path"], "r") as f:
        config = json.load(f)
    manager.launch_experiment_from_config(config, "exp_to_rename")
    manager.launch_experiment_from_config(config, "existing_name")

    success, msg = manager.rename_experiment("exp_to_rename", "existing_name")
    assert not success
    assert "already exists" in msg

    # --- 3. Archive a non-existent experiment ---
    success, msg = manager.archive_experiment("non_existent_exp")
    assert not success
    assert "not found" in msg

    # --- 4. Restore a non-existent experiment ---
    success, msg = manager.restore_experiment("non_existent_exp")
    assert not success
    assert "not found" in msg
