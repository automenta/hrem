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

    # Create base configs needed for race tests
    base_models_dir = tmp_path / "configs" / "base" / "models"
    base_models_dir.mkdir(parents=True)
    with open(base_models_dir / "lstm.json", "w") as f:
        json.dump({"name": "lstm", "params": {"hidden_size": 10}}, f)

    # Create a dummy experiment
    exp1_dir = results_dir / "exp1"
    exp1_dir.mkdir()
    config_data = {
        "experiment_name": "exp1",
        "model": {"name": "test_model"},
        "dataset": {"name": "test_dataset", "params": {}},
        "training": {"epochs": 1, "learning_rate": 0.01}
    }
    with open(exp1_dir / "config.json", "w") as f:
        json.dump(config_data, f)

    with patch("src.gui.experiment_manager.RESULTS_DIR", str(results_dir)), \
         patch("src.gui.experiment_manager.BASE_MODELS_DIR", str(base_models_dir)), \
         patch("src.gui.launch_dialog.CONFIGS_DIR", str(tmp_path / "configs")):
        yield str(results_dir)

def test_clone_experiment_tracks_parent(temp_experiment_dirs):
    manager = ExperimentManager()

    # Clone the experiment
    success, msg = manager.clone_experiment("exp1", "exp1_clone")
    assert success

    # Verify the parent is tracked in the new config
    config, err = manager.load_experiment_config("exp1_clone")
    assert err is None
    assert config["parent_experiment"] == "exp1"

def test_get_experiment_graph(temp_experiment_dirs):
    results_dir = temp_experiment_dirs
    manager = ExperimentManager()

    # Create a family of experiments
    # exp1 is a root
    # exp1 -> exp2
    # exp1 -> exp3
    # exp3 -> exp4
    # exp5 is another root
    manager.clone_experiment("exp1", "exp2")
    manager.clone_experiment("exp1", "exp3")
    manager.clone_experiment("exp3", "exp4")

    exp5_dir = os.path.join(results_dir, "exp5")
    os.mkdir(exp5_dir)
    with open(os.path.join(exp5_dir, "config.json"), "w") as f:
        json.dump({"experiment_name": "exp5"}, f)

    graph = manager.get_experiment_graph()

    assert set(graph["nodes"].keys()) == {"exp1", "exp2", "exp3", "exp4", "exp5"}
    assert set(graph["roots"]) == {"exp1", "exp5"}
    assert set(graph["edges"]) == {("exp1", "exp2"), ("exp1", "exp3"), ("exp3", "exp4")}

@patch("src.gui.experiment_manager.ExperimentManager.launch_experiment")
def test_launch_experiment_race(mock_launch, temp_experiment_dirs):
    results_dir = temp_experiment_dirs
    manager = ExperimentManager()

    challenger_config_path = os.path.join(results_dir, "exp1", "config.json")

    launch_info = {
        "challenger_config": challenger_config_path,
        "base_name": "my_race",
        "baselines": ["lstm"],
    }

    success, msg = manager.launch_experiment_race(launch_info)
    assert success

    # Check that challenger and baseline directories were created
    challenger_dir = os.path.join(results_dir, "my_race_challenger")
    baseline_dir = os.path.join(results_dir, "my_race_baseline_lstm")
    assert os.path.isdir(challenger_dir)
    assert os.path.isdir(baseline_dir)

    # Check challenger config
    challenger_config, _ = manager.load_experiment_config("my_race_challenger")
    assert "race_id" in challenger_config
    race_id = challenger_config["race_id"]

    # Check baseline config
    baseline_config, _ = manager.load_experiment_config("my_race_baseline_lstm")
    assert baseline_config["race_id"] == race_id
    assert baseline_config["is_baseline_for"] == "my_race_challenger"
    assert baseline_config["model"]["name"] == "lstm"
    assert baseline_config["model"]["params"]["hidden_size"] == 10

    # Check that launch_experiment was called for both
    assert mock_launch.call_count == 2
