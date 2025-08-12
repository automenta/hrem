import torch
import torch.nn as nn
import pytest
import os
from unittest.mock import patch
from src.training import Trainer
from src.datasets import ReverseDataset
from typing import Dict


class MockModel(nn.Module):
    """A mock model for testing the Trainer."""

    def __init__(self, input_size=16, output_size=16):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Accepts a batch dictionary and returns a dictionary with logits,
        mimicking the behavior of the real models.
        """
        x = batch["inputs"]
        # The trainer will pass a sequence, so we need to handle it.
        # For simplicity, we just process each element in the sequence.
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        # Ensure the input matches the linear layer size
        if x.shape[1] != self.linear.in_features:
            x = x[:, : self.linear.in_features]

        logits = self.linear(x)
        return {"logits": logits}


@pytest.fixture
def trainer_setup():
    """Sets up a Trainer instance with a MockModel."""
    model = MockModel()
    train_ds = ReverseDataset(size=20, seq_len=16)
    test_ds = ReverseDataset(size=10, seq_len=16)
    config = {
        "experiment_name": "test_trainer_experiment",
        "dataset": {"name": "reverse"},
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "checkpointing": True,
        },
    }
    trainer = Trainer(model, train_ds, test_ds, config)
    yield trainer
    # Teardown: clean up created files
    results_dir = trainer.logger.results_dir
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, f))
        os.rmdir(results_dir)


def test_trainer_init(trainer_setup):
    """Tests the initialization of the Trainer."""
    assert trainer_setup is not None
    assert trainer_setup.device == torch.device("cpu")
    assert isinstance(trainer_setup.criterion, nn.MSELoss)


def test_trainer_train_epoch(trainer_setup):
    """Tests a single training epoch."""
    trainer = trainer_setup
    from tqdm import tqdm

    pbar = tqdm(trainer.train_loader)
    train_loss = trainer._train_epoch(pbar)
    assert isinstance(train_loss, float)
    assert train_loss > 0


def test_trainer_evaluate(trainer_setup):
    """Tests the evaluation method."""
    trainer = trainer_setup
    test_loss = trainer._evaluate()
    assert isinstance(test_loss, float)
    assert test_loss >= 0.0


def test_trainer_run(trainer_setup):
    """Tests the main run loop."""
    trainer = trainer_setup
    trainer.run()
    # Check if results files were created
    results_dir = trainer.logger.results_dir
    assert os.path.exists(os.path.join(results_dir, "config.json"))
    assert os.path.exists(os.path.join(results_dir, "results.json"))
    assert os.path.exists(os.path.join(results_dir, "best_model.pt"))


def test_torch_compile_enabled():
    """Tests that torch.compile is called when enabled in the config."""
    with patch("torch.compile") as mock_compile:
        model = MockModel()
        train_ds = ReverseDataset(size=20, seq_len=16)
        test_ds = ReverseDataset(size=10, seq_len=16)
        config = {
            "experiment_name": "test_compile_experiment",
            "dataset": {"name": "reverse"},
            "training": {
                "batch_size": 4,
                "epochs": 1,
                "learning_rate": 0.001,
                "use_torch_compile": True,
            },
        }
        Trainer(model, train_ds, test_ds, config)
        mock_compile.assert_called_once()


def test_one_cycle_lr_scheduler():
    """Tests the OneCycleLR scheduler setup."""
    model = MockModel()
    train_ds = ReverseDataset(size=20, seq_len=16)
    test_ds = ReverseDataset(size=10, seq_len=16)
    config = {
        "experiment_name": "test_one_cycle_lr",
        "dataset": {"name": "reverse"},
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "lr_scheduler": {
                "enabled": True,
                "type": "OneCycleLR",
                "max_lr": 0.01,
            },
        },
    }
    trainer = Trainer(model, train_ds, test_ds, config)
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)
