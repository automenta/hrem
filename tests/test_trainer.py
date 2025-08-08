import torch
import torch.nn as nn
import pytest
from src.training import Trainer
from src.datasets import ReverseDataset

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

@pytest.fixture
def trainer_setup():
    model = MockModel()
    train_ds = ReverseDataset(size=20, seq_len=16)
    test_ds = ReverseDataset(size=10, seq_len=16)
    config = {
        "experiment_name": "test_experiment",
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "loss": "bce"
        }
    }
    trainer = Trainer(model, train_ds, test_ds, config)
    return trainer

def test_trainer_init(trainer_setup):
    assert trainer_setup is not None
    assert trainer_setup.device == torch.device("cpu") # Assuming no GPU in test env
    assert isinstance(trainer_setup.loss_fn, nn.BCELoss)

def test_trainer_train_epoch(trainer_setup):
    trainer = trainer_setup
    train_loss = trainer.train_epoch()
    assert isinstance(train_loss, float)
    assert train_loss > 0

def test_trainer_evaluate(trainer_setup):
    trainer = trainer_setup
    test_acc = trainer.evaluate()
    assert isinstance(test_acc, float)
    assert 0.0 <= test_acc <= 1.0

def test_trainer_run(trainer_setup):
    trainer = trainer_setup
    results = trainer.run()
    assert "train_loss" in results
    assert len(results["train_loss"]) == 1
    assert "test_acc" in results
    assert results["test_acc"] > 0
