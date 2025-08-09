import torch
import torch.nn as nn
import pytest
import os
from src.training import Trainer
from src.datasets import ReverseDataset

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # This mock model flattens the sequence and processes it.
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        # The trainer will pass a sequence, so we need to handle it.
        # For simplicity, we just process each element in the sequence.
        # This is not a realistic model, but it's fine for testing the trainer loop.
        if x.dim() > 2:
             x = x.reshape(x.size(0), -1) # Flatten if it's a sequence

        # A mock model for a binary sequence task
        if x.shape[1] != 16:
            x = x[:, :16]

        return torch.sigmoid(self.linear(x))


@pytest.fixture
def trainer_setup():
    model = MockModel()
    train_ds = ReverseDataset(size=20, seq_len=16)
    test_ds = ReverseDataset(size=10, seq_len=16)
    config = {
        "experiment_name": "test_experiment",
        "dataset": { "name": "reverse" }, # Added dataset name
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001
        }
    }
    trainer = Trainer(model, train_ds, test_ds, config)
    yield trainer
    # Teardown: clean up created files
    if os.path.exists(os.path.join(trainer.results_dir, 'config.json')):
        os.remove(os.path.join(trainer.results_dir, 'config.json'))
    if os.path.exists(os.path.join(trainer.results_dir, 'results.json')):
        os.remove(os.path.join(trainer.results_dir, 'results.json'))
    if os.path.exists(os.path.join(trainer.results_dir, 'best_model.pt')):
        os.remove(os.path.join(trainer.results_dir, 'best_model.pt'))
    if os.path.exists(trainer.results_dir):
        os.rmdir(trainer.results_dir)


def test_trainer_init(trainer_setup):
    assert trainer_setup is not None
    assert trainer_setup.device == torch.device("cpu")
    assert isinstance(trainer_setup.criterion, nn.MSELoss)

def test_trainer_train_epoch(trainer_setup):
    trainer = trainer_setup
    # The train_epoch method now takes a progress bar object
    from tqdm import tqdm
    pbar = tqdm(trainer.train_loader)
    train_loss = trainer._train_epoch(pbar)
    assert isinstance(train_loss, float)
    assert train_loss > 0

def test_trainer_evaluate(trainer_setup):
    trainer = trainer_setup
    test_loss = trainer._evaluate()
    assert isinstance(test_loss, float)
    assert test_loss >= 0.0

def test_trainer_run(trainer_setup):
    trainer = trainer_setup
    trainer.run()
    # Check if results files were created
    assert os.path.exists(os.path.join(trainer.results_dir, 'config.json'))
    assert os.path.exists(os.path.join(trainer.results_dir, 'results.json'))
    assert os.path.exists(os.path.join(trainer.results_dir, 'best_model.pt'))
