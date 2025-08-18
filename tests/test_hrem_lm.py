import pytest
import torch

from src.factories import get_dataset, get_model
from src.training import Trainer


def get_hrem_lm_config():
    """Provides a base configuration for an HREM language model test."""
    return {
        "experiment_name": "test_hrem_lm",
        "model": {
            "name": "hrem",
            "params": {
                "input_size": 1,  # Character-level
                "hidden_size": 128,
                "H_layers": 2,
                "L_layers": 1,
                "H_cycles": 1,
                "L_cycles": 1,
                "num_heads": 4,
                "expansion": 2.0,
                "pos_encodings": "rope",
                "halt_max_steps": 5,
                "halt_exploration_prob": 0.0,
                "use_memory": False,
                "vocab_size": 65,  # From Tiny Shakespeare dataset
                "seq_len": 16,
                "batch_size": 4,
                "puzzle_emb_ndim": 0,
                "num_puzzle_identifiers": 1,
                "forward_dtype": "float32",
            },
        },
        "dataset": {
            "name": "tiny_shakespeare",
            "params": {"seq_length": 16, "split": "train"},
        },
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
    }


def get_transformer_lm_config():
    """Provides a base configuration for a Transformer language model test."""
    return {
        "experiment_name": "test_transformer_lm",
        "model": {
            "name": "transformer",
            "params": {
                "input_size": 65,  # Vocab size
                "hidden_size": 128,
                "output_size": 65,  # Vocab size
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "forward_dtype": "float32",
            },
        },
        "dataset": {
            "name": "tiny_shakespeare",
            "params": {"seq_length": 16, "split": "train"},
        },
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
    }


def run_training_test(config):
    """Helper function to run a short training session and return final loss."""
    train_ds, test_ds = get_dataset(config)
    model = get_model(config)
    trainer = Trainer(model, train_ds, test_ds, config)
    final_metrics = trainer.run()
    return final_metrics


@pytest.mark.slow
def test_hrem_and_transformer_on_lm_task():
    """
    Tests that both HREM and a baseline Transformer can be successfully trained
    for one epoch on a small language modeling task.
    This test verifies that the core training loop and model integrations are
    working correctly, independent of the GUI.
    """
    # Test HREM
    hrem_config = get_hrem_lm_config()
    hrem_metrics = run_training_test(hrem_config)
    assert "test_loss" in hrem_metrics
    assert isinstance(hrem_metrics["test_loss"], float)

    # Test Transformer
    transformer_config = get_transformer_lm_config()
    transformer_metrics = run_training_test(transformer_config)
    assert "test_loss" in transformer_metrics
    assert isinstance(transformer_metrics["test_loss"], float)
