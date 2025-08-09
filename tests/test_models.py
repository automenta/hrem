import torch
import pytest
from src.models import MLP, HREM

@pytest.fixture
def dummy_seq_input():
    # Input for sequence models: (batch, seq_len, features)
    return torch.randn(4, 10, 8) # Batch=4, SeqLen=10, Features=8

@pytest.fixture
def dummy_flat_input():
    # Input for MLP models: (batch, features)
    return torch.randn(4, 64)

def test_mlp_forward(dummy_flat_input):
    model = MLP(input_size=64, hidden_size=128, output_size=64)
    output = model(dummy_flat_input)
    assert output.shape == (4, 64)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_hrem_baseline_forward(dummy_seq_input):
    # Test HREM without memory (acting as our new HRM baseline)
    model = HREM(
        input_size=8,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_cycles=1,
        use_memory=False
    )
    output = model(dummy_seq_input)
    assert output.shape == (4, 10, 8)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_hrem_with_memory_forward(dummy_seq_input):
    # Test HREM with memory enabled
    model = HREM(
        input_size=8,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_cycles=2,
        use_memory=True,
        m_loc=16,
        d_mem=12,
        top_k=4,
        use_location_addressing=True
    )
    output = model(dummy_seq_input)
    assert output.shape == (4, 10, 8)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_hrem_language_model_forward():
    # Test HREM in language modeling mode
    # Input is (batch, seq_len) of token indices
    dummy_lm_input = torch.randint(0, 50, (4, 15))
    model = HREM(
        input_size= -1, # Not used when vocab_size is provided
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_cycles=1,
        use_memory=False,
        vocab_size=50
    )
    output = model(dummy_lm_input)
    # Output should be (batch, seq_len, vocab_size)
    assert output.shape == (4, 15, 50)
