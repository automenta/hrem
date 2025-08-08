import torch
import pytest
from src.models import MLP, HRM, HREM

@pytest.fixture
def dummy_input():
    return torch.randn(4, 64) # Batch size of 4, sequence length of 64

def test_mlp_forward(dummy_input):
    model = MLP(input_size=64, hidden_size=128, output_size=64)
    output = model(dummy_input)
    assert output.shape == (4, 64)
    assert torch.all(output >= 0) and torch.all(output <= 1) # Check sigmoid output

def test_hrm_forward(dummy_input):
    model = HRM(seq_len=64, d_model=80, n_cycles=2, t_steps=3)
    output = model(dummy_input)
    assert output.shape == (4, 64)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_hrem_forward_with_memory(dummy_input):
    model = HREM(
        seq_len=64,
        d_model=50,
        n_cycles=2,
        t_steps=3,
        use_memory=True,
        m_loc=32,
        d_mem=16,
        top_k=4
    )
    output = model(dummy_input)
    assert output.shape == (4, 64)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_hrem_forward_no_memory(dummy_input):
    model = HREM(
        seq_len=64,
        d_model=50,
        n_cycles=2,
        t_steps=3,
        use_memory=False
    )
    output = model(dummy_input)
    assert output.shape == (4, 64)
    assert torch.all(output >= 0) and torch.all(output <= 1)
