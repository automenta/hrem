import torch
import pytest
from src.datasets import ReverseDataset, TinyShakespeareDataset

def test_reverse_dataset():
    seq_len = 16
    dataset = ReverseDataset(size=10, seq_len=seq_len)

    assert len(dataset) == 10

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert torch.equal(x.flip(0), y)
    assert x.dtype == torch.float32

def test_tiny_shakespeare_dataset():
    seq_len = 50
    dataset = TinyShakespeareDataset(seq_length=seq_len, split='train')

    assert dataset.vocab_size > 0
    assert len(dataset) > 0

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Check that y is the shifted version of x
    assert torch.equal(x[1:], y[:-1])
