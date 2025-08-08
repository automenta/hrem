import torch
from torch.utils.data import Dataset

class ReverseDataset(Dataset):
    """
    A dataset for the sequence reversal task.
    Each sample is a random binary sequence, and the target is its reverse.
    """
    def __init__(self, size, seq_len):
        self.size = size
        self.seq_len = seq_len
        self.data = (torch.rand(size, seq_len) < 0.5).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.flip(0)
        return x, y

class TinyShakespeareDataset(Dataset):
    """
    A character-level language modeling dataset based on a tiny snippet of Shakespeare.
    """
    def __init__(self, seq_length=100, split='train'):
        self.seq_length = seq_length
        self.text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking; let's do't: away, away!
"""
        self.chars = sorted(list(set(self.text)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.encoded_text = [self.char_to_int[ch] for ch in self.text]

        # Simple train/test split
        n = len(self.encoded_text)
        if split == 'train':
            self.encoded_text = self.encoded_text[:int(n*0.9)]
        else:
            self.encoded_text = self.encoded_text[int(n*0.9):]

    @property
    def vocab_size(self):
        return len(self.chars)

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.encoded_text[idx:idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
