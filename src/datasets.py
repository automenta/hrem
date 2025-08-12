import torch
from torch.utils.data import Dataset
import gymnasium as gym


class GymnasiumDataset:
    """
    A wrapper for Gymnasium environments to be used with the RL-extended Trainer.
    This is not a traditional PyTorch Dataset, but rather an environment manager.
    """

    def __init__(self, env_name, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def sample_action(self):
        return self.action_space.sample()

    def get_model_config_updates(self, model_name: str) -> dict:
        """
        Returns model-specific config updates for this dataset.
        """
        return {"input_size": self.obs_space.shape[0]}


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

    def get_model_config_updates(self, model_name: str) -> dict:
        """
        Returns model-specific config updates for this dataset.
        """
        if model_name == "mlp":
            return {
                "input_size": self.seq_len,
                "output_size": self.seq_len,
            }
        else:
            # For recurrent models
            return {"input_size_per_step": 1}


class TinyShakespeareDataset(Dataset):
    """
    A character-level language modeling dataset based on a tiny snippet of Shakespeare.
    """

    def __init__(self, seq_length=100, split="train"):
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
        if split == "train":
            self.encoded_text = self.encoded_text[: int(n * 0.9)]
        else:
            self.encoded_text = self.encoded_text[int(n * 0.9) :]

    @property
    def vocab_size(self):
        return len(self.chars)

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.encoded_text[idx : idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def get_model_config_updates(self, model_name: str) -> dict:
        """
        Returns model-specific config updates for this dataset.
        """
        return {
            "vocab_size": self.vocab_size,
            "output_size": self.vocab_size,
        }


class CopyTaskDataset(Dataset):
    """
    A dataset for the copy task.
    The model receives a flattened sequence of vectors and must reconstruct
    the flattened output sequence.
    """

    def __init__(self, size, seq_len=10, vec_len=8):
        self.size = size
        self.seq_len = seq_len
        self.vec_len = vec_len
        self.data = (torch.rand(size, seq_len, vec_len) < 0.5).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        total_len = 2 * self.seq_len + 1
        x = torch.zeros(total_len, self.vec_len)
        y = torch.zeros(total_len, self.vec_len)

        x[0 : self.seq_len, :] = self.data[idx]
        x[self.seq_len, -1] = 1.0  # Delimiter
        y[self.seq_len + 1 :, :] = self.data[idx]

        # The model expects a sequence of vectors
        return x, y

    def get_model_config_updates(self, model_name: str) -> dict:
        """
        Returns model-specific config updates for this dataset.
        """
        if model_name == "mlp":
            return {
                "input_size": (self.seq_len * 2 + 1) * self.vec_len,
                "output_size": (self.seq_len * 2 + 1) * self.vec_len,
            }
        elif model_name == "transformer":
            return {"input_size": self.vec_len, "output_size": self.vec_len}
        else:
            return {"input_size": self.vec_len}


class AssociativeRecallDataset(Dataset):
    """
    A dataset for the associative recall task.
    The model receives a flattened sequence and must reconstruct the flattened output.
    Sequences are padded to a fixed length.
    """

    def __init__(self, size, item_range=(3, 6), vec_len=8):
        self.size = size
        self.item_range = item_range
        self.max_items = item_range[1]
        self.data_len = vec_len - 2  # 2 bits are for markers
        assert self.data_len > 0, "vec_len must be > 2"

        self.data = []
        for _ in range(size):
            num_items = torch.randint(item_range[0], item_range[1] + 1, (1,)).item()
            items = (torch.rand(num_items, self.data_len) < 0.5).float()
            self.data.append(items)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        items = self.data[idx]
        num_items = items.shape[0]
        vec_len = self.data_len + 2

        # Pad sequences to max length for batching
        max_seq_len = self.max_items + 2
        x = torch.zeros(max_seq_len, vec_len)
        y = torch.zeros(max_seq_len, vec_len)

        item_vectors = torch.cat(
            [items, torch.ones(num_items, 1), torch.zeros(num_items, 1)], dim=1
        )
        x[:num_items] = item_vectors

        delimiter = torch.zeros(1, vec_len)
        delimiter[0, -1] = 1.0
        x[num_items] = delimiter

        query_idx = torch.randint(0, num_items - 1, (1,)).item()
        query_vec = item_vectors[query_idx]
        answer_vec = item_vectors[query_idx + 1]

        x[num_items + 1] = query_vec
        y[num_items + 1] = answer_vec

        return x, y

    def get_model_config_updates(self, model_name: str) -> dict:
        """
        Returns model-specific config updates for this dataset.
        """
        if model_name == "mlp":
            return {
                "input_size": (self.max_items + 2) * (self.data_len + 2),
                "output_size": (self.max_items + 2) * (self.data_len + 2),
            }
        elif model_name == "transformer":
            return {
                "input_size": self.data_len + 2,
                "output_size": self.data_len + 2,
            }
        else:
            return {"input_size": self.data_len + 2}
