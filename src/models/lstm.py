import torch
import torch.nn as nn
from typing import Dict


class LSTM(nn.Module):
    """
    A standard LSTM model for sequence modeling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        **kwargs,
    ):
        """
        Initializes the LSTM model.

        Args:
            vocab_size: The size of the vocabulary.
            embedding_dim: The dimension of the token embeddings.
            hidden_size: The number of features in the hidden state.
            num_layers: The number of recurrent layers.
            output_size: The size of the output (e.g., vocab_size for LM).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the LSTM model.

        Args:
            batch: A dictionary containing the input tensor under the key 'inputs'.
                   Shape: (batch_size, seq_len).

        Returns:
            A dictionary containing the output logits under the key 'logits'.
            Shape: (batch_size, seq_len, output_size).
        """
        inputs = batch["inputs"]
        embedded = self.embedding(inputs)

        # LSTM returns output, (hidden, cell)
        lstm_out, _ = self.lstm(embedded)

        # We pass the entire sequence of hidden states to the linear layer
        logits = self.fc(lstm_out)

        return {"logits": logits}

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        This model does not use a carry in the same way as the HREM models,
        as the nn.LSTM module handles its own hidden state. This is a placeholder
        for compatibility with the Trainer, which expects this method.
        """
        # The LSTM's hidden state is managed internally by the nn.LSTM layer,
        # so we don't need to return an explicit carry state here.
        return ()
