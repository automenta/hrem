import torch
import torch.nn as nn
from typing import Dict


class RNN(nn.Module):
    """
    A standard RNN model for sequence modeling.
    """

    def __init__(
        self,
        input_size_per_step: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        **kwargs,
    ):
        """
        Initializes the RNN model.

        Args:
            input_size_per_step: The number of features in the input.
            hidden_size: The number of features in the hidden state.
            num_layers: The number of recurrent layers.
            output_size: The size of the output.
        """
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size_per_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the RNN model.

        Args:
            batch: A dictionary containing the input tensor under the key 'inputs'.
                   Shape: (batch_size, seq_len, input_size_per_step).

        Returns:
            A dictionary containing the output logits under the key 'logits'.
            Shape: (batch_size, seq_len).
        """
        inputs = batch["inputs"]
        if len(inputs.shape) == 2:  # (batch_size, seq_len)
            inputs = inputs.unsqueeze(-1)  # (batch_size, seq_len, 1)

        inputs = inputs.float()

        # RNN returns output, hidden
        rnn_out, _ = self.rnn(inputs)

        # We pass the entire sequence of hidden states to the linear layer
        logits = self.fc(rnn_out).squeeze(-1)

        return {"logits": logits}

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        This model does not use a carry in the same way as the HREM models,
        as the nn.RNN module handles its own hidden state. This is a placeholder
        for compatibility with the Trainer, which expects this method.
        """
        # The RNN's hidden state is managed internally by the nn.RNN layer,
        # so we don't need to return an explicit carry state here.
        return ()
