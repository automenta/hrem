import torch
import torch.nn as nn
import math
from typing import Dict


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    """
    A standard Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        input_size: int,  # Vocab size
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        output_size: int,  # Vocab size
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initializes the Transformer model.

        Args:
            input_size: The size of the vocabulary.
            hidden_size: The number of features in the transformer layers (d_model).
            num_layers: The number of sub-encoder-layers in the encoder.
            num_heads: The number of heads in the multiheadattention models.
            output_size: The size of the vocabulary.
            dropout: The dropout value.
        """
        super().__init__()
        self.d_model = hidden_size
        self.input_embedding = nn.Embedding(input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,  # Important for compatibility with DataLoader
        )

        self.fc_out = nn.Linear(self.d_model, output_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Transformer model.

        Args:
            batch: A dictionary containing the input tensor under the key 'inputs'
                   and target tensor under the key 'targets'.
                   Shape: (batch_size, seq_len). Values are token indices.

        Returns:
            A dictionary containing the output logits under the key 'logits'.
            Shape: (batch_size, seq_len, output_size).
        """
        src = batch["inputs"]
        tgt = batch["targets"]

        # Embed and add positional encoding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Generate masks
        # The causal mask ensures that the prediction for position i can depend
        # only on the known outputs at positions less than i.
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            src.device
        )

        # Transformer forward pass
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Final linear layer
        logits = self.fc_out(output)

        return {"logits": logits}

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        Placeholder for compatibility with the Trainer.
        """
        return ()
