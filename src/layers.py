import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def rms_norm(hidden_states, variance_epsilon=1e-5):
    """
    A functional implementation of Root Mean Square Layer Normalization.
    This is based on the usage in the original hrm_act_v1.py, which suggests
    a parameter-less, functional application.
    """
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward layer.
    As seen in Llama, PaLM, and other models.
    """

    def __init__(self, hidden_size, expansion, bias=False, dtype=torch.float32):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos_sin):
    """Applies rotary position embedding to q and k."""
    cos, sin = cos_sin
    # cos, sin: [1, 1, seq_len, dim]
    # q, k: [bs, num_heads, seq_len, head_dim]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device
        )

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(self.dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(self.dtype), persistent=False
        )

    def forward(self, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        causal=False,
        bias=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal

        self.q_proj = nn.Linear(
            hidden_size, num_heads * head_dim, bias=bias, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, num_heads * head_dim, bias=bias, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, num_heads * head_dim, bias=bias, dtype=dtype
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim, hidden_size, bias=bias, dtype=dtype
        )

    def forward(self, hidden_states, cos_sin: Tuple[torch.Tensor, torch.Tensor]):
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if cos_sin is not None:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos_sin
            )

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        return self.o_proj(attn_output)


CosSin = Tuple[torch.Tensor, torch.Tensor]


class CastedEmbedding(nn.Embedding):
    """
    An nn.Embedding wrapper that casts the weights to a specified dtype upon initialization.
    """

    def __init__(self, *args, cast_to: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.to(cast_to))

    def forward(self, *args, **kwargs):
        # The output will have the correct dtype because the weights have been cast.
        return super().forward(*args, **kwargs)


class CastedLinear(nn.Linear):
    """
    An nn.Linear wrapper that casts its parameters to a specified dtype.
    """

    def __init__(self, *args, cast_to: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.to(cast_to))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(cast_to))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input x is expected to have the correct dtype.
        return super().forward(x)


class CastedSparseEmbedding(nn.Embedding):
    """
    An nn.Embedding(sparse=True) wrapper that casts the weights to a specified dtype.
    """

    def __init__(self, *args, cast_to: torch.dtype, **kwargs):
        kwargs.pop("batch_size", None)
        kwargs.pop("init_std", None)
        super().__init__(*args, sparse=True, **kwargs)
        self.weight = nn.Parameter(self.weight.to(cast_to))

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
