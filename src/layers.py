import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    def __init__(self, hidden_size, expansion, bias=False):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias)

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
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype())
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, causal=False, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

    def forward(self, hidden_states, cos_sin: Tuple[torch.Tensor, torch.Tensor]):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cos_sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_sin)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output)

CosSin = Tuple[torch.Tensor, torch.Tensor]

class CastedEmbedding(nn.Embedding):
    """
    An nn.Embedding wrapper that casts the output to a specified dtype.
    This is to ensure faithfulness with the original implementation's mixed-precision setup.
    """
    def __init__(self, *args, cast_to: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.cast_to = cast_to

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).to(self.cast_to)


class CastedLinear(nn.Linear):
    """
    An nn.Linear wrapper that casts the input to a specified dtype.
    This is to ensure faithfulness with the original implementation's mixed-precision setup.
    """
    def __init__(self, *args, cast_to: torch.dtype = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cast_to = cast_to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input if a dtype is specified
        if self.cast_to:
             x = x.to(self.cast_to)
        return super().forward(x)


class CastedSparseEmbedding(nn.Embedding):
    """
    An nn.Embedding(sparse=True) wrapper that casts the output to a specified dtype.
    """
    def __init__(self, *args, cast_to: torch.dtype, **kwargs):
        # The original implementation seems to have some custom args like `batch_size` and `init_std`
        # which are not standard for nn.Embedding. I will ignore them for now and focus on the sparse aspect.
        kwargs.pop('batch_size', None)
        kwargs.pop('init_std', None)
        super().__init__(*args, sparse=True, **kwargs)
        self.cast_to = cast_to

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).to(self.cast_to)
