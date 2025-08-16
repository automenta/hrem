from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

# Assuming these layers are in src.layers
from ..layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
    CastedSparseEmbedding,
)


# This is a placeholder for a function that might be needed from the
# original repo's common.py
def trunc_normal_init_(*args, **kwargs):
    # In a real scenario, we'd copy the implementation of this function.
    # For now, we'll just use a standard normal init.
    return nn.init.normal_(*args, **kwargs)


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    # This config class is based on the hrm_act_v1.py file.
    # It will be populated from the JSON config files for experiments.
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: Optional[int] = None
    input_size: Optional[int] = None  # For vectorized inputs

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    use_memory: bool = False

    # HREM-specific memory parameters
    m_loc: Optional[int] = None
    d_mem: Optional[int] = None
    top_k: Optional[int] = None
    sparse_addressing: Optional[bool] = None
    use_location_addressing: Optional[bool] = None


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        dtype = getattr(torch, config.forward_dtype)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            causal=False,  # In the original, this is non-causal attention
            dtype=dtype,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size, expansion=config.expansion, dtype=dtype
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # This is a post-norm architecture
        # Self Attention
        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        norm_variance_epsilon = self.norm_eps
        hidden_states = rms_norm(
            hidden_states + attn_out, variance_epsilon=norm_variance_epsilon
        )
        # Fully Connected
        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(
            hidden_states + mlp_out, variance_epsilon=self.norm_eps
        )
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)

        if self.config.vocab_size:
            self.embed_tokens = CastedEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                cast_to=self.forward_dtype,
            )
            self.lm_head = CastedLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
                cast_to=self.forward_dtype,
            )
        else:
            # If no vocab_size, assume input is already vectorized. Use a linear projection.
            # This requires 'input_size' to be in the config.
            self.input_proj = CastedLinear(
                self.config.input_size,
                self.config.hidden_size,
                cast_to=self.forward_dtype,
            )
            self.output_proj = CastedLinear(
                self.config.hidden_size,
                self.config.input_size,
                bias=False,
                cast_to=self.forward_dtype,
            )

        self.q_head = nn.Linear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(
            self.config.puzzle_emb_ndim // -self.config.hidden_size
        )  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
                dtype=self.forward_dtype,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.H_layers)
            ]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.L_layers)
            ]
        )

        self.H_init = nn.Parameter(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            )
        )
        self.L_init = nn.Parameter(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            )
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, batch: Dict[str, torch.Tensor]):
        input_tensor = batch["inputs"].to(self.forward_dtype)
        if self.config.vocab_size:
            embedding = self.embed_tokens(input_tensor.to(torch.int32))
        else:
            embedding = self.input_proj(input_tensor)

        if self.config.puzzle_emb_ndim > 0:
            puzzle_identifiers = batch["puzzle_identifiers"]
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
    ):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        memory_readout: Optional[torch.Tensor] = None,
    ) -> Tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        z_H_full, z_L_full = carry.z_H, carry.z_L
        current_seq_len = batch["inputs"].shape[1] + self.puzzle_emb_len
        cos_sin = (
            self.rotary_emb(seq_len=current_seq_len)
            if hasattr(self, "rotary_emb")
            else None
        )
        seq_info = dict(cos_sin=cos_sin)

        # Slice the states to the current batch's sequence length
        z_H = z_H_full[:, :current_seq_len]
        z_L = z_L_full[:, :current_seq_len]

        input_embeddings = self._input_embeddings(batch)

        # Add memory readout to input injection if provided
        input_injection = z_H + input_embeddings
        if memory_readout is not None:
            input_injection = input_injection + memory_readout

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not (
                        (_H_step == self.config.H_cycles - 1)
                        and (_L_step == self.config.L_cycles - 1)
                    ):
                        z_L = self.L_level(z_L, input_injection, **seq_info)
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        z_L = self.L_level(z_L, input_injection, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Update the full-sized carry states
        new_z_H = z_H_full.clone()
        new_z_L = z_L_full.clone()
        new_z_H[:, :current_seq_len] = z_H
        new_z_L[:, :current_seq_len] = z_L

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=new_z_H.detach(), z_L=new_z_L.detach()
        )

        if self.config.vocab_size:
            output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        else:
            # For vectorized tasks, we might apply a sigmoid for reconstruction tasks
            output = torch.sigmoid(self.output_proj(z_H))

        q_logits = self.q_head(z_H[:, 0].to(torch.float32))

        return new_carry, output.to(torch.float32), (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
        memory_readout: Optional[torch.Tensor] = None,
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, memory_readout=memory_readout
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Re-run inner forward to get next Q-values for the target
                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry, new_current_data, memory_readout=memory_readout
                )[-1]
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return (
            HierarchicalReasoningModel_ACTV1Carry(
                new_inner_carry, new_steps, halted, new_current_data
            ),
            outputs,
        )
