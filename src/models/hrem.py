import torch
import torch.nn as nn
from typing import Dict, Tuple

from .hrm import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Carry
from .external_memory import ExternalMemory

class HREM(HierarchicalReasoningModel_ACTV1):
    """
    Hierarchical Recurrent-External Memory (HREM) model.
    This model extends the Hierarchical Reasoning Model (HRM) by inheriting from it
    and augmenting its forward pass with an external DNC/NTM memory module.
    """
    def __init__(self, config_dict: dict):
        # Pass the config to the parent HRM class
        super().__init__(config_dict)

        self.use_memory = self.config.use_memory
        if not self.use_memory:
            # If not using memory, this class is identical to HRM.
            return

        # Memory-specific initializations
        self.memory = ExternalMemory(
            d_model=self.config.hidden_size,
            m_loc=config_dict.get('m_loc', 64),
            d_mem=config_dict.get('d_mem', 32),
            top_k=config_dict.get('top_k', 4),
            sparse_addressing=config_dict.get('sparse_addressing', True),
            use_location_addressing=config_dict.get('use_location_addressing', True)
        )
        self.memory_readout_proj = nn.Linear(config_dict.get('d_mem', 32), self.config.hidden_size)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict]:
        """Initializes the carry state for both HRM and the external memory."""
        # Get the standard HRM carry from the parent
        hrm_carry = super().initial_carry(batch)

        mem_states = {}
        if self.use_memory:
            # Initialize memory states
            M, mem_states = self.memory.init_memory(batch['inputs'].size(0), batch['inputs'].device)
            mem_states['M'] = M

        return hrm_carry, mem_states

    def forward(self, carry: Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict], batch: Dict[str, torch.Tensor]) -> Tuple[Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict], Dict[str, torch.Tensor]]:
        """
        Overrides the HRM forward pass to inject memory operations.
        """
        if not self.use_memory:
            # If no memory, just call the parent's forward pass.
            # We need to adjust the carry format.
            hrm_carry, _ = carry
            new_hrm_carry, outputs = super().forward(hrm_carry, batch)
            return (new_hrm_carry, {}), outputs

        hrm_carry, mem_states = carry

        # We need the current z_H to generate the memory interface vector.
        # It's reset inside the parent's forward pass, so we need to peek into the carry.
        # Let's reset it here to get the correct initial state.
        current_inner_carry = self.inner.reset_carry(hrm_carry.halted, hrm_carry.inner_carry)
        z_H_summary = current_inner_carry.z_H.mean(dim=1)

        # Run the memory controller
        M_prev = mem_states['M']
        prev_mem_states = {k: v for k, v in mem_states.items() if k != 'M'}
        M, r, new_mem_states_inner = self.memory(z_H_summary, M_prev, prev_mem_states)

        # Update memory states for the next step
        mem_states.update(new_mem_states_inner)
        mem_states['M'] = M

        # Project readout to be injected into the HRM
        memory_readout = self.memory_readout_proj(r).unsqueeze(1)

        # Call the parent's forward method, passing the memory readout.
        # The parent class will handle the entire ACT loop.
        new_hrm_carry, outputs = super().forward(hrm_carry, batch, memory_readout=memory_readout)

        # Package the new carries and outputs
        return (new_hrm_carry, mem_states), outputs
