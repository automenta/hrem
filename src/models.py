import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron baseline."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

class HRM(nn.Module):
    """The original Hierarchical Recurrent Memory model."""
    def __init__(self, seq_len, d_model, n_cycles, t_steps):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_cycles = n_cycles
        self.t_steps = t_steps
        self.f_I = nn.Linear(seq_len, d_model)
        self.f_L = nn.Linear(d_model * 3, d_model)
        self.f_H = nn.Linear(d_model * 2, d_model)
        self.f_O = nn.Linear(d_model, seq_len)

    def forward(self, x):
        x_tilde = self.f_I(x)
        z_H = torch.zeros(x.size(0), self.d_model, device=x.device)
        for n in range(self.n_cycles):
            z_L = torch.zeros(x.size(0), self.d_model, device=x.device)
            for t in range(self.t_steps):
                input_L = torch.cat([z_L, z_H, x_tilde], dim=1)
                z_L = torch.tanh(self.f_L(input_L))
            input_H = torch.cat([z_H, z_L], dim=1)
            z_H = torch.tanh(self.f_H(input_H))
        out = self.f_O(z_H)
        return F.sigmoid(out)

class HREM(nn.Module):
    """
    A modular Hierarchical Recurrent-External Memory model.
    The external memory component can be disabled for ablation studies.
    Includes optional DNC-style location-based addressing.
    """
    def __init__(self, seq_len, d_model, n_cycles, t_steps, use_memory=True,
                 m_loc=64, d_mem=32, top_k=4, sparse_addressing=True,
                 use_location_addressing=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_cycles = n_cycles
        self.t_steps = t_steps
        self.use_memory = use_memory
        self.use_location_addressing = use_location_addressing
        self.sparse_addressing = sparse_addressing

        # Core components
        self.f_I = nn.Linear(seq_len, d_model)
        self.f_H = nn.Linear(d_model * 2, d_model)
        self.f_O = nn.Linear(d_model, seq_len)

        if self.use_memory:
            self.m_loc = m_loc
            self.d_mem = d_mem
            self.top_k = top_k if sparse_addressing else m_loc
            self.init_proj = nn.Linear(seq_len, m_loc * d_mem)

            # Controller input size is constant
            input_dim_L = d_model * 3 + d_mem

            # Controller output size depends on features enabled
            # Base: z_L, read_key, read_beta, write_key, write_beta, erase, add
            output_dim_L = d_model + (4 * d_mem) + 2
            if self.use_location_addressing:
                # Add gates: write_gate, allocation_gate, read_modes (3)
                output_dim_L += 1 + 1 + 3

            self.f_L = nn.Linear(input_dim_L, output_dim_L)
        else:
            # If no memory, f_L is simpler, similar to HRM
            input_dim_L = d_model * 3
            output_dim_L = d_model
            self.f_L = nn.Linear(input_dim_L, output_dim_L)

    def forward(self, x):
        batch = x.size(0)
        device = x.device
        x_tilde = self.f_I(x)
        z_H = torch.zeros(batch, self.d_model, device=device)

        if self.use_memory:
            M = self.init_proj(x).reshape(batch, self.m_loc, self.d_mem)

        for n in range(self.n_cycles):
            z_L = torch.zeros(batch, self.d_model, device=device)
            r = torch.zeros(batch, self.d_mem, device=device) if self.use_memory else None

            # Initialize memory-related states for the cycle
            if self.use_memory:
                w_r_prev = torch.zeros(batch, self.m_loc, device=device)
                w_w_prev = torch.zeros(batch, self.m_loc, device=device)
                if self.use_location_addressing:
                    # Usage vector, precedence vector, and link matrix for DNC
                    usage = torch.zeros(batch, self.m_loc, device=device)
                    precedence = torch.zeros(batch, self.m_loc, device=device)
                    link_matrix = torch.zeros(batch, self.m_loc, self.m_loc, device=device)


            for t in range(self.t_steps):
                if self.use_memory:
                    input_L = torch.cat([z_L, z_H, x_tilde, r], dim=1)
                    out_L = self.f_L(input_L)

                    # Deconstruct the output of the low-level controller
                    idx = 0
                    z_L = torch.tanh(out_L[:, idx:idx+self.d_model]); idx += self.d_model

                    # Base interface vector for memory
                    read_key = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem
                    read_beta_raw = out_L[:, idx:idx+1]; idx += 1
                    write_key = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem
                    write_beta_raw = out_L[:, idx:idx+1]; idx += 1
                    erase = torch.sigmoid(out_L[:, idx:idx+self.d_mem]); idx += self.d_mem
                    add = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem

                    read_beta = F.softplus(read_beta_raw + 1)
                    write_beta = F.softplus(write_beta_raw + 1)

                    # Get content-based weightings
                    w_c_r, _ = self._content_addressing(read_key, read_beta, M)
                    w_c_w, _ = self._content_addressing(write_key, write_beta, M)

                    if self.use_location_addressing:
                        # Unpack DNC gates
                        write_gate = torch.sigmoid(out_L[:, idx:idx+1]); idx += 1
                        alloc_gate = torch.sigmoid(out_L[:, idx:idx+1]); idx += 1
                        read_modes = F.softmax(out_L[:, idx:idx+3], dim=1); idx += 3

                        # Placeholder for DNC logic
                        w_w = w_c_w # TODO: Implement allocation
                        w_r = w_c_r # TODO: Implement location-based read
                    else:
                        w_w = w_c_w
                        w_r = w_c_r

                    # Write to memory
                    erase_m = torch.einsum('bi,bj->bij', w_w, erase)
                    add_m = torch.einsum('bi,bj->bij', w_w, add)
                    M = M * (1 - erase_m) + add_m

                    # Read from memory
                    r = torch.einsum('bi,bij->bj', w_r, M)

                    # Update previous weightings
                    w_r_prev = w_r
                    w_w_prev = w_w
                else:
                    # No memory, so the inner loop is simpler
                    input_L = torch.cat([z_L, z_H, x_tilde], dim=1)
                    z_L = torch.tanh(self.f_L(input_L))

            input_H = torch.cat([z_H, z_L], dim=1)
            z_H = torch.tanh(self.f_H(input_H))

        out = self.f_O(z_H)
        return F.sigmoid(out)

    def _content_addressing(self, key, beta, M):
        # key: (batch, d_mem), beta: (batch, 1), M: (batch, m_loc, d_mem)
        sim = F.cosine_similarity(key.unsqueeze(1), M, dim=2)
        weighted_sim = sim * beta

        if self.sparse_addressing:
            # Top-K sparse addressing
            values, indices = torch.topk(weighted_sim, self.top_k, dim=1)
            sparse_w = F.softmax(values, dim=1)

            # Scatter back to full size
            w = torch.zeros_like(weighted_sim).scatter(1, indices, sparse_w)
            return w, indices # Returning indices for legacy compatibility
        else:
            # Dense addressing
            w = F.softmax(weighted_sim, dim=1)
            return w, None # No specific indices for dense
