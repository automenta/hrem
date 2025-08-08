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
    """
    def __init__(self, seq_len, d_model, n_cycles, t_steps, use_memory=True, m_loc=64, d_mem=32, top_k=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_cycles = n_cycles
        self.t_steps = t_steps
        self.use_memory = use_memory

        # Core components
        self.f_I = nn.Linear(seq_len, d_model)
        self.f_H = nn.Linear(d_model * 2, d_model)
        self.f_O = nn.Linear(d_model, seq_len)

        if self.use_memory:
            self.m_loc = m_loc
            self.d_mem = d_mem
            self.top_k = top_k
            self.init_proj = nn.Linear(seq_len, m_loc * d_mem)
            input_dim_L = d_model * 3 + d_mem
            # z_L, write_key, write_beta, erase, add, read_key, read_beta
            output_dim_L = d_model + (4 * d_mem) + 2
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

            for t in range(self.t_steps):
                if self.use_memory:
                    input_L = torch.cat([z_L, z_H, x_tilde, r], dim=1)
                    out_L = self.f_L(input_L)

                    # Deconstruct the output of the low-level controller
                    idx = 0
                    z_L = torch.tanh(out_L[:, idx:idx+self.d_model])
                    idx += self.d_model
                    write_key = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem
                    write_beta_raw = out_L[:, idx:idx+1]; idx += 1
                    erase = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem
                    add = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem
                    read_key = out_L[:, idx:idx+self.d_mem]; idx += self.d_mem
                    read_beta_raw = out_L[:, idx:idx+1]

                    # Memory operations
                    write_beta = F.softplus(write_beta_raw + 1).squeeze(1)
                    read_beta = F.softplus(read_beta_raw + 1).squeeze(1)
                    erase = torch.sigmoid(erase)

                    w_w, indices_w = self._content_addressing(write_key, write_beta, M)
                    erase_m = torch.einsum('bk,bd->bkd', w_w, erase)
                    add_m = torch.einsum('bk,bd->bkd', w_w, add)

                    gathered_M_w = torch.gather(M, 1, indices_w.unsqueeze(2).expand(-1, -1, self.d_mem))
                    updated_M = gathered_M_w * (1 - erase_m) + add_m
                    M.scatter_(1, indices_w.unsqueeze(2).expand(-1, -1, self.d_mem), updated_M)

                    w_r, indices_r = self._content_addressing(read_key, read_beta, M)
                    gathered_M_r = torch.gather(M, 1, indices_r.unsqueeze(2).expand(-1, -1, self.d_mem))
                    r = torch.einsum('bk,bkd->bd', w_r, gathered_M_r)
                else:
                    # No memory, so the inner loop is simpler
                    input_L = torch.cat([z_L, z_H, x_tilde], dim=1)
                    z_L = torch.tanh(self.f_L(input_L))

            input_H = torch.cat([z_H, z_L], dim=1)
            z_H = torch.tanh(self.f_H(input_H))

        out = self.f_O(z_H)
        return F.sigmoid(out)

    def _content_addressing(self, key, beta, M):
        sim = F.cosine_similarity(key.unsqueeze(1), M, dim=2)
        weighted = beta.unsqueeze(1) * sim
        values, indices = torch.topk(weighted, self.top_k, dim=1)
        sparse_w = F.softmax(values, dim=1)
        return sparse_w, indices
