import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron baseline."""
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The input for MLP is expected to be flattened
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

class ExternalMemory(nn.Module):
    """
    A Differentiable Neural Computer (DNC) memory module.
    This implementation includes content-based addressing, allocation, and temporal linking.
    """
    def __init__(self, d_model, m_loc, d_mem, top_k, sparse_addressing, use_location_addressing):
        super().__init__()
        self.m_loc = m_loc
        self.d_mem = d_mem
        self.top_k = top_k if sparse_addressing else m_loc
        self.sparse_addressing = sparse_addressing
        self.use_location_addressing = use_location_addressing

        # Controller to produce the interface vector from the model's hidden state
        # Base: read_key, read_beta, write_key, write_beta, erase, add
        output_dim_L = (4 * d_mem) + 2
        if self.use_location_addressing:
            # Add gates for DNC: write_gate, allocation_gate, read_modes (3 gates)
            output_dim_L += 1 + 1 + 3
        self.memory_controller = nn.Linear(d_model, output_dim_L)

    def forward(self, z, M_prev, states):
        batch_size = z.size(0)
        device = z.device

        # Unpack previous states
        w_r_prev = states['w_r_prev']
        w_w_prev = states['w_w_prev']
        usage = states['usage']
        precedence = states['precedence']
        link_matrix = states['link_matrix']

        # 1. Produce interface vector from controller
        interface_vec = self.memory_controller(z)
        idx = 0

        # Base NTM interface
        read_key = interface_vec[:, idx:idx+self.d_mem]; idx += self.d_mem
        read_beta = F.softplus(interface_vec[:, idx:idx+1] + 1); idx += 1
        write_key = interface_vec[:, idx:idx+self.d_mem]; idx += self.d_mem
        write_beta = F.softplus(interface_vec[:, idx:idx+1] + 1); idx += 1
        erase = torch.sigmoid(interface_vec[:, idx:idx+self.d_mem]); idx += self.d_mem
        add = interface_vec[:, idx:idx+self.d_mem]; idx += self.d_mem

        # 2. Compute content-based addressing
        w_c_r = self._content_addressing(read_key, read_beta, M_prev)
        w_c_w = self._content_addressing(write_key, write_beta, M_prev)

        if self.use_location_addressing:
            # DNC gates
            write_gate = torch.sigmoid(interface_vec[:, idx:idx+1]); idx += 1
            alloc_gate = torch.sigmoid(interface_vec[:, idx:idx+1]); idx += 1
            read_modes = F.softmax(interface_vec[:, idx:idx+3], dim=1); idx += 3

            # 3. Compute allocation and write weightings
            # Update usage vector
            usage = (usage + w_w_prev - usage * w_w_prev)

            # Allocation weighting is based on usage
            free_list = 1 - usage
            alloc_weights = free_list / (torch.sum(free_list, dim=1, keepdim=True) + 1e-8)

            # Final write weighting is a mix of content and allocation
            w_w = write_gate * (alloc_gate * alloc_weights + (1 - alloc_gate) * w_c_w)

            # 4. Update temporal links and precedence
            precedence_prev = precedence
            precedence = (1 - torch.sum(w_w, dim=1, keepdim=True)) * precedence + w_w

            link_matrix_update = torch.einsum('bi,bj->bij', w_w, precedence_prev)
            link_matrix = (1 - w_w.unsqueeze(2) - w_w.unsqueeze(1)) * link_matrix + link_matrix_update
            link_matrix.diagonal(dim1=-2, dim2=-1).zero_() # No self-loops

            # 5. Compute read weighting
            forward_w = torch.bmm(link_matrix, w_r_prev.unsqueeze(2)).squeeze(2)
            backward_w = torch.bmm(link_matrix.transpose(1, 2), w_r_prev.unsqueeze(2)).squeeze(2)

            w_r = (read_modes[:, 0].unsqueeze(1) * backward_w +
                   read_modes[:, 1].unsqueeze(1) * w_c_r +
                   read_modes[:, 2].unsqueeze(1) * forward_w)
        else:
            # If not using location addressing, it's a simpler NTM
            w_w = w_c_w
            w_r = w_c_r

        # 6. Write to memory
        erase_m = torch.einsum('bi,bj->bij', w_w, erase)
        add_m = torch.einsum('bi,bj->bij', w_w, add)
        M = M_prev * (1 - erase_m) + add_m

        # 7. Read from memory
        r = torch.einsum('bi,bij->bj', w_r, M)

        # 8. Pack new states
        new_states = {
            'w_r_prev': w_r,
            'w_w_prev': w_w,
            'usage': usage,
            'precedence': precedence,
            'link_matrix': link_matrix
        }
        return M, r, new_states

    def _content_addressing(self, key, beta, M):
        # key: (batch, d_mem), beta: (batch, 1), M: (batch, m_loc, d_mem)
        sim = F.cosine_similarity(key.unsqueeze(1), M, dim=2)
        weighted_sim = sim * beta

        if self.sparse_addressing:
            values, indices = torch.topk(weighted_sim, self.top_k, dim=1)
            sparse_w = F.softmax(values, dim=1)
            w = torch.zeros_like(weighted_sim).scatter(1, indices, sparse_w)
        else:
            w = F.softmax(weighted_sim, dim=1)
        return w

    def init_memory(self, batch_size, device):
        M = torch.zeros(batch_size, self.m_loc, self.d_mem, device=device)
        states = {
            'w_r_prev': torch.zeros(batch_size, self.m_loc, device=device),
            'w_w_prev': torch.zeros(batch_size, self.m_loc, device=device),
            'usage': torch.zeros(batch_size, self.m_loc, device=device),
            'precedence': torch.zeros(batch_size, self.m_loc, device=device),
            'link_matrix': torch.zeros(batch_size, self.m_loc, self.m_loc, device=device)
        }
        return M, states


class HREM(nn.Module):
    """
    A Hierarchical Recurrent-External Memory model, with a transformer-based architecture.
    - If `use_memory` is False, it acts as a standard transformer encoder (our new HRM baseline).
    - If `use_memory` is True, it uses an external DNC/NTM memory module.
    """
    def __init__(self, input_size, d_model, n_layers, n_heads, n_cycles,
                 use_memory=False, m_loc=64, d_mem=32, top_k=4, sparse_addressing=True,
                 use_location_addressing=False, vocab_size=None, **kwargs):
        super().__init__()
        self.use_memory = use_memory
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_cycles = n_cycles

        # Input projection
        if self.vocab_size:
            self.embedding = nn.Embedding(self.vocab_size, d_model)
        else:
            # For non-language tasks, input_size is the feature dimension
            self.input_proj = nn.Linear(input_size, d_model)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        if self.vocab_size:
            self.output_proj = nn.Linear(d_model, self.vocab_size)
        else:
            self.output_proj = nn.Linear(d_model, input_size)

        # External memory module
        if self.use_memory:
            self.memory = ExternalMemory(d_model, m_loc, d_mem, top_k, sparse_addressing, use_location_addressing)
            # Project memory read-out to d_model to be added to the representation
            self.memory_readout_proj = nn.Linear(d_mem, d_model)


    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device

        # 1. Input Processing
        if self.vocab_size:
            # For language tasks, x is (batch, seq_len) of indices
            embedded_x = self.embedding(x) * math.sqrt(self.d_model)
        else:
            # For other tasks, x is (batch, seq_len, features)
            embedded_x = self.input_proj(x)

        # Initialize memory for the sequence
        if self.use_memory:
            M, mem_states = self.memory.init_memory(batch_size, device)

        # 2. Recurrent Processing Cycles
        # The "z" here is the sequence representation, analogous to z_H in the original paper
        z = torch.zeros_like(embedded_x)

        for _ in range(self.n_cycles):
            # Create a global summary of the state for the memory controller
            z_summary = z.mean(dim=1) # Simple averaging for now

            if self.use_memory:
                M, r, mem_states = self.memory(z_summary, M, mem_states)
                readout = self.memory_readout_proj(r).unsqueeze(1) # (batch, 1, d_model)
            else:
                readout = 0

            # Transformer processing
            # Input is the original input + current recurrent state + memory readout
            transformer_input = embedded_x + z + readout
            z = self.transformer_encoder(transformer_input)

        # 3. Output
        if self.vocab_size:
            # Output is (batch, seq_len, vocab_size) for language modeling
            output = self.output_proj(z)
        else:
            # Output is (batch, seq_len, features) for reconstruction tasks
            output = torch.sigmoid(self.output_proj(z))

        return output
