import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# Helper Modules from the original implementation

def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, int(hidden_size * expansion))
        self.w2 = nn.Linear(hidden_size, int(hidden_size * expansion))
        self.w3 = nn.Linear(int(hidden_size * expansion), hidden_size)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, cos, sin):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

class HRMBlock(nn.Module):
    def __init__(self, hidden_size, expansion, num_heads):
        super().__init__()
        self.self_attn = Attention(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = 1e-5

    def forward(self, hidden_states, cos, sin):
        # Pre-Norm
        normalized_hidden_states = rms_norm(hidden_states, eps=self.norm_eps)
        attn_output = self.self_attn(hidden_states=normalized_hidden_states, cos=cos, sin=sin)
        hidden_states = hidden_states + attn_output

        normalized_hidden_states = rms_norm(hidden_states, eps=self.norm_eps)
        mlp_output = self.mlp(normalized_hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states

class HRM(nn.Module):
    def __init__(self, vocab_size, hidden_size, h_layers, l_layers, num_heads, expansion, seq_len, h_cycles, l_cycles, use_act=False, halt_max_steps=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.num_heads = num_heads
        self.expansion = expansion
        self.seq_len = seq_len
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.use_act = use_act
        self.halt_max_steps = halt_max_steps

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads, max_position_embeddings=seq_len)

        self.H_level = nn.ModuleList([HRMBlock(hidden_size, expansion, num_heads) for _ in range(h_layers)])
        self.L_level = nn.ModuleList([HRMBlock(hidden_size, expansion, num_heads) for _ in range(l_layers)])

        self.H_init = nn.Parameter(torch.randn(hidden_size))
        self.L_init = nn.Parameter(torch.randn(hidden_size))

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if self.use_act:
            self.q_head = nn.Linear(hidden_size, 1)
            self.halt_threshold = 0.9

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        input_embeddings = self.embed_tokens(x)
        cos, sin = self.rotary_emb(x, seq_len)

        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        if self.use_act:
            halting_probability = torch.zeros(batch_size, seq_len, device=device)
            remainders = torch.ones(batch_size, seq_len, device=device)
            n_updates = torch.zeros(batch_size, seq_len, device=device)

            total_z_H = torch.zeros_like(z_H)

            for step in range(self.halt_max_steps):
                z_L_input = z_L + z_H + input_embeddings
                for layer in self.L_level:
                    z_L = layer(z_L_input, cos, sin)

                z_H_input = z_H + z_L
                for layer in self.H_level:
                    z_H = layer(z_H_input, cos, sin)

                p = torch.sigmoid(self.q_head(z_H)).squeeze(-1)

                still_running = (halting_probability < self.halt_threshold).float()

                new_halted = (halting_probability + p * remainders > self.halt_threshold) * still_running
                still_running = (halting_probability < self.halt_threshold).float()

                halting_probability = halting_probability + p * remainders
                remainders = remainders * (1 - p)

                total_z_H += z_H * new_halted.unsqueeze(-1)
                n_updates += still_running

                if (still_running == 0).all():
                    break

            z_H = total_z_H / (n_updates.unsqueeze(-1) + 1e-7)

        else:
            for _ in range(self.h_cycles):
                for _ in range(self.l_cycles):
                    z_L_input = z_L + z_H + input_embeddings
                    for layer in self.L_level:
                        z_L = layer(z_L_input, cos, sin)

                z_H_input = z_H + z_L
                for layer in self.H_level:
                    z_H = layer(z_H_input, cos, sin)

        logits = self.lm_head(z_H)
        return logits

class MLP(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(seq_len * hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, seq_len * vocab_size)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embed_tokens(x)
        x = x.view(batch_size, -1)
        logits = self.mlp(x)
        return logits.view(batch_size, seq_len, self.vocab_size)

# The old HREM and ReverseDataset are removed.

class CopyTaskDataset(Dataset):
    def __init__(self, size, seq_len, vocab_size):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # sequence of random integers
        x = torch.randint(1, self.vocab_size, (self.seq_len,))
        # target is the same as input
        y = x.clone()
        return x, y

class HREM(HRM):
    def __init__(self, vocab_size, hidden_size, h_layers, l_layers, num_heads, expansion, seq_len, h_cycles, l_cycles, use_act=False, halt_max_steps=10, m_loc=128, d_mem=64, top_k=4):
        super().__init__(vocab_size, hidden_size, h_layers, l_layers, num_heads, expansion, seq_len, h_cycles, l_cycles, use_act, halt_max_steps)
        self.m_loc = m_loc
        self.d_mem = d_mem
        self.top_k = top_k

        # Memory and controller components
        self.init_proj = nn.Linear(hidden_size, m_loc * d_mem)

        # Adjust L_level input dimension to include memory read-out
        input_dim_L = hidden_size + d_mem

        # Controller to produce memory interface vectors
        self.controller = nn.Linear(hidden_size, 4 * d_mem + 2) # write_key, write_beta, erase, add, read_key, read_beta
        self.mem_to_hidden = nn.Linear(d_mem, hidden_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        input_embeddings = self.embed_tokens(x)
        cos, sin = self.rotary_emb(x, seq_len)

        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        M = torch.randn(batch_size, self.m_loc, self.d_mem, device=device)

        for _ in range(self.h_cycles):
            for _ in range(self.l_cycles):
                # Use mean of z_L to generate memory interface vectors
                z_L_summary = z_L.mean(dim=1)
                interface_vectors = self.controller(z_L_summary)

                idx = 0
                write_key = interface_vectors[:, idx:idx+self.d_mem]
                idx += self.d_mem
                write_beta = F.softplus(interface_vectors[:, idx:idx+1])
                idx +=1
                erase = torch.sigmoid(interface_vectors[:, idx:idx+self.d_mem])
                idx += self.d_mem
                add = interface_vectors[:, idx:idx+self.d_mem]
                idx += self.d_mem
                read_key = interface_vectors[:, idx:idx+self.d_mem]
                idx += self.d_mem
                read_beta = F.softplus(interface_vectors[:, idx:idx+1])
                idx +=1

                # Write to memory
                w_w_sim = F.cosine_similarity(write_key.unsqueeze(1), M, dim=2)
                w_w = F.softmax(write_beta * w_w_sim, dim=-1)

                erase_m = torch.einsum('bm,bd->bmd', w_w, erase)
                add_m = torch.einsum('bm,bd->bmd', w_w, add)
                M = M * (1 - erase_m) + add_m

                # Read from memory
                w_r_sim = F.cosine_similarity(read_key.unsqueeze(1), M, dim=2)
                w_r = F.softmax(read_beta * w_r_sim, dim=-1)
                r = torch.einsum('bm,bmd->bd', w_r, M)

                # Project memory readout to hidden size and inject into z_L_input
                r_proj = self.mem_to_hidden(r)
                z_L_input = z_L + z_H + input_embeddings + r_proj.unsqueeze(1)
                for layer in self.L_level:
                    z_L = layer(z_L_input, cos, sin)

            z_H_input = z_H + z_L
            for layer in self.H_level:
                z_H = layer(z_H_input, cos, sin)

        logits = self.lm_head(z_H)
        return logits

# Hyperparameters
vocab_size = 16
seq_len = 32
hidden_size = 128
h_layers = 2
l_layers = 2
num_heads = 4
expansion = 2.0
h_cycles = 3
l_cycles = 5
m_loc = 64
d_mem = 32

# Datasets
train_ds = CopyTaskDataset(1000, seq_len, vocab_size)
test_ds = CopyTaskDataset(200, seq_len, vocab_size)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

def train_and_eval(model, name):
    print(f'\nTraining {name}')
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameter count: {param_count}')
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 1 # smoke test
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            out = model(x)
            loss = loss_fn(out.view(-1, vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Epoch {ep}, loss {loss.item():.4f}')
    model.eval()
    acc = 0
    n = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = out.argmax(dim=-1)
            acc += (preds == y).float().sum()
            n += y.numel()
    acc = acc / n
    print(f'Test accuracy for {name}: {acc:.4f}')

# Train and evaluate models
model_mlp = MLP(vocab_size, hidden_size, seq_len)
train_and_eval(model_mlp, 'MLP')

model_hrm = HRM(vocab_size, hidden_size, h_layers, l_layers, num_heads, expansion, seq_len, h_cycles, l_cycles)
train_and_eval(model_hrm, 'HRM')

model_hrem = HREM(vocab_size, hidden_size, h_layers, l_layers, num_heads, expansion, seq_len, h_cycles, l_cycles, m_loc=m_loc, d_mem=d_mem)
train_and_eval(model_hrem, 'HREM')
