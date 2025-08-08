import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ReverseDataset(Dataset):
    def __init__(self, size, seq_len):
        self.size = size
        self.seq_len = seq_len
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        x = torch.rand(self.seq_len) < 0.5
        x = x.float()
        y = x.flip(0)
        return x, y

class HRM(nn.Module):
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
    def __init__(self, seq_len, d_model, n_cycles, t_steps, m_loc, d_mem, top_k=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_cycles = n_cycles
        self.t_steps = t_steps
        self.m_loc = m_loc
        self.d_mem = d_mem
        self.top_k = top_k
        self.init_proj = nn.Linear(seq_len, m_loc * d_mem)
        self.f_I = nn.Linear(seq_len, d_model)
        input_dim_L = d_model * 3 + d_mem
        output_dim_L = d_model + 4 * d_mem + 2
        self.f_L = nn.Linear(input_dim_L, output_dim_L)
        self.f_H = nn.Linear(d_model * 2, d_model)
        self.f_O = nn.Linear(d_model, seq_len)
    def forward(self, x):
        batch = x.size(0)
        device = x.device
        x_tilde = self.f_I(x)
        z_H = torch.zeros(batch, self.d_model, device=device)
        M = self.init_proj(x).reshape(batch, self.m_loc, self.d_mem)
        for n in range(self.n_cycles):
            z_L = torch.zeros(batch, self.d_model, device=device)
            r = torch.zeros(batch, self.d_mem, device=device)
            for t in range(self.t_steps):
                input_L = torch.cat([z_L, z_H, x_tilde, r], dim=1)
                out_L = self.f_L(input_L)
                idx = 0
                z_L = torch.tanh(out_L[:, idx:idx+self.d_model])
                idx += self.d_model
                write_key = out_L[:, idx:idx+self.d_mem]
                idx += self.d_mem
                write_beta_raw = out_L[:, idx:idx+1]
                idx +=1
                erase = out_L[:, idx:idx+self.d_mem]
                idx += self.d_mem
                add = out_L[:, idx:idx+self.d_mem]
                idx += self.d_mem
                read_key = out_L[:, idx:idx+self.d_mem]
                idx += self.d_mem
                read_beta_raw = out_L[:, idx:idx+1]
                write_beta = F.softplus(write_beta_raw + 1).squeeze(1)  # sharper
                read_beta = F.softplus(read_beta_raw + 1).squeeze(1)  # sharper
                erase = F.sigmoid(erase)
                def content_addressing(key, beta, M):
                    sim = F.cosine_similarity(key.unsqueeze(1), M, dim=2)
                    weighted = beta.unsqueeze(1) * sim
                    values, indices = torch.topk(weighted, self.top_k, dim=1)
                    sparse_w = F.softmax(values, dim=1)
                    return sparse_w, indices
                w_w, indices_w = content_addressing(write_key, write_beta, M)
                erase_m = torch.einsum('bk,bd->bkd', w_w, erase)
                add_m = torch.einsum('bk,bd->bkd', w_w, add)
                gathered_M = torch.gather(M, 1, indices_w.unsqueeze(2).expand(-1, -1, self.d_mem))
                updated = gathered_M * (1 - erase_m) + add_m
                M.scatter_(1, indices_w.unsqueeze(2).expand(-1, -1, self.d_mem), updated)
                w_r, indices_r = content_addressing(read_key, read_beta, M)
                gathered_M_r = torch.gather(M, 1, indices_r.unsqueeze(2).expand(-1, -1, self.d_mem))
                r = torch.einsum('bk,bkd->bd', w_r, gathered_M_r)
            input_H = torch.cat([z_H, z_L], dim=1)
            z_H = torch.tanh(self.f_H(input_H))
        out = self.f_O(z_H)
        return F.sigmoid(out)

# Hyperparameters
seq_len = 64
n_cycles = 5
t_steps = 10
d_model_hrm = 80
d_model_hrem = 50
m_loc = 64
d_mem = 32
top_k = 4

# Datasets
train_ds = ReverseDataset(1000, seq_len)
test_ds = ReverseDataset(200, seq_len)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

def train_and_eval(model, name):
    print(f'\nTraining {name}')
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameter count: {param_count}')
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    epochs = 20
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0:
            print(f'Epoch {ep}, loss {loss.item():.4f}')
    model.eval()
    acc = 0
    n = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            acc += ((out > 0.5) == y).float().sum()
            n += y.numel()
    acc = acc / n
    print(f'Test accuracy for {name}: {acc:.4f}')

# Train and evaluate baseline HRM
model_hrm = HRM(seq_len, d_model_hrm, n_cycles, t_steps)
train_and_eval(model_hrm, 'HRM')

# Train and evaluate refined HREM
model_hrem = HREM(seq_len, d_model_hrem, n_cycles, t_steps, m_loc, d_mem, top_k)
train_and_eval(model_hrem, 'HREM')
