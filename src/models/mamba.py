import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        return self.out_proj(y)

    def ssm(self, x):
        (d_inner, d_state) = self.A_log.shape

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[1, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(torch.einsum("b l d, d n -> b l d n", delta, A))
        deltaB_u = torch.einsum("b l d, b l n, b l d -> b l d n", delta, B, u)

        h = torch.zeros(b, d_in, n, device=deltaA.device)
        ys = []

        for i in range(l):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = torch.einsum("b d n, b n -> b d", h, C[:, i, :])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D

        return y

    def step(self, x, cache):
        h, conv_state = cache

        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x
        x = torch.sum(
            conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        x += self.conv1d.bias
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(
            split_size=[1, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        h = h * torch.exp(A * delta.transpose(1, 0)) + (
            B * x.unsqueeze(-1) * delta.transpose(1, 0)
        )
        y = torch.einsum("b d n, b n -> b d", h, C)
        y = y + x * D

        return y, (h, conv_state)
