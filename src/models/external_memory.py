import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalMemory(nn.Module):
    """
    A Differentiable Neural Computer (DNC) memory module.

    This implementation includes content-based addressing, allocation,
    and temporal linking.
    """

    def __init__(
        self,
        d_model,
        m_loc,
        d_mem,
        top_k,
        sparse_addressing,
        use_location_addressing,
        forward_dtype="float32",
    ):
        super().__init__()
        self.m_loc = m_loc
        self.d_mem = d_mem
        self.top_k = top_k if sparse_addressing else m_loc
        self.sparse_addressing = sparse_addressing
        self.use_location_addressing = use_location_addressing
        self.dtype = getattr(torch, forward_dtype)

        # Controller to produce the interface vector from the model's hidden state
        # Base: read_key, read_beta, write_key, write_beta, erase, add
        output_dim_L = (4 * d_mem) + 2
        if self.use_location_addressing:
            # Add gates for DNC: write, alloc, read_modes (3)
            output_dim_L += 1 + 1 + 3
        self.memory_controller = nn.Linear(d_model, output_dim_L, dtype=self.dtype)

    def forward(self, z, M_prev, states):
        # Unpack previous states
        w_r_prev = states["w_r_prev"]
        w_w_prev = states["w_w_prev"]
        usage = states["usage"]
        precedence = states["precedence"]
        link_matrix = states["link_matrix"]

        # 1. Produce interface vector from controller
        interface_vec = self.memory_controller(z)
        idx = 0

        # Base NTM interface
        read_key = interface_vec[:, idx : idx + self.d_mem]
        idx += self.d_mem
        read_beta = F.softplus(interface_vec[:, idx : idx + 1] + 1)
        idx += 1
        write_key = interface_vec[:, idx : idx + self.d_mem]
        idx += self.d_mem
        write_beta = F.softplus(interface_vec[:, idx : idx + 1] + 1)
        idx += 1
        erase = torch.sigmoid(interface_vec[:, idx : idx + self.d_mem])
        idx += self.d_mem
        add = interface_vec[:, idx : idx + self.d_mem]
        idx += self.d_mem

        # 2. Compute content-based addressing
        w_c_r = self._content_addressing(read_key, read_beta, M_prev)
        w_c_w = self._content_addressing(write_key, write_beta, M_prev)

        if self.use_location_addressing:
            # DNC gates
            write_gate = torch.sigmoid(interface_vec[:, idx : idx + 1])
            idx += 1
            alloc_gate = torch.sigmoid(interface_vec[:, idx : idx + 1])
            idx += 1
            read_modes = F.softmax(interface_vec[:, idx : idx + 3], dim=1)
            idx += 3

            # 3. Compute allocation and write weightings
            # Update usage vector
            usage = usage + w_w_prev - usage * w_w_prev

            # Allocation weighting is based on usage
            free_list = 1 - usage
            alloc_weights = free_list / (
                torch.sum(free_list, dim=1, keepdim=True) + 1e-8
            )

            # Final write weighting is a mix of content and allocation
            w_w = write_gate * (alloc_gate * alloc_weights + (1 - alloc_gate) * w_c_w)

            # 4. Update temporal links and precedence
            precedence_prev = precedence
            precedence = (1 - torch.sum(w_w, dim=1, keepdim=True)) * precedence + w_w

            link_matrix_update = torch.einsum("bi,bj->bij", w_w, precedence_prev)
            link_matrix = (
                1 - w_w.unsqueeze(2) - w_w.unsqueeze(1)
            ) * link_matrix + link_matrix_update
            link_matrix.diagonal(dim1=-2, dim2=-1).zero_()  # No self-loops

            # 5. Compute read weighting
            forward_w = torch.bmm(link_matrix, w_r_prev.unsqueeze(2)).squeeze(2)
            backward_w = torch.bmm(
                link_matrix.transpose(1, 2), w_r_prev.unsqueeze(2)
            ).squeeze(2)

            w_r = (
                read_modes[:, 0].unsqueeze(1) * backward_w
                + read_modes[:, 1].unsqueeze(1) * w_c_r
                + read_modes[:, 2].unsqueeze(1) * forward_w
            )
        else:
            # If not using location addressing, it's a simpler NTM
            w_w = w_c_w
            w_r = w_c_r

        # 6. Write to memory
        erase_m = torch.einsum("bi,bj->bij", w_w, erase)
        add_m = torch.einsum("bi,bj->bij", w_w, add)
        M = M_prev * (1 - erase_m) + add_m

        # 7. Read from memory
        r = torch.einsum("bi,bij->bj", w_r, M)

        # 8. Pack new states
        new_states = {
            "w_r_prev": w_r,
            "w_w_prev": w_w,
            "usage": usage,
            "precedence": precedence,
            "link_matrix": link_matrix,
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
        M = torch.zeros(
            batch_size, self.m_loc, self.d_mem, device=device, dtype=self.dtype
        )
        states = {
            "w_r_prev": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "w_w_prev": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "usage": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "precedence": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "link_matrix": torch.zeros(
                batch_size, self.m_loc, self.m_loc, device=device, dtype=self.dtype
            ),
        }
        return M, states
