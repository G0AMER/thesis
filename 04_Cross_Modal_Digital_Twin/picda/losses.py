"""Domain-alignment losses. All are computed in float32 on pooled features."""
from typing import List

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


def grad_reverse(x, coeff: float):
    return _GradReverse.apply(x, coeff)


class DomainDiscriminator(nn.Module):
    """DANN head (Ganin et al., JMLR 2016)."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


def mmd_loss(fs: torch.Tensor, ft: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5):
    """Multi-kernel Gaussian MMD^2 with a data-driven bandwidth (DAN, Long et al., ICML 2015)."""
    fs, ft = fs.float(), ft.float()
    ns = fs.size(0)
    total = torch.cat([fs, ft], dim=0)
    d2 = torch.cdist(total, total).pow(2)
    n = total.size(0)
    bandwidth = d2.detach().sum() / max(n * n - n, 1)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    K = sum(torch.exp(-d2 / (bandwidth * kernel_mul ** i + 1e-8)) for i in range(kernel_num))
    return K[:ns, :ns].mean() + K[ns:, ns:].mean() - 2 * K[:ns, ns:].mean()


def coral_loss(fs: torch.Tensor, ft: torch.Tensor):
    """Deep CORAL (Sun & Saenko, ECCV-W 2016)."""
    fs, ft = fs.float(), ft.float()
    d = fs.size(1)

    def cov(x):
        xm = x - x.mean(0, keepdim=True)
        return xm.t() @ xm / max(x.size(0) - 1, 1)

    return ((cov(fs) - cov(ft)) ** 2).sum() / (4 * d * d)


def topology_consistency_loss(topos: List[torch.Tensor], n_source: int):
    """Mean over layers/subsets of ||mean_batch(M_s) - mean_batch(M_t)||_F^2.

    `topos` are channel-averaged refined topologies (N, V, V) recorded on a forward pass of
    the concatenated [source; target] batch; the first `n_source` rows belong to source.
    """
    if not topos:
        raise RuntimeError("No topologies recorded; call model.set_record_topology(True) first")
    terms = [((t[:n_source].float().mean(0) - t[n_source:].float().mean(0)) ** 2).sum() for t in topos]
    return torch.stack(terms).mean()
