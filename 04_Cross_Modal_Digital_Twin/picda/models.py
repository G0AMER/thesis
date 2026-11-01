"""Skeleton backbones for single-person input of shape (N, C, T, V).

CTRGCN follows the official implementation of Chen et al., "Channel-wise Topology
Refinement Graph Convolution for Skeleton-Based Action Recognition", ICCV 2021
(10 TCN-GCN units, 64-128-256 channels, multi-scale temporal convolution, CTRGC with
three graph subsets). The legacy notebooks used a simplified variant that should not be
reported as CTR-GCN. STGCN follows Yan et al., AAAI 2018, with learnable edge importance.

Both expose: forward(x) -> (logits, features) and `feature_dim`. CTRGCN can additionally
record its refined topologies (channel-averaged, per layer and subset) for the PI-CDA
topology-consistency loss.
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, (kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_c)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, stride=1, dilations=(1, 2), residual=True):
        super().__init__()
        n_br = len(dilations) + 2
        assert out_c % n_br == 0, "out channels must be divisible by number of branches"
        bc = out_c // n_br
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
                          TemporalConv(bc, bc, kernel_size, stride, d))
            for d in dilations
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_c, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)), nn.BatchNorm2d(bc)))
        self.branches.append(nn.Sequential(nn.Conv2d(in_c, bc, 1, stride=(stride, 1)), nn.BatchNorm2d(bc)))
        if not residual:
            self.residual = None
        elif in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_c, out_c, 1, stride)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        if self.residual is not None:
            out = out + self.residual(x)
        return out


class CTRGC(nn.Module):
    def __init__(self, in_c, out_c, rel_reduction=8):
        super().__init__()
        rel_c = 8 if in_c < 16 else in_c // rel_reduction
        self.conv1 = nn.Conv2d(in_c, rel_c, 1)
        self.conv2 = nn.Conv2d(in_c, rel_c, 1)
        self.conv3 = nn.Conv2d(in_c, out_c, 1)
        self.conv4 = nn.Conv2d(rel_c, out_c, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)

    def forward(self, x, A, alpha):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        rel = torch.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))          # N, R, V, V
        topo = self.conv4(rel) * alpha + A.unsqueeze(0).unsqueeze(0)    # N, C, V, V
        out = torch.einsum("ncuv,nctv->nctu", topo, x3)
        return out, topo


class UnitGCN(nn.Module):
    def __init__(self, in_c, out_c, A: np.ndarray, residual=True):
        super().__init__()
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList([CTRGC(in_c, out_c) for _ in range(self.num_subset)])
        if not residual:
            self.down = None
        elif in_c != out_c:
            self.down = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))
        else:
            self.down = nn.Identity()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        self.record_topology = False
        self.topologies: List[torch.Tensor] = []

    def forward(self, x):
        y = None
        for i in range(self.num_subset):
            z, topo = self.convs[i](x, self.PA[i], self.alpha)
            y = z if y is None else y + z
            if self.record_topology:
                self.topologies.append(topo.mean(1))  # N, V, V
        y = self.bn(y)
        if self.down is not None:
            y = y + self.down(x)
        return self.relu(y)


class TCNGCNUnit(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, residual=True, kernel_size=5, dilations=(1, 2)):
        super().__init__()
        self.gcn1 = UnitGCN(in_c, out_c, A)
        self.tcn1 = MultiScaleTemporalConv(out_c, out_c, kernel_size, stride, dilations, residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = None
        elif in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_c, out_c, 1, stride)

    def forward(self, x):
        y = self.tcn1(self.gcn1(x))
        if self.residual is not None:
            y = y + self.residual(x)
        return self.relu(y)


class CTRGCN(nn.Module):
    def __init__(self, num_classes, A: np.ndarray, in_channels=3, base=64, dropout=0.0):
        super().__init__()
        V = A.shape[-1]
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        b = base
        self.layers = nn.ModuleList([
            TCNGCNUnit(in_channels, b, A, residual=False),
            TCNGCNUnit(b, b, A), TCNGCNUnit(b, b, A), TCNGCNUnit(b, b, A),
            TCNGCNUnit(b, 2 * b, A, stride=2), TCNGCNUnit(2 * b, 2 * b, A), TCNGCNUnit(2 * b, 2 * b, A),
            TCNGCNUnit(2 * b, 4 * b, A, stride=2), TCNGCNUnit(4 * b, 4 * b, A), TCNGCNUnit(4 * b, 4 * b, A),
        ])
        self.feature_dim = 4 * b
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_classes))
        bn_init(self.data_bn, 1)

    def _gcn_units(self):
        return [m for m in self.modules() if isinstance(m, UnitGCN)]

    def set_record_topology(self, flag: bool):
        for u in self._gcn_units():
            u.record_topology = flag
            u.topologies = []

    def pop_topologies(self) -> List[torch.Tensor]:
        out = []
        for u in self._gcn_units():
            out.extend(u.topologies)
            u.topologies = []
        return out

    def extract_features(self, x):
        N, Cc, T, V = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, V * Cc, T)
        x = self.data_bn(x)
        x = x.view(N, V, Cc, T).permute(0, 2, 3, 1).contiguous()
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        f = self.extract_features(x)
        return self.fc(self.drop(f)), f


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A: np.ndarray, stride=1, residual=True, dropout=0.0):
        super().__init__()
        self.K = A.shape[0]
        self.register_buffer("A", torch.from_numpy(A.astype(np.float32)))
        self.edge_importance = nn.Parameter(torch.ones_like(self.A))
        self.gcn = nn.Conv2d(in_c, out_c * self.K, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_c), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = None
        elif in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_c, out_c, 1, (stride, 1)), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, _, T, V = x.shape
        y = self.gcn(x).view(N, self.K, -1, T, V)
        y = torch.einsum("nkctv,kwv->nctw", y, self.A * self.edge_importance)
        y = self.tcn(y)
        if self.residual is not None:
            y = y + self.residual(x)
        return self.relu(y)


class STGCN(nn.Module):
    def __init__(self, num_classes, A: np.ndarray, in_channels=3, base=64, dropout=0.0):
        super().__init__()
        V = A.shape[-1]
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        b = base
        cfg = [(in_channels, b, 1, False), (b, b, 1, True), (b, b, 1, True), (b, b, 1, True),
               (b, 2 * b, 2, True), (2 * b, 2 * b, 1, True), (2 * b, 2 * b, 1, True),
               (2 * b, 4 * b, 2, True), (4 * b, 4 * b, 1, True), (4 * b, 4 * b, 1, True)]
        self.layers = nn.ModuleList([STGCNBlock(i, o, A, s, r, dropout=0.0 if not r else dropout)
                                     for i, o, s, r in cfg])
        self.feature_dim = 4 * b
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x):
        N, Cc, T, V = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, V * Cc, T)
        x = self.data_bn(x)
        x = x.view(N, V, Cc, T).permute(0, 2, 3, 1).contiguous()
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        f = self.extract_features(x)
        return self.fc(f), f


def build_model(name: str, num_classes: int, A: np.ndarray, in_channels: int, dropout: float = 0.0):
    if name == "ctrgcn":
        return CTRGCN(num_classes, A, in_channels, dropout=dropout)
    if name == "stgcn":
        return STGCN(num_classes, A, in_channels, dropout=dropout)
    raise ValueError(name)
