#!/usr/bin/env python3
# Generator for NB09: CPG-Net -- Contextual Policy Gating Network
# A novel continual learning architecture for shared autonomy
#
# Usage: python create_nb09.py
# Generates: 09_cpg_net_experiment.ipynb

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.0'},
})

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ============================================================
# CELL 1: Title & Motivation
# ============================================================
md(r"""# NB09 -- CPG-Net: Contextual Policy Gating Network

## A Novel Architecture for Continual Learning in Shared Autonomy

### Motivation

NB08 showed that **replay-based CL methods hit a ceiling** in our shared autonomy domain:
- DER-SA (best replay method): ACC = 0.6185, closing only ~20% of the gap to Joint Training (0.6442)
- ARW alone captures 94% of the total DER-SA improvement -- FKD and JAC contribute minimally
- The fundamental issue: a **single shared-weight MLP** forces all participants' policies into the same weight space

### The Core Insight

Different participants exhibit **different control strategies** for the same task. A shared-weight network suffers
representational conflict: learning participant B's mapping partially overwrites participant A's.

**Replay slows this overwriting but cannot prevent it** -- the architecture itself is the bottleneck.

### CPG-Net: Key Idea

Instead of fighting weight interference with replay, **learn which neurons to activate per-task**:

1. **Shared Backbone**: All tasks share the same feature extractor (enables positive transfer)
2. **Context Module**: A lightweight network that observes input statistics and generates soft gating masks
3. **Gated Computation**: Element-wise gating on hidden activations -- each task uses a different subnetwork
4. **Progressive Freezing**: After training on task $t$, freeze the gates and important weights for that task

**This is NOT PackNet** (hard binary masks via pruning), **NOT Progressive Networks** (separate columns),
and **NOT Mixture of Experts** (routing to separate experts). CPG-Net uses **learned soft gates** that
discover optimal neuron sharing/isolation patterns per-task.

### Metrics

Same 6 standard CL metrics as NB08:
- **ACC** = Average Accuracy (mean R2 on all tasks after final training)
- **F** = Forgetting (max previous accuracy minus final, averaged)
- **BWT** = Backward Transfer
- **FWT** = Forward Transfer
- **Memory** = Extra memory usage (MB)
- **Time** = Training time (seconds)
""")

# ============================================================
# CELL 2: Imports
# ============================================================
code(r"""import os, sys, abc, copy, json, logging, random, time, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 120})

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING, format='%(message)s')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'PyTorch {torch.__version__} -- device: {DEVICE}')
if DEVICE == 'cuda': print(f'  GPU: {torch.cuda.get_device_name()}')

PROJECT_ROOT = "/home/g0amer/Desktop/thesis/phd_project/thesis_project"
HARMONIC_DIR = f'{PROJECT_ROOT}/data/processed/harmonic'
RESULTS_DIR  = f'{PROJECT_ROOT}/experiments/exp07_cpg_net'
os.makedirs(RESULTS_DIR, exist_ok=True)

participants = sorted([d for d in os.listdir(HARMONIC_DIR)
                       if os.path.isdir(os.path.join(HARMONIC_DIR, d)) and d.startswith('p')])
print(f'Available participants: {len(participants)} -- {participants}')
print(f'Results: {RESULTS_DIR}')
""")

# ============================================================
# CELL 3: CLMetrics + TaskData (identical to NB08)
# ============================================================
code(r"""# ================================================================
# Continual Learning Metrics: ACC, F, BWT, FWT, Memory, Time
# (Identical to NB08 for fair comparison)
# ================================================================

@dataclass
class TaskResult:
    task_id: int; loss: float; mse: float; mae: float; r2: float
    n_samples: int = 0
    per_dim_mse: list = field(default_factory=list)
    ss_res: float = 0.0
    ss_tot: float = 0.0


class CLMetrics:
    # Tracks the R2 accuracy matrix a[i][j] = R2 on task j after training up to task i
    def __init__(self):
        self.R = []       # accuracy matrix (R2): R[i][j]
        self.MSE = []     # mse matrix
        self.per_dim = [] # per-dim MSE matrix
        self._last_results = []
        self.task_names = []
        self._random_baselines = []  # b_j for FWT

    def set_random_baselines(self, baselines):
        self._random_baselines = baselines

    def record(self, trained_up_to, task_results):
        r2_row = [r.r2 for r in task_results]
        mse_row = [r.mse for r in task_results]
        pdim_row = [r.per_dim_mse for r in task_results]
        while len(self.R) <= trained_up_to:
            self.R.append([])
            self.MSE.append([])
            self.per_dim.append([])
        self.R[trained_up_to] = r2_row
        self.MSE[trained_up_to] = mse_row
        self.per_dim[trained_up_to] = pdim_row
        self._last_results = task_results

    @property
    def T(self):
        return len(self.R)

    @property
    def ACC(self):
        if self.T == 0: return float('nan')
        last = self.R[-1]
        return float(np.mean(last)) if last else float('nan')

    @property
    def forgetting(self):
        T = self.T
        if T < 2: return 0.0
        fgt_sum = 0.0
        for j in range(T - 1):
            max_prev = max(self.R[l][j] for l in range(T - 1)
                          if j < len(self.R[l]))
            final = self.R[T - 1][j] if j < len(self.R[T - 1]) else 0.0
            fgt_sum += max(0.0, max_prev - final)
        return fgt_sum / (T - 1)

    @property
    def BWT(self):
        T = self.T
        if T < 2: return 0.0
        bwt = 0.0
        cnt = 0
        for j in range(T - 1):
            if j < len(self.R[T - 1]) and j < len(self.R[j]):
                bwt += self.R[T - 1][j] - self.R[j][j]
                cnt += 1
        return bwt / cnt if cnt > 0 else 0.0

    @property
    def FWT(self):
        T = self.T
        if T < 2: return 0.0
        fwt = 0.0
        cnt = 0
        for j in range(1, T):
            if j < len(self.R[j - 1]):
                b_j = self._random_baselines[j] if j < len(self._random_baselines) else 0.0
                fwt += self.R[j - 1][j] - b_j
                cnt += 1
        return fwt / cnt if cnt > 0 else 0.0

    @property
    def overall_r2(self):
        if not self._last_results: return float('nan')
        ss_res = sum(r.ss_res for r in self._last_results)
        ss_tot = sum(r.ss_tot for r in self._last_results)
        return 1 - ss_res / max(ss_tot, 1e-8)

    def summary_dict(self):
        return {
            'ACC': self.ACC, 'F': self.forgetting,
            'BWT': self.BWT, 'FWT': self.FWT,
            'overall_r2': self.overall_r2,
            'r2_matrix': self.R, 'mse_matrix': self.MSE,
            'per_dim_matrix': self.per_dim,
            'task_names': self.task_names,
        }


@dataclass
class TaskData:
    task_id: int; participant_id: str
    obs_train: torch.Tensor; act_train: torch.Tensor
    obs_val: torch.Tensor; act_val: torch.Tensor
    obs_test: torch.Tensor; act_test: torch.Tensor

    @property
    def n_train(self): return self.obs_train.shape[0]
    @property
    def n_val(self): return self.obs_val.shape[0]
    @property
    def n_test(self): return self.obs_test.shape[0]
    def train_loader(self, bs=256):
        return DataLoader(TensorDataset(self.obs_train, self.act_train), batch_size=bs, shuffle=True)
    def val_loader(self, bs=512):
        return DataLoader(TensorDataset(self.obs_val, self.act_val), batch_size=bs)
    def test_loader(self, bs=512):
        return DataLoader(TensorDataset(self.obs_test, self.act_test), batch_size=bs)

print('CLMetrics (ACC, F, BWT, FWT) & TaskData defined')
""")

# ============================================================
# CELL 4: Baseline models + CPG-Net
# ============================================================
code(r"""# ================================================================
# Models & CL Strategies (Baselines + CPG-Net)
# ================================================================

# ---------- Standard MLP (for baselines) ----------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256), dropout=0.1):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, act_dim)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self._feat_dim = d

    def forward(self, x):
        return self.head(self.backbone(x))

    def forward_with_features(self, x):
        feat = self.backbone(x)
        return self.head(feat), feat


# ================================================================
# CPG-Net: Contextual Policy Gating Network (NOVEL ARCHITECTURE)
# ================================================================
#
# Architecture overview:
#
#   Input (obs) ---> [Shared Backbone Layer 1] --(*)--> [Shared Backbone Layer 2] --(*)--> [Head] --> Output
#                                                 |                                   |
#                                          gate_1(task_id)                     gate_2(task_id)
#
# Each gate is a learned vector per task, applied element-wise (sigmoid gating).
# The context module computes task-specific statistics from input data to initialize gates.
# After training on task t, the gates for task t are frozen (progressive protection).
#
# Key novelty: Gates are SOFT (sigmoid, continuous 0..1), not binary.
# This allows partial neuron sharing across tasks -- neurons important for
# multiple participants remain shared, while task-specific neurons are isolated.

class ContextGateGenerator(nn.Module):
    # Generates soft gating masks from task-level input statistics.
    # Input: running mean and variance of observations for the current task
    # Output: sigmoid gate vectors for each hidden layer
    def __init__(self, obs_dim, hidden_dims, gate_hidden=64):
        super().__init__()
        # Context features: concatenation of mean and variance of input
        ctx_dim = obs_dim * 2  # [mean, var]
        self.gate_nets = nn.ModuleList()
        for h_dim in hidden_dims:
            net = nn.Sequential(
                nn.Linear(ctx_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, h_dim),
                # No sigmoid here -- we apply it in forward to allow gradient flow
            )
            # Initialize bias to positive values so gates start open (near 1 after sigmoid)
            nn.init.constant_(net[-1].bias, 2.0)
            self.gate_nets.append(net)

    def forward(self, ctx_features):
        # ctx_features: [1, obs_dim*2] = [mean, var]
        gates = []
        for net in self.gate_nets:
            g = torch.sigmoid(net(ctx_features))  # [1, h_dim], values in (0,1)
            gates.append(g)
        return gates


class CPGNet(nn.Module):
    # Contextual Policy Gating Network
    #
    # Shared backbone with per-task soft gating masks.
    # The context module generates initial gates from task statistics,
    # then gates are fine-tuned during training.
    # After each task, its gates are frozen.
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), dropout=0.1,
                 max_tasks=20, gate_hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = list(hidden)
        self.max_tasks = max_tasks
        self._n_layers = len(hidden)

        # Shared backbone layers (NOT Sequential -- we need to interleave gates)
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        d = obs_dim
        for h in hidden:
            self.layers.append(nn.Linear(d, h))
            self.dropouts.append(nn.Dropout(dropout))
            d = h

        # Shared output head
        self.head = nn.Linear(d, act_dim)

        # Context gate generator
        self.ctx_gen = ContextGateGenerator(obs_dim, hidden, gate_hidden)

        # Per-task gate storage: task_id -> list of gate tensors
        # These are nn.ParameterList so they can be fine-tuned
        self._task_gates = {}  # dict of task_id -> list of Parameter tensors
        self._frozen_tasks = set()

        # Running statistics per task (for context)
        self._task_stats = {}  # task_id -> (mean, var) tensors

    def register_task(self, task_id, obs_data):
        # Compute context features from task data and generate initial gates
        with torch.no_grad():
            obs_mean = obs_data.mean(0)
            obs_var = obs_data.var(0)
            ctx = torch.cat([obs_mean, obs_var]).unsqueeze(0).to(next(self.parameters()).device)
            init_gates = self.ctx_gen(ctx)

        # Store as trainable parameters
        gate_params = []
        for g in init_gates:
            # Store raw logit (pre-sigmoid) for training stability
            raw = torch.log(g.squeeze(0).clamp(1e-4, 1-1e-4) / (1 - g.squeeze(0).clamp(1e-4, 1-1e-4)))
            param = nn.Parameter(raw.detach().clone())
            gate_params.append(param)
        self._task_gates[task_id] = gate_params
        self._task_stats[task_id] = (obs_mean.detach(), obs_var.detach())

    def freeze_task(self, task_id):
        # Freeze gates for a completed task
        if task_id in self._task_gates:
            for p in self._task_gates[task_id]:
                p.requires_grad = False
            self._frozen_tasks.add(task_id)

    def get_gate_params(self, task_id):
        # Return trainable gate parameters for the current task
        if task_id in self._task_gates:
            return [p for p in self._task_gates[task_id] if p.requires_grad]
        return []

    def forward(self, x, task_id=0):
        if task_id not in self._task_gates:
            # Fallback: no gating (all ones)
            h = x
            for layer, drop in zip(self.layers, self.dropouts):
                h = drop(F.relu(layer(h)))
            return self.head(h)

        gates = self._task_gates[task_id]
        h = x
        for i, (layer, drop) in enumerate(zip(self.layers, self.dropouts)):
            h = F.relu(layer(h))
            # Apply soft gate: element-wise multiplication with sigmoid of stored logits
            gate_mask = torch.sigmoid(gates[i]).to(h.device)
            h = h * gate_mask.unsqueeze(0)  # broadcast over batch
            h = drop(h)
        return self.head(h)

    def forward_all_tasks(self, x, task_ids):
        # Forward for multiple tasks (used during importance computation)
        outputs = {}
        for tid in task_ids:
            outputs[tid] = self.forward(x, task_id=tid)
        return outputs


# ================================================================
# ContinualLearner base class (same as NB08)
# ================================================================

class ContinualLearner(abc.ABC):
    def __init__(self, model, lr=1e-3, weight_decay=0., device='cpu'):
        self.model = model.to(device); self.device = device; self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self._current_task = -1

    @abc.abstractmethod
    def on_task_start(self, task_id, train_loader): ...
    @abc.abstractmethod
    def compute_loss(self, obs, act, task_id): ...
    @abc.abstractmethod
    def on_task_end(self, task_id, train_loader): ...

    def train_task(self, task_id, train_loader, val_loader=None, epochs=50, patience=10, verbose=False):
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)
        self.model.train()
        best_val, best_state, no_improve = float('inf'), None, 0
        for epoch in range(epochs):
            eloss, nb = 0., 0
            for batch in train_loader:
                obs, act = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, act, task_id)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                eloss += loss.item(); nb += 1
            if val_loader is not None:
                vl = self._eval_loss(val_loader, task_id)
                if vl < best_val:
                    best_val = vl; best_state = copy.deepcopy(self.model.state_dict()); no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
        if best_state is not None: self.model.load_state_dict(best_state)
        self.on_task_end(task_id, train_loader)

    def _eval_loss(self, loader, task_id=0):
        self.model.eval()
        tot, n = 0., 0
        with torch.no_grad():
            for b in loader:
                o, a = b[0].to(self.device), b[1].to(self.device)
                if isinstance(self.model, CPGNet):
                    pred = self.model(o, task_id=task_id)
                else:
                    pred = self.model(o)
                tot += F.mse_loss(pred, a).item() * o.size(0); n += o.size(0)
        self.model.train()
        return tot / max(n, 1)

    def evaluate_task(self, test_loader, task_id):
        self.model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for b in test_loader:
                o, a = b[0].to(self.device), b[1].to(self.device)
                if isinstance(self.model, CPGNet):
                    preds.append(self.model(o, task_id=task_id).cpu())
                else:
                    preds.append(self.model(o).cpu())
                tgts.append(a.cpu())
        p, t = torch.cat(preds), torch.cat(tgts)
        mse = F.mse_loss(p, t).item()
        mae = (p - t).abs().mean().item()
        ss_res = ((t - p)**2).sum().item()
        ss_tot = ((t - t.mean(0))**2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-8)
        per_dim = ((p - t)**2).mean(0).tolist()
        self.model.train()
        return TaskResult(task_id, mse, mse, mae, r2, len(t), per_dim, ss_res, ss_tot)


# ================================================================
# Baseline strategies (same as NB08 for fair comparison)
# ================================================================

class NaiveFineTune(ContinualLearner):
    def on_task_start(self, t, l): pass
    def compute_loss(self, obs, act, t): return F.mse_loss(self.model(obs), act)
    def on_task_end(self, t, l): pass
    def memory_mb(self): return 0.0


class JointTraining(ContinualLearner):
    def __init__(self, model, lr=1e-3, device='cpu'):
        super().__init__(model, lr=lr, device=device)
        self._datasets = {}; self._init_state = copy.deepcopy(model.state_dict())
    def on_task_start(self, task_id, train_loader):
        all_o, all_a = [], []
        for b in train_loader: all_o.append(b[0]); all_a.append(b[1])
        self._datasets[task_id] = TensorDataset(torch.cat(all_o), torch.cat(all_a))
        self.model.load_state_dict(copy.deepcopy(self._init_state))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    def compute_loss(self, obs, act, t): return F.mse_loss(self.model(obs), act)
    def on_task_end(self, t, l): pass
    def train_task(self, task_id, train_loader, val_loader=None, epochs=50, patience=10, verbose=False):
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)
        joint_ds = ConcatDataset(list(self._datasets.values()))
        bs = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 256
        joint_loader = DataLoader(joint_ds, batch_size=bs, shuffle=True)
        self.model.train()
        super().train_task(task_id, joint_loader, val_loader, epochs, patience, verbose)
    def memory_mb(self):
        total = sum(t.tensors[0].nelement() + t.tensors[1].nelement() for t in self._datasets.values())
        return total * 4 / (1024**2)


class ReplayBuffer:
    def __init__(self, capacity=10000, store_features=False):
        self.capacity = capacity
        self.store_features = store_features
        self.obs, self.act, self.logits = [], [], []
        self.features = []
        self._n = 0
    def __len__(self): return len(self.obs)
    def add(self, obs, act, logits, features=None):
        for i in range(obs.size(0)):
            self._n += 1
            if len(self.obs) < self.capacity:
                self.obs.append(obs[i].cpu())
                self.act.append(act[i].cpu())
                self.logits.append(logits[i].cpu())
                if self.store_features and features is not None:
                    self.features.append(features[i].cpu())
            else:
                j = random.randint(0, self._n - 1)
                if j < self.capacity:
                    self.obs[j] = obs[i].cpu()
                    self.act[j] = act[i].cpu()
                    self.logits[j] = logits[i].cpu()
                    if self.store_features and features is not None:
                        self.features[j] = features[i].cpu()
    def sample(self, n):
        n = min(n, len(self.obs))
        idx = random.sample(range(len(self.obs)), n)
        result = (torch.stack([self.obs[i] for i in idx]),
                  torch.stack([self.act[i] for i in idx]),
                  torch.stack([self.logits[i] for i in idx]))
        if self.store_features and self.features:
            result = result + (torch.stack([self.features[i] for i in idx]),)
        return result
    def memory_mb(self):
        if len(self.obs) == 0: return 0.0
        per_sample = self.obs[0].nelement() + self.act[0].nelement() + self.logits[0].nelement()
        if self.store_features and self.features:
            per_sample += self.features[0].nelement()
        return len(self.obs) * per_sample * 4 / (1024**2)


class DERPlusPlus(ContinualLearner):
    def __init__(self, model, lr=1e-3, device='cpu', buffer_size=10000, alpha=0.5, beta=0.5):
        super().__init__(model, lr=lr, device=device)
        self.buffer = ReplayBuffer(buffer_size)
        self.alpha = alpha; self.beta = beta
    def on_task_start(self, t, l): pass
    def compute_loss(self, obs, act, task_id):
        pred = self.model(obs); loss = F.mse_loss(pred, act)
        with torch.no_grad(): logits = self.model(obs).detach()
        self.buffer.add(obs.detach(), act.detach(), logits)
        if len(self.buffer) > 0 and task_id > 0:
            bo, ba, bl = self.buffer.sample(max(1, obs.size(0)))
            bo, ba, bl = bo.to(self.device), ba.to(self.device), bl.to(self.device)
            bp = self.model(bo)
            loss = loss + self.alpha * F.mse_loss(bp, bl) + self.beta * F.mse_loss(bp, ba)
        return loss
    def on_task_end(self, t, l): pass
    def memory_mb(self): return self.buffer.memory_mb()


class EWC(ContinualLearner):
    def __init__(self, model, lr=1e-3, device='cpu', ewc_lambda=10000., fisher_samples=1000,
                 online=True, gamma=0.99):
        super().__init__(model, lr=lr, device=device)
        self.ewc_lambda = ewc_lambda; self.fisher_samples = fisher_samples
        self.online = online; self.gamma = gamma
        self._fishers = []; self._opt_params = []
        self._run_fisher = None; self._run_params = None
    def on_task_start(self, t, l): pass
    def compute_loss(self, obs, act, task_id):
        loss = F.mse_loss(self.model(obs), act)
        if task_id == 0: return loss
        return loss + self._penalty()
    def on_task_end(self, task_id, train_loader):
        fisher = self._compute_fisher(train_loader)
        if self.online:
            if self._run_fisher is None: self._run_fisher = fisher
            else:
                for n in self._run_fisher:
                    self._run_fisher[n] = self.gamma * self._run_fisher[n] + fisher[n]
            self._run_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        else:
            self._fishers.append(fisher)
            self._opt_params.append({n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad})
    def _compute_fisher(self, loader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval(); ns = 0
        for b in loader:
            o, a = b[0].to(self.device), b[1].to(self.device)
            self.model.zero_grad()
            F.mse_loss(self.model(o), a).backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * o.size(0)
            ns += o.size(0)
            if ns >= self.fisher_samples: break
        for n in fisher: fisher[n] /= max(ns, 1)
        self.model.train()
        return fisher
    def _penalty(self):
        pen = torch.tensor(0., device=self.device)
        if self.online and self._run_fisher is not None:
            for n, p in self.model.named_parameters():
                if n in self._run_fisher:
                    pen += (self._run_fisher[n] * (p - self._run_params[n]).pow(2)).sum()
        else:
            for fi, op in zip(self._fishers, self._opt_params):
                for n, p in self.model.named_parameters():
                    if n in fi: pen += (fi[n] * (p - op[n]).pow(2)).sum()
        return (self.ewc_lambda / 2.) * pen
    def memory_mb(self):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.online:
            return 2 * n_params * 4 / (1024**2) if self._run_fisher else 0.0
        else:
            return 2 * len(self._fishers) * n_params * 4 / (1024**2)


class AGEM(ContinualLearner):
    def __init__(self, model, lr=1e-3, device='cpu', mem_per_task=256):
        super().__init__(model, lr=lr, device=device)
        self._mem = {}; self._cap = mem_per_task
    def on_task_start(self, t, l): pass
    def compute_loss(self, obs, act, task_id): return F.mse_loss(self.model(obs), act)
    def on_task_end(self, task_id, train_loader):
        all_o, all_a = [], []
        for b in train_loader: all_o.append(b[0]); all_a.append(b[1])
        obs = torch.cat(all_o); act = torch.cat(all_a)
        if len(obs) > self._cap:
            idx = torch.randperm(len(obs))[:self._cap]
            obs, act = obs[idx], act[idx]
        self._mem[task_id] = (obs, act)
    def _get_ref(self):
        if not self._mem: return None, None
        return torch.cat([v[0] for v in self._mem.values()]), torch.cat([v[1] for v in self._mem.values()])
    def train_task(self, task_id, train_loader, val_loader=None, epochs=50, patience=10, verbose=False):
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)
        self.model.train()
        best_val, best_state, no_improve = float('inf'), None, 0
        for epoch in range(epochs):
            eloss, nb = 0., 0
            for batch in train_loader:
                obs, act = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, act, task_id)
                loss.backward()
                if task_id > 0:
                    g = torch.cat([p.grad.data.view(-1) for p in self.model.parameters() if p.grad is not None])
                    mem_o, mem_a = self._get_ref()
                    if mem_o is not None:
                        mem_o, mem_a = mem_o.to(self.device), mem_a.to(self.device)
                        self.optimizer.zero_grad()
                        ref_loss = F.mse_loss(self.model(mem_o), mem_a)
                        ref_loss.backward()
                        g_ref = torch.cat([p.grad.data.view(-1) for p in self.model.parameters() if p.grad is not None])
                        dot = (g * g_ref).sum()
                        if dot < 0:
                            g = g - (dot / (g_ref.norm()**2 + 1e-8)) * g_ref
                        offset = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                numel = p.grad.numel()
                                p.grad.data.copy_(g[offset:offset+numel].view_as(p.grad))
                                offset += numel
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                eloss += loss.item(); nb += 1
            if val_loader is not None:
                vl = self._eval_loss(val_loader)
                if vl < best_val:
                    best_val = vl; best_state = copy.deepcopy(self.model.state_dict()); no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
        if best_state is not None: self.model.load_state_dict(best_state)
        self.on_task_end(task_id, train_loader)
    def memory_mb(self):
        total = sum(v[0].nelement() + v[1].nelement() for v in self._mem.values())
        return total * 4 / (1024**2)


# ================================================================
# CPG-Net Learner (OUR NOVEL METHOD)
# ================================================================

class CPGNetLearner(ContinualLearner):
    # Continual learner using CPG-Net architecture.
    #
    # Training protocol per task t:
    # 1. Register task: compute context stats, generate initial gates
    # 2. Train: optimize backbone weights + gate params for task t
    #    - Backbone weights are shared but protected by importance weighting
    #    - Gate params are task-specific
    # 3. Freeze: lock gates for task t, compute weight importance
    #
    # Protection mechanism:
    # After each task, compute Fisher-like importance of backbone weights
    # When training new tasks, penalize changes to important weights (like EWC,
    # but targeted: only backbone weights used by previous tasks' gates)
    #
    # This combination of gating + importance weighting gives:
    # - Near-zero forgetting from gate isolation
    # - Positive transfer from shared backbone
    # - Capacity efficiency from soft (not hard) gates

    def __init__(self, model, lr=1e-3, device='cpu',
                 importance_lambda=5000., fisher_samples=2000,
                 gate_lr=5e-3, gate_sparsity=0.01):
        super().__init__(model, lr=lr, device=device)
        self.importance_lambda = importance_lambda
        self.fisher_samples = fisher_samples
        self.gate_lr = gate_lr
        self.gate_sparsity = gate_sparsity
        self._cum_importance = None  # cumulative importance (like online EWC)
        self._anchor_params = None   # anchor point (params after last task)
        self._completed_tasks = []

    def on_task_start(self, task_id, train_loader):
        # Collect all obs from train loader for context computation
        all_obs = []
        for b in train_loader:
            all_obs.append(b[0])
        all_obs = torch.cat(all_obs)

        # Register task: compute context, generate initial gates
        self.model.register_task(task_id, all_obs)

        # Build optimizer: backbone params + current task's gate params
        gate_params = self.model.get_gate_params(task_id)
        param_groups = [
            {'params': [p for p in self.model.parameters() if p.requires_grad and
                        not any(p is gp for gp in gate_params)],
             'lr': self.lr},
        ]
        if gate_params:
            param_groups.append({'params': gate_params, 'lr': self.gate_lr})
        self.optimizer = torch.optim.Adam(param_groups)

    def compute_loss(self, obs, act, task_id):
        pred = self.model(obs, task_id=task_id)
        loss = F.mse_loss(pred, act)

        # Importance-weighted regularization on backbone (like EWC)
        if self._cum_importance is not None and self._anchor_params is not None:
            reg = torch.tensor(0., device=self.device)
            for name, param in self.model.named_parameters():
                if name in self._cum_importance and param.requires_grad:
                    reg += (self._cum_importance[name] * (param - self._anchor_params[name]).pow(2)).sum()
            loss = loss + (self.importance_lambda / 2.) * reg

        # Gate sparsity: encourage gates to be sparse (close to 0 or 1)
        if task_id in self.model._task_gates:
            for gate_logit in self.model._task_gates[task_id]:
                if gate_logit.requires_grad:
                    g = torch.sigmoid(gate_logit)
                    # Entropy penalty: push gates toward 0 or 1
                    entropy = -(g * torch.log(g + 1e-8) + (1-g) * torch.log(1-g + 1e-8))
                    loss = loss + self.gate_sparsity * entropy.mean()

        return loss

    def on_task_end(self, task_id, train_loader):
        # 1. Freeze gates for this task
        self.model.freeze_task(task_id)
        self._completed_tasks.append(task_id)

        # 2. Compute importance of backbone weights (Fisher diagonal)
        importance = {}
        self.model.eval()
        ns = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'gate' not in name and '_task_gates' not in name:
                importance[name] = torch.zeros_like(param)

        for b in train_loader:
            o, a = b[0].to(self.device), b[1].to(self.device)
            self.model.zero_grad()
            pred = self.model(o, task_id=task_id)
            F.mse_loss(pred, a).backward()
            for name, param in self.model.named_parameters():
                if name in importance and param.grad is not None:
                    importance[name] += param.grad.data.pow(2) * o.size(0)
            ns += o.size(0)
            if ns >= self.fisher_samples:
                break
        for name in importance:
            importance[name] /= max(ns, 1)
        self.model.train()

        # 3. Accumulate importance (online EWC style)
        if self._cum_importance is None:
            self._cum_importance = importance
        else:
            for name in importance:
                if name in self._cum_importance:
                    self._cum_importance[name] = 0.9 * self._cum_importance[name] + importance[name]
                else:
                    self._cum_importance[name] = importance[name]

        # 4. Save anchor params
        self._anchor_params = {name: param.clone().detach()
                               for name, param in self.model.named_parameters()
                               if param.requires_grad}

    def train_task(self, task_id, train_loader, val_loader=None, epochs=50, patience=10, verbose=False):
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)
        self.model.train()
        best_val, best_state, no_improve = float('inf'), None, 0
        for epoch in range(epochs):
            eloss, nb = 0., 0
            for batch in train_loader:
                obs, act = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, act, task_id)
                loss.backward()

                # Clip gradients on all params
                all_params = list(self.model.parameters())
                gate_params = self.model.get_gate_params(task_id)
                for gp in gate_params:
                    all_params.append(gp)
                nn.utils.clip_grad_norm_(
                    [p for p in all_params if p.requires_grad and p.grad is not None], 1.0)
                self.optimizer.step()
                eloss += loss.item(); nb += 1
            if val_loader is not None:
                vl = self._eval_loss(val_loader, task_id)
                if vl < best_val:
                    best_val = vl
                    # Save model state + gate states
                    best_state = copy.deepcopy(self.model.state_dict())
                    # Also save gate parameters
                    if task_id in self.model._task_gates:
                        best_gate_data = [g.data.clone() for g in self.model._task_gates[task_id]]
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            # Restore best gate params
            if task_id in self.model._task_gates and 'best_gate_data' in dir():
                for g, gd in zip(self.model._task_gates[task_id], best_gate_data):
                    g.data.copy_(gd)
        self.on_task_end(task_id, train_loader)

    def memory_mb(self):
        # Memory = stored gate vectors + importance matrix + anchor params
        gate_mem = 0
        for tid, gates in self.model._task_gates.items():
            for g in gates:
                gate_mem += g.numel()
        # Importance + anchor: each = n_backbone_params floats
        n_backbone = sum(p.numel() for n, p in self.model.named_parameters()
                        if 'gate' not in n and '_task_gates' not in n)
        imp_mem = n_backbone if self._cum_importance is not None else 0
        anc_mem = n_backbone if self._anchor_params is not None else 0
        total_elements = gate_mem + imp_mem + anc_mem
        return total_elements * 4 / (1024**2)  # float32


# --- DER-SA from NB08 for comparison ---
class DERSA(ContinualLearner):
    def __init__(self, model, lr=1e-3, device='cpu',
                 buffer_size=10000, alpha=0.5, beta=0.5,
                 gamma_fkd=0.3, gamma_jac=0.2, tau=2.0):
        super().__init__(model, lr=lr, device=device)
        self.buffer = ReplayBuffer(buffer_size, store_features=True)
        self.alpha = alpha; self.beta = beta
        self.gamma_fkd = gamma_fkd; self.gamma_jac = gamma_jac; self.tau = tau
        self._ref_cov = None
    def on_task_start(self, task_id, train_loader): pass
    def on_task_end(self, task_id, train_loader):
        if len(self.buffer) > 100:
            sample = self.buffer.sample(min(len(self.buffer), 2000))
            bo = sample[0].to(self.device)
            with torch.no_grad():
                pred = self.model(bo)
            pred_centered = pred - pred.mean(0)
            self._ref_cov = (pred_centered.T @ pred_centered) / (pred.size(0) - 1)
    def compute_loss(self, obs, act, task_id):
        pred, feat = self.model.forward_with_features(obs)
        loss_current = F.mse_loss(pred, act)
        with torch.no_grad():
            logits_snap = self.model(obs).detach()
            _, feat_snap = self.model.forward_with_features(obs)
        self.buffer.add(obs.detach(), act.detach(), logits_snap, feat_snap.detach())
        if len(self.buffer) == 0 or task_id == 0: return loss_current
        sample = self.buffer.sample(max(1, obs.size(0)))
        bo, ba, bl, bf = sample[0].to(self.device), sample[1].to(self.device), \
                         sample[2].to(self.device), sample[3].to(self.device)
        bp, bp_feat = self.model.forward_with_features(bo)
        with torch.no_grad():
            drift = (bp - bl).pow(2).mean(dim=1)
            weights = torch.softmax(drift / self.tau, dim=0) * drift.size(0)
        loss_logit = (weights.unsqueeze(1) * (bp - bl).pow(2)).mean()
        loss_target = (weights.unsqueeze(1) * (bp - ba).pow(2)).mean()
        loss_fkd = F.mse_loss(bp_feat, bf)
        loss_jac = torch.tensor(0., device=self.device)
        if self._ref_cov is not None:
            bp_centered = bp - bp.mean(0)
            cur_cov = (bp_centered.T @ bp_centered) / (bp.size(0) - 1)
            loss_jac = F.mse_loss(cur_cov, self._ref_cov)
        return (loss_current + self.alpha * loss_logit + self.beta * loss_target
                + self.gamma_fkd * loss_fkd + self.gamma_jac * loss_jac)
    def memory_mb(self): return self.buffer.memory_mb()


print('All strategies defined: Naive, Joint, DER++, Online EWC, A-GEM, DER-SA, CPG-Net')
""")

# ============================================================
# CELL 5: Data loading (identical to NB08)
# ============================================================
code(r"""# Data Loading -- identical to previous experiments

JOY_COLS   = ['axes_x', 'axes_y', 'axes_z']
JPOS_COLS  = [f'mico_joint_{i}_pos' for i in range(1, 7)]
EE_COLS    = ['mico_end_effector_x', 'mico_end_effector_y', 'mico_end_effector_z']
JVEL_COLS  = [f'mico_joint_{i}_vel' for i in range(1, 7)]
OBS_COLS   = JOY_COLS + JPOS_COLS + EE_COLS
TARGET_COLS = JVEL_COLS
JOINT_POS_CLIP = 50.0

def load_participant(processed_dir, participant, val_ratio=0.15, seed=42):
    pdir = Path(processed_dir) / participant
    assert pdir.exists(), f'Not found: {pdir}'

    all_obs, all_tgt = [], []
    for trial_dir in sorted(pdir.iterdir()):
        if not trial_dir.is_dir():
            continue
        joy_path  = trial_dir / 'ada_joy.parquet'
        jpos_path = trial_dir / 'joint_positions.parquet'
        rpos_path = trial_dir / 'robot_position.parquet'
        if not all(p.exists() for p in [joy_path, jpos_path, rpos_path]):
            continue
        df_joy  = pd.read_parquet(joy_path)
        df_jpos = pd.read_parquet(jpos_path)
        df_rpos = pd.read_parquet(rpos_path)
        n = min(len(df_joy), len(df_jpos), len(df_rpos))
        if n < 10:
            continue
        for col_name in JPOS_COLS:
            if col_name in df_jpos.columns:
                mask = df_jpos[col_name].abs() > JOINT_POS_CLIP
                if mask.any():
                    median_val = df_jpos.loc[~mask, col_name].median()
                    df_jpos.loc[mask, col_name] = median_val
        obs_parts = []
        for cols, df in [(JOY_COLS, df_joy), (JPOS_COLS, df_jpos), (EE_COLS, df_rpos)]:
            available = [c for c in cols if c in df.columns]
            if len(available) != len(cols):
                break
            obs_parts.append(df[available].values[:n])
        else:
            tgt_available = [c for c in JVEL_COLS if c in df_jpos.columns]
            if len(tgt_available) != len(JVEL_COLS):
                continue
            tgt_arr = df_jpos[tgt_available].values[:n]
            obs_arr = np.concatenate(obs_parts, axis=1)
            valid = ~(np.isnan(obs_arr).any(1) | np.isnan(tgt_arr).any(1))
            obs_arr, tgt_arr = obs_arr[valid], tgt_arr[valid]
            if len(obs_arr) > 0:
                all_obs.append(obs_arr)
                all_tgt.append(tgt_arr)
            continue
        continue

    assert all_obs, f'No data for {participant}'
    obs = np.nan_to_num(np.concatenate(all_obs).astype(np.float32))
    tgt = np.nan_to_num(np.concatenate(all_tgt).astype(np.float32))

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(obs))
    nv = int(len(obs) * val_ratio)
    nt = nv
    obs_t, tgt_t = torch.from_numpy(obs), torch.from_numpy(tgt)
    return (obs_t[idx[nt+nv:]], tgt_t[idx[nt+nv:]],
            obs_t[idx[nt:nt+nv]], tgt_t[idx[nt:nt+nv]],
            obs_t[idx[:nt]],     tgt_t[idx[:nt]])

print(f'Data loading defined (outlier clipping at |val|>{JOINT_POS_CLIP})')
""")

# ============================================================
# CELL 6: Hyperparameters + Setup
# ============================================================
code(r"""# Hyperparameters & Task Setup

SEED = 42
HIDDEN   = (256, 256)
LR       = 1e-3
DROPOUT  = 0.1
EPOCHS   = 50
PATIENCE = 10
BS       = 256

# Baseline HPs (tuned in exp02)
EWC_LAMBDA   = 10000.0
EWC_FISHER_N = 1000
OEWC_GAMMA   = 0.99
DER_BUFFER   = 10000
AGEM_MEM     = 256

# DER-SA HPs (from NB08)
DERSA_BUFFER    = 10000
DERSA_ALPHA     = 0.5
DERSA_BETA      = 0.5
DERSA_GAMMA_FKD = 0.3
DERSA_GAMMA_JAC = 0.2
DERSA_TAU       = 2.0

# CPG-Net HPs
CPG_GATE_HIDDEN     = 64      # hidden dim of gate generator network
CPG_GATE_LR         = 5e-3    # learning rate for gate parameters
CPG_IMPORTANCE_LAM  = 5000.   # importance regularization strength
CPG_FISHER_N        = 2000    # samples for Fisher computation
CPG_GATE_SPARSITY   = 0.01    # gate sparsity penalty (entropy)

N_TASKS = 10
obs_dim = 12
act_dim = 6

def build_tasks(n_tasks, pids):
    selected = pids[:n_tasks]
    print(f'\nBuilding {n_tasks} tasks from: {selected}')
    task_list = []
    for tid, pid in enumerate(selected):
        try:
            data = load_participant(HARMONIC_DIR, pid, seed=SEED)
            td = TaskData(tid, pid, *data)
            task_list.append(td)
            print(f'  Task {tid:2d} ({pid}): train={td.n_train:5d}  val={td.n_val:4d}  test={td.n_test:4d}')
        except Exception as e:
            print(f'  SKIP {pid}: {e}')
    return task_list

def normalize_tasks(task_list):
    all_obs = torch.cat([t.obs_train for t in task_list])
    all_act = torch.cat([t.act_train for t in task_list])
    o_mean = all_obs.mean(0); o_std = all_obs.std(0).clamp(min=1e-6)
    a_mean = all_act.mean(0); a_std = all_act.std(0).clamp(min=1e-6)
    for t in task_list:
        t.obs_train = (t.obs_train - o_mean) / o_std
        t.obs_val   = (t.obs_val   - o_mean) / o_std
        t.obs_test  = (t.obs_test  - o_mean) / o_std
        t.act_train = (t.act_train - a_mean) / a_std
        t.act_val   = (t.act_val   - a_mean) / a_std
        t.act_test  = (t.act_test  - a_mean) / a_std
    return o_mean, o_std, a_mean, a_std

def make_model():
    return MLPPolicy(obs_dim, act_dim, HIDDEN, DROPOUT)

def make_cpg_model():
    return CPGNet(obs_dim, act_dim, HIDDEN, DROPOUT,
                  max_tasks=N_TASKS, gate_hidden=CPG_GATE_HIDDEN)

def run_strategy(name, learner, task_list, epochs=EPOCHS, patience=PATIENCE, bs=BS, verbose=True):
    # Compute random baselines for FWT
    torch.manual_seed(SEED)
    random_baselines = []
    if isinstance(learner.model, CPGNet):
        untrained = make_cpg_model().to(DEVICE)
    else:
        untrained = make_model().to(DEVICE)

    for t in task_list:
        untrained.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for b in t.test_loader(bs*2):
                o, a = b[0].to(DEVICE), b[1].to(DEVICE)
                if isinstance(untrained, CPGNet):
                    preds.append(untrained(o, task_id=t.task_id).cpu())
                else:
                    preds.append(untrained(o).cpu())
                tgts.append(a.cpu())
        p, tgt = torch.cat(preds), torch.cat(tgts)
        ss_res = ((tgt - p)**2).sum().item()
        ss_tot = ((tgt - tgt.mean(0))**2).sum().item()
        random_baselines.append(1 - ss_res / max(ss_tot, 1e-8))

    metrics = CLMetrics()
    metrics.task_names = [t.participant_id for t in task_list]
    metrics.set_random_baselines(random_baselines)

    t0 = time.time()
    for task in task_list:
        if verbose:
            print(f'  [{name}] Task {task.task_id}/{len(task_list)-1} ({task.participant_id})')
        learner.train_task(task.task_id, task.train_loader(bs),
                          task.val_loader(bs*2), epochs=epochs, patience=patience, verbose=False)
        results = [learner.evaluate_task(et.test_loader(bs*2), et.task_id) for et in task_list]
        metrics.record(task.task_id, results)
    elapsed = time.time() - t0

    mem = learner.memory_mb() if hasattr(learner, 'memory_mb') else 0.0

    return {
        'name': name, 'metrics': metrics, 'time': elapsed,
        'memory_mb': mem,
        'ACC': metrics.ACC, 'F': metrics.forgetting,
        'BWT': metrics.BWT, 'FWT': metrics.FWT,
        'overall_r2': metrics.overall_r2,
        'summary': metrics.summary_dict(),
    }

# Build and normalize
task_list = build_tasks(N_TASKS, participants)
norm_stats = normalize_tasks(task_list)

print(f'\nReady: {len(task_list)} tasks, {obs_dim}D --> {act_dim}D')
""")

# ============================================================
# CELL 7: Markdown header for comparison
# ============================================================
md(r"""---
## Run Full Comparison

All 7 strategies (6 baselines + CPG-Net) run sequentially. Each in its own cell.
Metrics: **ACC**, **F**, **BWT**, **FWT**, **Memory (MB)**, **Time (s)**
""")

# ============================================================
# CELLS 8-14: One per strategy
# ============================================================
strategies_code = {
    'Naive Fine-Tune': (
        'NaiveFineTune(make_model(), lr=LR, device=DEVICE)',
        'res_naive'
    ),
    'Joint Training': (
        'JointTraining(make_model(), lr=LR, device=DEVICE)',
        'res_joint'
    ),
    'Online EWC': (
        'EWC(make_model(), lr=LR, device=DEVICE,\n'
        '              ewc_lambda=EWC_LAMBDA, fisher_samples=EWC_FISHER_N,\n'
        '              online=True, gamma=OEWC_GAMMA)',
        'res_ewc'
    ),
    'A-GEM': (
        'AGEM(make_model(), lr=LR, device=DEVICE, mem_per_task=AGEM_MEM)',
        'res_agem'
    ),
    'DER++': (
        'DERPlusPlus(make_model(), lr=LR, device=DEVICE, buffer_size=DER_BUFFER)',
        'res_derpp'
    ),
    'DER-SA': (
        'DERSA(make_model(), lr=LR, device=DEVICE,\n'
        '                buffer_size=DERSA_BUFFER, alpha=DERSA_ALPHA, beta=DERSA_BETA,\n'
        '                gamma_fkd=DERSA_GAMMA_FKD, gamma_jac=DERSA_GAMMA_JAC,\n'
        '                tau=DERSA_TAU)',
        'res_dersa'
    ),
    'CPG-Net (ours)': (
        'CPGNetLearner(make_cpg_model(), lr=LR, device=DEVICE,\n'
        '                importance_lambda=CPG_IMPORTANCE_LAM,\n'
        '                fisher_samples=CPG_FISHER_N,\n'
        '                gate_lr=CPG_GATE_LR,\n'
        '                gate_sparsity=CPG_GATE_SPARSITY)',
        'res_cpg'
    ),
}

for strat_name, (constructor, var_name) in strategies_code.items():
    label = strat_name
    if strat_name == 'CPG-Net (ours)':
        label = 'CPG-Net (ours) -- NOVEL METHOD'
    code(f"""# --- {label} ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: {strat_name}')
print('='*60)
learner = {constructor}
{var_name} = run_strategy('{strat_name}', learner, task_list)
print(f'\\n  ACC={{{var_name}["ACC"]:.4f}}  F={{{var_name}["F"]:.4f}}  '
      f'BWT={{{var_name}["BWT"]:+.4f}}  FWT={{{var_name}["FWT"]:+.4f}}  '
      f'Mem={{{var_name}["memory_mb"]:.2f}}MB  Time={{{var_name}["time"]:.0f}}s')
""")

# ============================================================
# CELL 15: Collect results + ranked table
# ============================================================
code(r"""# Collect all results
all_results = {
    'Naive Fine-Tune': res_naive,
    'Joint Training':  res_joint,
    'Online EWC':      res_ewc,
    'A-GEM':           res_agem,
    'DER++':           res_derpp,
    'DER-SA':          res_dersa,
    'CPG-Net (ours)':  res_cpg,
}

# ================================================================
# Results Table -- Ranked by ACC
# ================================================================
print('='*120)
print('EXPERIMENT 7 -- CPG-Net vs ALL BASELINES (Ranked by ACC)')
print('='*120)
print(f'{"Rank":>4s}  {"Strategy":20s}  {"ACC":>8s}  {"F":>8s}  {"BWT":>8s}  '
      f'{"FWT":>8s}  {"Ovrl R2":>8s}  {"Mem(MB)":>8s}  {"Time(s)":>8s}')
print('-' * 115)

ranked = sorted(all_results.items(), key=lambda x: x[1]['ACC'], reverse=True)

for rank, (name, d) in enumerate(ranked, 1):
    marker = ' ***' if name == 'CPG-Net (ours)' else ''
    print(f'{rank:4d}  {name:20s}  {d["ACC"]:8.4f}  {d["F"]:8.4f}  {d["BWT"]:+8.4f}  '
          f'{d["FWT"]:+8.4f}  {d["overall_r2"]:8.4f}  {d["memory_mb"]:8.2f}  {d["time"]:7.0f}s{marker}')

# Gap analysis
print(f'\n{"="*120}')
print('GAP ANALYSIS')
print(f'{"="*120}')

cpg = all_results['CPG-Net (ours)']
dp = all_results['DER++']
sa = all_results['DER-SA']
jt = all_results['Joint Training']

print(f'  CPG-Net vs DER++ (previous best CL):')
print(f'    ACC improvement:         {cpg["ACC"] - dp["ACC"]:+.4f}')
print(f'    Forgetting reduction:    {dp["F"] - cpg["F"]:+.4f} (positive = less forgetting)')
print(f'    BWT improvement:         {cpg["BWT"] - dp["BWT"]:+.4f}')
print(f'    FWT improvement:         {cpg["FWT"] - dp["FWT"]:+.4f}')

print(f'\n  CPG-Net vs DER-SA (NB08 best):')
print(f'    ACC improvement:         {cpg["ACC"] - sa["ACC"]:+.4f}')
print(f'    Forgetting reduction:    {sa["F"] - cpg["F"]:+.4f}')
print(f'    BWT improvement:         {cpg["BWT"] - sa["BWT"]:+.4f}')

print(f'\n  CPG-Net vs Joint Training (upper bound):')
print(f'    ACC gap:                 {cpg["ACC"] - jt["ACC"]:+.4f}')
gap_closed = (cpg["ACC"] - dp["ACC"]) / max(jt["ACC"] - dp["ACC"], 1e-8) * 100
print(f'    Gap closed (vs DER++):   {gap_closed:.1f}%')
print(f'    Speed ratio:             {jt["time"]/max(cpg["time"],1):.1f}x faster')

# Save
save_data = {}
for k, v in all_results.items():
    save_data[k] = {kk: vv for kk, vv in v.items()
                    if kk not in ('metrics', 'summary')}
    save_data[k]['summary'] = v['summary']
with open(f'{RESULTS_DIR}/full_results.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f'\nResults saved to {RESULTS_DIR}/full_results.json')
""")

# ============================================================
# CELL 16: 6-metric visualization
# ============================================================
code(r"""# Visualization: All 6 Metrics (7 strategies)

colors = {
    'Naive Fine-Tune': '#e74c3c', 'Joint Training': '#2ecc71',
    'DER++': '#e67e22', 'Online EWC': '#9b59b6', 'A-GEM': '#34495e',
    'DER-SA': '#f39c12', 'CPG-Net (ours)': '#3498db'
}

ranked_names = [n for n, _ in ranked]

fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# --- ACC ---
ax = axes[0, 0]
vals = [all_results[n]['ACC'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'CPG-Net (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('ACC (higher = better)')
ax.set_title('Average Accuracy (ACC)', fontweight='bold')
ax.axvline(0, color='gray', linewidth=0.5)

# --- Forgetting ---
ax = axes[0, 1]
vals = [all_results[n]['F'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'CPG-Net (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Forgetting F (lower = better)')
ax.set_title('Forgetting (F)', fontweight='bold')

# --- BWT ---
ax = axes[0, 2]
vals = [all_results[n]['BWT'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'CPG-Net (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('BWT (closer to 0 = better)')
ax.set_title('Backward Transfer (BWT)', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

# --- FWT ---
ax = axes[1, 0]
vals = [all_results[n]['FWT'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'CPG-Net (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('FWT (higher = better)')
ax.set_title('Forward Transfer (FWT)', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

# --- Memory ---
ax = axes[1, 1]
vals = [all_results[n]['memory_mb'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'CPG-Net (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Memory Usage (MB, lower = better)')
ax.set_title('Memory Usage', fontweight='bold')

# --- Time ---
ax = axes[1, 2]
vals = [all_results[n]['time'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'CPG-Net (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Training Time (s)')
ax.set_title('Training Time', fontweight='bold')

plt.suptitle('CPG-Net vs All Baselines -- All 6 Metrics', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/all_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 17: R2 evolution + forgetting trajectory
# ============================================================
code(r"""# R2 evolution: forgetting trajectory + average accuracy over time

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

focus_strats = ['Naive Fine-Tune', 'DER++', 'DER-SA', 'CPG-Net (ours)', 'Joint Training']

# Plot 1: Task 0 R2 over time
ax = axes[0]
for name in focus_strats:
    if name in all_results and 'summary' in all_results[name]:
        r2_mat = all_results[name]['summary']['r2_matrix']
        task0_r2 = [r2_mat[i][0] for i in range(len(r2_mat)) if len(r2_mat[i]) > 0]
        ax.plot(range(len(task0_r2)), task0_r2, 'o-', label=name,
                color=colors.get(name, '#888'), linewidth=2, markersize=6)
ax.set_xlabel('After training on task #')
ax.set_ylabel('R2 on Task 0 (first participant)')
ax.set_title('Forgetting Trajectory: Task 0')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Average R2 across all seen tasks
ax = axes[1]
for name in focus_strats:
    if name in all_results and 'summary' in all_results[name]:
        r2_mat = all_results[name]['summary']['r2_matrix']
        avg_r2 = []
        for i in range(len(r2_mat)):
            if r2_mat[i]:
                avg_r2.append(np.mean(r2_mat[i][:i+1]))
        ax.plot(range(len(avg_r2)), avg_r2, 'o-', label=name,
                color=colors.get(name, '#888'), linewidth=2, markersize=6)
ax.set_xlabel('After training on task #')
ax.set_ylabel('Avg R2 on all tasks seen so far')
ax.set_title('Average Accuracy Evolution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/r2_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 18: Markdown header for ablation
# ============================================================
md(r"""---
## CPG-Net Ablation Study

Test the contribution of each CPG-Net component:
1. **Gate Mechanism**: Soft gating vs no gating (standard MLP with EWC only)
2. **Context Generator**: Learned gates vs random gates
3. **Progressive Freezing**: With vs without gate freezing
4. **Importance Regularization**: With vs without backbone protection
5. **Gate Sparsity**: With vs without entropy penalty
""")

# ============================================================
# CELL 19: Ablation experiments
# ============================================================
code(r"""# CPG-Net Ablation Study

ablation_configs = {
    'No gating (EWC only)': {
        'use_cpg': False,
        'ewc_lambda': CPG_IMPORTANCE_LAM,
        'desc': 'Standard MLP + EWC (no gates at all)',
    },
    'Random gates (no context)': {
        'use_cpg': True,
        'random_gates': True,
        'freeze': True,
        'importance': True,
        'sparsity': CPG_GATE_SPARSITY,
        'desc': 'CPGNet but gates initialized randomly (not from context)',
    },
    'No gate freezing': {
        'use_cpg': True,
        'random_gates': False,
        'freeze': False,
        'importance': True,
        'sparsity': CPG_GATE_SPARSITY,
        'desc': 'CPGNet without progressive freezing of gates',
    },
    'No importance reg.': {
        'use_cpg': True,
        'random_gates': False,
        'freeze': True,
        'importance': False,
        'sparsity': CPG_GATE_SPARSITY,
        'desc': 'CPGNet without backbone importance regularization',
    },
    'No gate sparsity': {
        'use_cpg': True,
        'random_gates': False,
        'freeze': True,
        'importance': True,
        'sparsity': 0.0,
        'desc': 'CPGNet without gate sparsity penalty',
    },
    'CPG-Net (full)': {
        'use_cpg': True,
        'random_gates': False,
        'freeze': True,
        'importance': True,
        'sparsity': CPG_GATE_SPARSITY,
        'desc': 'Full CPG-Net with all components',
    },
}


class CPGNetLearnerNoFreeze(CPGNetLearner):
    # Variant: do not freeze gates after tasks
    def on_task_end(self, task_id, train_loader):
        # Skip freezing, still compute importance
        self._completed_tasks.append(task_id)
        importance = {}
        self.model.eval()
        ns = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'gate' not in name:
                importance[name] = torch.zeros_like(param)
        for b in train_loader:
            o, a = b[0].to(self.device), b[1].to(self.device)
            self.model.zero_grad()
            pred = self.model(o, task_id=task_id)
            F.mse_loss(pred, a).backward()
            for name, param in self.model.named_parameters():
                if name in importance and param.grad is not None:
                    importance[name] += param.grad.data.pow(2) * o.size(0)
            ns += o.size(0)
            if ns >= self.fisher_samples: break
        for name in importance: importance[name] /= max(ns, 1)
        self.model.train()
        if self._cum_importance is None:
            self._cum_importance = importance
        else:
            for name in importance:
                if name in self._cum_importance:
                    self._cum_importance[name] = 0.9 * self._cum_importance[name] + importance[name]
        self._anchor_params = {name: param.clone().detach()
                               for name, param in self.model.named_parameters() if param.requires_grad}


class CPGNetRandomGates(CPGNet):
    # Variant: gates are random (not context-derived)
    def register_task(self, task_id, obs_data):
        gate_params = []
        for h_dim in self.hidden_dims:
            raw = torch.randn(h_dim) * 0.5 + 2.0  # start near open
            param = nn.Parameter(raw)
            gate_params.append(param)
        self._task_gates[task_id] = gate_params
        with torch.no_grad():
            obs_mean = obs_data.mean(0)
            obs_var = obs_data.var(0)
        self._task_stats[task_id] = (obs_mean.detach(), obs_var.detach())


ablation_results = {}

for ab_name, cfg in ablation_configs.items():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print(f'\nAblation: {ab_name}')
    print(f'  {cfg["desc"]}')

    if not cfg.get('use_cpg', True):
        # No gating: just EWC on standard MLP
        learner = EWC(make_model(), lr=LR, device=DEVICE,
                      ewc_lambda=cfg.get('ewc_lambda', CPG_IMPORTANCE_LAM),
                      fisher_samples=CPG_FISHER_N, online=True, gamma=0.9)
        res = run_strategy(ab_name, learner, task_list, verbose=False)
    else:
        if cfg.get('random_gates', False):
            model = CPGNetRandomGates(obs_dim, act_dim, HIDDEN, DROPOUT,
                                      max_tasks=N_TASKS, gate_hidden=CPG_GATE_HIDDEN)
        else:
            model = make_cpg_model()

        imp_lambda = CPG_IMPORTANCE_LAM if cfg.get('importance', True) else 0.0
        sparsity = cfg.get('sparsity', CPG_GATE_SPARSITY)

        if cfg.get('freeze', True):
            learner = CPGNetLearner(model, lr=LR, device=DEVICE,
                                    importance_lambda=imp_lambda,
                                    fisher_samples=CPG_FISHER_N,
                                    gate_lr=CPG_GATE_LR,
                                    gate_sparsity=sparsity)
        else:
            learner = CPGNetLearnerNoFreeze(model, lr=LR, device=DEVICE,
                                            importance_lambda=imp_lambda,
                                            fisher_samples=CPG_FISHER_N,
                                            gate_lr=CPG_GATE_LR,
                                            gate_sparsity=sparsity)
        res = run_strategy(ab_name, learner, task_list, verbose=False)

    ablation_results[ab_name] = {
        'ACC': res['ACC'], 'F': res['F'],
        'BWT': res['BWT'], 'FWT': res['FWT'],
        'overall_r2': res['overall_r2'],
        'memory_mb': res['memory_mb'],
        'time': res['time'],
    }
    print(f'  --> ACC={res["ACC"]:.4f}  F={res["F"]:.4f}  BWT={res["BWT"]:+.4f}  '
          f'FWT={res["FWT"]:+.4f}  ({res["time"]:.0f}s)')

# Ablation table
print(f'\n{"="*110}')
print('CPG-NET ABLATION RESULTS')
print(f'{"="*110}')
print(f'{"Configuration":25s}  {"ACC":>8s}  {"F":>8s}  {"BWT":>8s}  '
      f'{"FWT":>8s}  {"Ovrl R2":>8s}  {"Time":>7s}')
print('-' * 90)
for name, d in ablation_results.items():
    print(f'{name:25s}  {d["ACC"]:8.4f}  {d["F"]:8.4f}  {d["BWT"]:+8.4f}  '
          f'{d["FWT"]:+8.4f}  {d["overall_r2"]:8.4f}  {d["time"]:6.0f}s')

with open(f'{RESULTS_DIR}/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2, default=str)
print(f'\nSaved to {RESULTS_DIR}/ablation_results.json')
""")

# ============================================================
# CELL 20: Ablation visualization
# ============================================================
code(r"""# Ablation Visualization: ACC + Forgetting side by side

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ab_names = list(ablation_results.keys())

# ACC
ax = axes[0]
acc_vals = [ablation_results[n]['ACC'] for n in ab_names]
bar_colors = ['#95a5a6'] * len(ab_names)
for i, n in enumerate(ab_names):
    if 'full' in n.lower():
        bar_colors[i] = '#3498db'
    elif 'EWC' in n or 'No gating' in n:
        bar_colors[i] = '#9b59b6'

bars = ax.barh(ab_names[::-1], acc_vals[::-1], color=bar_colors[::-1], edgecolor='white')
for bar, name in zip(bars, ab_names[::-1]):
    if 'full' in name.lower():
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('ACC (higher = better)')
ax.set_title('Ablation: Average Accuracy', fontweight='bold')

# Forgetting
ax = axes[1]
f_vals = [ablation_results[n]['F'] for n in ab_names]
bars = ax.barh(ab_names[::-1], f_vals[::-1], color=bar_colors[::-1], edgecolor='white')
for bar, name in zip(bars, ab_names[::-1]):
    if 'full' in name.lower():
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Forgetting (lower = better)')
ax.set_title('Ablation: Forgetting', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/ablation_chart.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 21: Gate analysis visualization (NOVEL)
# ============================================================
code(r"""# Gate Analysis: Visualize learned gating patterns across tasks
# This is unique to CPG-Net -- shows which neurons are shared vs isolated

if hasattr(res_cpg, '__getitem__') and 'metrics' in res_cpg:
    cpg_model = None
    # Re-build and re-train to capture gate values (use saved model from run)
    # Instead, reconstruct from the learner's model
    print("Gate analysis using CPG-Net model from training run")

# Re-create the model to analyze gates
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
cpg_model_analysis = make_cpg_model().to(DEVICE)
cpg_learner_analysis = CPGNetLearner(
    cpg_model_analysis, lr=LR, device=DEVICE,
    importance_lambda=CPG_IMPORTANCE_LAM,
    fisher_samples=CPG_FISHER_N,
    gate_lr=CPG_GATE_LR,
    gate_sparsity=CPG_GATE_SPARSITY
)

# Quick training for gate analysis (fewer epochs for speed)
for task in task_list:
    cpg_learner_analysis.train_task(task.task_id, task.train_loader(BS),
                                     task.val_loader(BS*2), epochs=30, patience=8)

# Extract gate values
gate_data = {}  # task_id -> layer -> gate_values (sigmoid)
for tid in range(len(task_list)):
    if tid in cpg_model_analysis._task_gates:
        gate_data[tid] = []
        for g in cpg_model_analysis._task_gates[tid]:
            gate_vals = torch.sigmoid(g).detach().cpu().numpy()
            gate_data[tid].append(gate_vals)

if gate_data:
    n_layers = len(gate_data[0])
    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        # Build matrix: tasks x neurons
        gate_matrix = []
        task_labels = []
        for tid in sorted(gate_data.keys()):
            if layer_idx < len(gate_data[tid]):
                gate_matrix.append(gate_data[tid][layer_idx])
                task_labels.append(f'T{tid} ({task_list[tid].participant_id})')
        gate_matrix = np.array(gate_matrix)

        im = ax.imshow(gate_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_yticks(range(len(task_labels)))
        ax.set_yticklabels(task_labels, fontsize=8)
        ax.set_xlabel('Neuron index')
        ax.set_title(f'Layer {layer_idx+1} Gate Values\n(green=active, red=inactive)', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Gate value')

        # Compute sharing statistics
        active = (gate_matrix > 0.5).sum(axis=0)
        shared_pct = (active == len(task_labels)).sum() / gate_matrix.shape[1] * 100
        isolated_pct = (active == 1).sum() / gate_matrix.shape[1] * 100
        ax.set_xlabel(f'Neuron index\n(Shared by all: {shared_pct:.0f}%, Isolated: {isolated_pct:.0f}%)')

    plt.suptitle('CPG-Net: Learned Gating Patterns Across Tasks', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/gate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No gate data available for visualization")
""")

# ============================================================
# CELL 22: Per-joint MSE comparison
# ============================================================
code(r"""# Per-Joint MSE comparison: DER++ vs DER-SA vs CPG-Net

joint_names = ['J1 (shoulder)', 'J2 (shoulder)', 'J3 (elbow)',
               'J4 (wrist)', 'J5 (wrist)', 'J6 (wrist)']

fig, ax = plt.subplots(figsize=(14, 5))

strats_to_compare = ['DER++', 'DER-SA', 'CPG-Net (ours)']
strat_colors = {'DER++': '#e67e22', 'DER-SA': '#f39c12', 'CPG-Net (ours)': '#3498db'}
n_strats = len(strats_to_compare)
w = 0.25

for si, sname in enumerate(strats_to_compare):
    if sname in all_results and 'summary' in all_results[sname]:
        per_dim = all_results[sname]['summary']['per_dim_matrix'][-1]
        avg_per_dim = np.mean(per_dim, axis=0)
        x = np.arange(len(joint_names))
        offset = (si - n_strats/2 + 0.5) * w
        ax.bar(x + offset, avg_per_dim, w, label=sname,
               color=strat_colors.get(sname, '#888'), alpha=0.85)

ax.set_xticks(np.arange(len(joint_names)))
ax.set_xticklabels(joint_names)
ax.set_ylabel('Avg MSE (lower = better)')
ax.set_title('Per-Joint MSE: DER++ vs DER-SA vs CPG-Net')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/per_joint_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ============================================================
# CELL 23: Save config
# ============================================================
code(r"""# Save experiment config

config = {
    'experiment': 'cpg_net_comparison',
    'date': time.strftime('%Y-%m-%d'),
    'metrics_used': ['ACC', 'F (Forgetting)', 'BWT', 'FWT', 'Memory (MB)', 'Time (s)'],
    'model_mlp': {'hidden': list(HIDDEN), 'dropout': DROPOUT, 'obs_dim': obs_dim, 'act_dim': act_dim},
    'model_cpg': {
        'hidden': list(HIDDEN), 'dropout': DROPOUT,
        'gate_hidden': CPG_GATE_HIDDEN,
        'gate_lr': CPG_GATE_LR,
        'importance_lambda': CPG_IMPORTANCE_LAM,
        'fisher_samples': CPG_FISHER_N,
        'gate_sparsity': CPG_GATE_SPARSITY,
    },
    'training': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE, 'batch_size': BS},
    'baselines': {
        'ewc_lambda': EWC_LAMBDA, 'oewc_gamma': OEWC_GAMMA,
        'der_buffer': DER_BUFFER, 'agem_mem': AGEM_MEM,
    },
    'der_sa': {
        'buffer_size': DERSA_BUFFER, 'alpha': DERSA_ALPHA, 'beta': DERSA_BETA,
        'gamma_fkd': DERSA_GAMMA_FKD, 'gamma_jac': DERSA_GAMMA_JAC, 'tau': DERSA_TAU,
    },
    'n_tasks': N_TASKS, 'seed': SEED, 'device': DEVICE,
}
with open(f'{RESULTS_DIR}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Config saved to:', f'{RESULTS_DIR}/config.json')
print(f'\nAll outputs in {RESULTS_DIR}:')
for f_name in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f_name))
    print(f'  {f_name:45s} {size/1024:.1f} KB')
""")

# ============================================================
# CELL 24: Summary (to be filled after execution)
# ============================================================
md(r"""---
## Summary

### Metrics Overview (Ranked by ACC)

| Rank | Strategy | ACC | F (Forgetting) | BWT | FWT | Memory (MB) | Time (s) |
|------|----------|-----|----------------|-----|-----|-------------|----------|
| ? | ? | ? | ? | ? | ? | ? | ? |

### CPG-Net vs Previous Best (DER-SA)

| Metric | DER-SA | CPG-Net | Delta | Interpretation |
|--------|--------|---------|-------|----------------|
| ? | ? | ? | ? | ? |

### CPG-Net Ablation Results

| Configuration | ACC | F | Interpretation |
|---------------|-----|---|----------------|
| ? | ? | ? | ? |

### Key Findings

1. **?**
2. **?**
3. **?**
""")


# ============================================================
# Build and save notebook
# ============================================================
nb.cells = cells

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '09_cpg_net_experiment.ipynb')
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print(f'Notebook written to: {out_path}')
print(f'Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type=="code")} code, '
      f'{sum(1 for c in cells if c.cell_type=="markdown")} markdown)')
