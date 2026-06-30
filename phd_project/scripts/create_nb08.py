#!/usr/bin/env python3
# Generate notebook 08_der_sa_experiment.ipynb
# Each algorithm runs in its own cell; metrics: ACC, F, BWT, FWT, Memory, Time
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.0'}
})

cells = []

# ═══════════════════════════════════════════════════════════════════════
# CELL — Markdown header
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""# Experiment 6 -- DER-SA: Domain-Aware Replay for Shared Autonomy

**PhD Project:** Collaboration Humain-Robot : Apprentissage incremental et adaptation comportementale
**Author:** Ameur Gargouri
**Notebook:** `08_der_sa_experiment.ipynb`

## Motivation

From our benchmark (Exp03/05), **DER++** is the best continual learning strategy for
shared autonomy, achieving **R2=0.597** (vs Joint Training R2=0.649). But it still
suffers from **systemic forgetting** (R2 BWT = -0.15): all 6 joints lose accuracy
together as new participants are learned.

## DER-SA: Three Novel Components

1. **Adaptive Replay Weighting (ARW):** Weight buffer samples by forgetting risk
2. **Joint-Aware Consistency Loss (JAC):** Exploit systemic co-forgetting structure
3. **Feature-Level Knowledge Distillation (FKD):** Distill penultimate hidden features

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **ACC** (Average Accuracy) | $\frac{1}{T}\sum_{j=1}^{T} a_{T,j}$ | Mean R2 after all tasks |
| **F** (Forgetting) | $\frac{1}{T-1}\sum_{j=1}^{T-1}\max_{l\in\{1,..,T-1\}}(a_{l,j}-a_{T,j})$ | Avg max drop per task |
| **BWT** (Backward Transfer) | $\frac{1}{T-1}\sum_{j=1}^{T-1}(a_{T,j}-a_{j,j})$ | Avg change on old tasks |
| **FWT** (Forward Transfer) | $\frac{1}{T-1}\sum_{j=2}^{T}(a_{j-1,j}-\bar{b}_j)$ | Avg zero-shot on unseen tasks |
| **Memory** | Buffer size x sample dims | Memory footprint in MB |
| **Time** | Wall-clock seconds | Total training time |

---
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Imports
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""import os, sys, abc, copy, json, logging, random, time, warnings, tracemalloc
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
RESULTS_DIR  = f'{PROJECT_ROOT}/experiments/exp06_der_sa'
os.makedirs(RESULTS_DIR, exist_ok=True)

participants = sorted([d for d in os.listdir(HARMONIC_DIR)
                       if os.path.isdir(os.path.join(HARMONIC_DIR, d)) and d.startswith('p')])
print(f'Available participants: {len(participants)} -> {participants}')
print(f'Results: {RESULTS_DIR}')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Metrics: ACC, F, BWT, FWT + helpers
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# ================================================================
# Continual Learning Metrics: ACC, F, BWT, FWT, Memory, Time
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
        # b_j = R2 of random/untrained model on task j (for FWT)
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
        # Average Accuracy: (1/T) * sum_j a_{T,j}
        if self.T == 0: return float('nan')
        last = self.R[-1]
        return float(np.mean(last)) if last else float('nan')

    @property
    def forgetting(self):
        # Forgetting: (1/(T-1)) * sum_{j=1}^{T-1} max_{l in 1..T-1}(a_{l,j} - a_{T,j})
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
        # Backward Transfer: (1/(T-1)) * sum_{j=1}^{T-1} (a_{T,j} - a_{j,j})
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
        # Forward Transfer: (1/(T-1)) * sum_{j=2}^{T} (a_{j-1,j} - b_j)
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
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Model + Base CL + All strategy classes
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# ================================================================
# Model & CL Strategies
# ================================================================

class MLPPolicy(nn.Module):
    # MLP with accessible penultimate features for FKD
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
                vl = self._eval_loss(val_loader)
                if vl < best_val:
                    best_val = vl; best_state = copy.deepcopy(self.model.state_dict()); no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
        if best_state is not None: self.model.load_state_dict(best_state)
        self.on_task_end(task_id, train_loader)

    def _eval_loss(self, loader):
        self.model.eval()
        tot, n = 0., 0
        with torch.no_grad():
            for b in loader:
                o, a = b[0].to(self.device), b[1].to(self.device)
                tot += F.mse_loss(self.model(o), a).item() * o.size(0); n += o.size(0)
        self.model.train()
        return tot / max(n, 1)

    def evaluate_task(self, test_loader, task_id):
        self.model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for b in test_loader:
                o, a = b[0].to(self.device), b[1].to(self.device)
                preds.append(self.model(o).cpu()); tgts.append(a.cpu())
        p, t = torch.cat(preds), torch.cat(tgts)
        mse = F.mse_loss(p, t).item()
        mae = (p - t).abs().mean().item()
        ss_res = ((t - p)**2).sum().item()
        ss_tot = ((t - t.mean(0))**2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-8)
        per_dim = ((p - t)**2).mean(0).tolist()
        self.model.train()
        return TaskResult(task_id, mse, mse, mae, r2, len(t), per_dim, ss_res, ss_tot)


# --- Baselines ---

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
        return total * 4 / (1024**2)  # float32


class ReplayBuffer:
    # Reservoir-sampled replay buffer storing (obs, act, logits[, features])
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
    # DER++ (Buzzega et al., NeurIPS 2020)
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
    # Online EWC (Schwarz et al., ICML 2018)
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
    # A-GEM (Chaudhry et al., ICLR 2019)
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


# --- DER-SA (OUR METHOD) ---

class DERSA(ContinualLearner):
    # DER-SA: Dark Experience Replay with domain-aware enhancements
    # Components: ARW + FKD + JAC
    def __init__(self, model, lr=1e-3, device='cpu',
                 buffer_size=10000, alpha=0.5, beta=0.5,
                 gamma_fkd=0.3, gamma_jac=0.2, tau=2.0):
        super().__init__(model, lr=lr, device=device)
        self.buffer = ReplayBuffer(buffer_size, store_features=True)
        self.alpha = alpha
        self.beta = beta
        self.gamma_fkd = gamma_fkd
        self.gamma_jac = gamma_jac
        self.tau = tau
        self._ref_cov = None

    def on_task_start(self, task_id, train_loader):
        pass

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

        if len(self.buffer) == 0 or task_id == 0:
            return loss_current

        sample = self.buffer.sample(max(1, obs.size(0)))
        bo, ba, bl, bf = sample[0].to(self.device), sample[1].to(self.device), \
                         sample[2].to(self.device), sample[3].to(self.device)

        bp, bp_feat = self.model.forward_with_features(bo)

        # ARW: adaptive replay weighting
        with torch.no_grad():
            drift = (bp - bl).pow(2).mean(dim=1)
            weights = torch.softmax(drift / self.tau, dim=0) * drift.size(0)

        loss_logit = (weights.unsqueeze(1) * (bp - bl).pow(2)).mean()
        loss_target = (weights.unsqueeze(1) * (bp - ba).pow(2)).mean()

        # FKD: feature-level knowledge distillation
        loss_fkd = F.mse_loss(bp_feat, bf)

        # JAC: joint-aware consistency
        loss_jac = torch.tensor(0., device=self.device)
        if self._ref_cov is not None:
            bp_centered = bp - bp.mean(0)
            cur_cov = (bp_centered.T @ bp_centered) / (bp.size(0) - 1)
            loss_jac = F.mse_loss(cur_cov, self._ref_cov)

        total = (loss_current
                 + self.alpha * loss_logit
                 + self.beta * loss_target
                 + self.gamma_fkd * loss_fkd
                 + self.gamma_jac * loss_jac)
        return total

    def memory_mb(self): return self.buffer.memory_mb()


print('All strategies defined: Naive, Joint, DER++, Online EWC, A-GEM, DER-SA')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Data Loading
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Data Loading -- identical to previous experiments

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
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Setup, build tasks, compute random baselines for FWT
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Hyperparameters & Task Setup

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

# DER-SA HPs
DERSA_BUFFER    = 10000
DERSA_ALPHA     = 0.5
DERSA_BETA      = 0.5
DERSA_GAMMA_FKD = 0.3
DERSA_GAMMA_JAC = 0.2
DERSA_TAU       = 2.0

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

def run_strategy(name, learner, task_list, epochs=EPOCHS, patience=PATIENCE, bs=BS, verbose=True):
    # Compute random baselines for FWT (untrained model R2 on each task)
    torch.manual_seed(SEED)
    random_baselines = []
    untrained = make_model().to(DEVICE)
    for t in task_list:
        untrained.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for b in t.test_loader(bs*2):
                o, a = b[0].to(DEVICE), b[1].to(DEVICE)
                preds.append(untrained(o).cpu()); tgts.append(a.cpu())
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

print(f'\nReady: {len(task_list)} tasks, {obs_dim}D -> {act_dim}D')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Markdown: run comparison
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""---
## Run Full Comparison

Each algorithm runs in its own cell. Metrics tracked per strategy:
**ACC** (Average Accuracy R2), **F** (Forgetting), **BWT**, **FWT**, **Memory (MB)**, **Time (s)**
"""))

# ═══════════════════════════════════════════════════════════════════════
# One cell per strategy
# ═══════════════════════════════════════════════════════════════════════
strategy_cells = [
    ('Naive Fine-Tune', r"""# --- Naive Fine-Tune ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: Naive Fine-Tune')
print('='*60)
learner = NaiveFineTune(make_model(), lr=LR, device=DEVICE)
res_naive = run_strategy('Naive Fine-Tune', learner, task_list)
print(f'\n  ACC={res_naive["ACC"]:.4f}  F={res_naive["F"]:.4f}  '
      f'BWT={res_naive["BWT"]:+.4f}  FWT={res_naive["FWT"]:+.4f}  '
      f'Mem={res_naive["memory_mb"]:.2f}MB  Time={res_naive["time"]:.0f}s')
"""),
    ('Joint Training', r"""# --- Joint Training (upper bound) ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: Joint Training')
print('='*60)
learner = JointTraining(make_model(), lr=LR, device=DEVICE)
res_joint = run_strategy('Joint Training', learner, task_list)
print(f'\n  ACC={res_joint["ACC"]:.4f}  F={res_joint["F"]:.4f}  '
      f'BWT={res_joint["BWT"]:+.4f}  FWT={res_joint["FWT"]:+.4f}  '
      f'Mem={res_joint["memory_mb"]:.2f}MB  Time={res_joint["time"]:.0f}s')
"""),
    ('Online EWC', r"""# --- Online EWC ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: Online EWC')
print('='*60)
learner = EWC(make_model(), lr=LR, device=DEVICE,
              ewc_lambda=EWC_LAMBDA, fisher_samples=EWC_FISHER_N,
              online=True, gamma=OEWC_GAMMA)
res_ewc = run_strategy('Online EWC', learner, task_list)
print(f'\n  ACC={res_ewc["ACC"]:.4f}  F={res_ewc["F"]:.4f}  '
      f'BWT={res_ewc["BWT"]:+.4f}  FWT={res_ewc["FWT"]:+.4f}  '
      f'Mem={res_ewc["memory_mb"]:.2f}MB  Time={res_ewc["time"]:.0f}s')
"""),
    ('A-GEM', r"""# --- A-GEM ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: A-GEM')
print('='*60)
learner = AGEM(make_model(), lr=LR, device=DEVICE, mem_per_task=AGEM_MEM)
res_agem = run_strategy('A-GEM', learner, task_list)
print(f'\n  ACC={res_agem["ACC"]:.4f}  F={res_agem["F"]:.4f}  '
      f'BWT={res_agem["BWT"]:+.4f}  FWT={res_agem["FWT"]:+.4f}  '
      f'Mem={res_agem["memory_mb"]:.2f}MB  Time={res_agem["time"]:.0f}s')
"""),
    ('DER++', r"""# --- DER++ ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: DER++')
print('='*60)
learner = DERPlusPlus(make_model(), lr=LR, device=DEVICE, buffer_size=DER_BUFFER)
res_derpp = run_strategy('DER++', learner, task_list)
print(f'\n  ACC={res_derpp["ACC"]:.4f}  F={res_derpp["F"]:.4f}  '
      f'BWT={res_derpp["BWT"]:+.4f}  FWT={res_derpp["FWT"]:+.4f}  '
      f'Mem={res_derpp["memory_mb"]:.2f}MB  Time={res_derpp["time"]:.0f}s')
"""),
    ('DER-SA (ours)', r"""# --- DER-SA (OUR METHOD) ---
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
print('='*60)
print('  Running: DER-SA (ours)')
print('='*60)
learner = DERSA(make_model(), lr=LR, device=DEVICE,
                buffer_size=DERSA_BUFFER, alpha=DERSA_ALPHA, beta=DERSA_BETA,
                gamma_fkd=DERSA_GAMMA_FKD, gamma_jac=DERSA_GAMMA_JAC,
                tau=DERSA_TAU)
res_dersa = run_strategy('DER-SA (ours)', learner, task_list)
print(f'\n  ACC={res_dersa["ACC"]:.4f}  F={res_dersa["F"]:.4f}  '
      f'BWT={res_dersa["BWT"]:+.4f}  FWT={res_dersa["FWT"]:+.4f}  '
      f'Mem={res_dersa["memory_mb"]:.2f}MB  Time={res_dersa["time"]:.0f}s')
"""),
]

for name, code in strategy_cells:
    cells.append(nbf.v4.new_code_cell(code))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Collect all results & Results Table
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Collect all results into one dict
all_results = {
    'Naive Fine-Tune': res_naive,
    'Joint Training':  res_joint,
    'Online EWC':      res_ewc,
    'A-GEM':           res_agem,
    'DER++':           res_derpp,
    'DER-SA (ours)':   res_dersa,
}

# ================================================================
# Results Table -- Ranked by ACC (Average Accuracy R2)
# ================================================================
print('='*115)
print('EXPERIMENT 6 -- DER-SA vs BASELINES (Ranked by ACC)')
print('='*115)
print(f'{"Rank":>4s}  {"Strategy":20s}  {"ACC":>8s}  {"F":>8s}  {"BWT":>8s}  '
      f'{"FWT":>8s}  {"Ovrl R2":>8s}  {"Mem(MB)":>8s}  {"Time(s)":>8s}')
print('-' * 110)

ranked = sorted(all_results.items(), key=lambda x: x[1]['ACC'], reverse=True)

for rank, (name, d) in enumerate(ranked, 1):
    marker = ' ***' if name == 'DER-SA (ours)' else ''
    print(f'{rank:4d}  {name:20s}  {d["ACC"]:8.4f}  {d["F"]:8.4f}  {d["BWT"]:+8.4f}  '
          f'{d["FWT"]:+8.4f}  {d["overall_r2"]:8.4f}  {d["memory_mb"]:8.2f}  {d["time"]:7.0f}s{marker}')

# Gap analysis
print(f'\n{"="*115}')
print('GAP ANALYSIS')
print(f'{"="*115}')
if 'DER-SA (ours)' in all_results and 'DER++' in all_results:
    sa = all_results['DER-SA (ours)']
    dp = all_results['DER++']
    jt = all_results.get('Joint Training', {})
    print(f'  DER-SA vs DER++:')
    print(f'    ACC improvement:         {sa["ACC"] - dp["ACC"]:+.4f}')
    print(f'    Forgetting reduction:    {dp["F"] - sa["F"]:+.4f} (positive = less forgetting)')
    print(f'    BWT improvement:         {sa["BWT"] - dp["BWT"]:+.4f}')
    print(f'    FWT improvement:         {sa["FWT"] - dp["FWT"]:+.4f}')
    print(f'    Overall R2 improvement:  {sa["overall_r2"] - dp["overall_r2"]:+.4f}')
    print(f'    Memory overhead:         {sa["memory_mb"] - dp["memory_mb"]:+.2f} MB')
    if jt:
        print(f'  DER-SA vs Joint Training:')
        print(f'    ACC gap:                 {sa["ACC"] - jt["ACC"]:+.4f}')
        print(f'    Speed ratio:             {jt["time"]/max(sa["time"],1):.1f}x')

# Save
save_data = {}
for k, v in all_results.items():
    save_data[k] = {kk: vv for kk, vv in v.items()
                    if kk not in ('metrics', 'summary')}
    save_data[k]['summary'] = v['summary']
with open(f'{RESULTS_DIR}/full_results.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f'\nResults saved to {RESULTS_DIR}/full_results.json')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Visualization: 6-panel metric comparison
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Visualization: All 6 Metrics

colors = {
    'Naive Fine-Tune': '#e74c3c', 'Joint Training': '#2ecc71',
    'DER++': '#e67e22', 'Online EWC': '#9b59b6', 'A-GEM': '#34495e',
    'DER-SA (ours)': '#3498db'
}

ranked_names = [n for n, _ in ranked]

fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# --- ACC ---
ax = axes[0, 0]
vals = [all_results[n]['ACC'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
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
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Forgetting F (lower = better)')
ax.set_title('Forgetting (F)', fontweight='bold')

# --- BWT ---
ax = axes[0, 2]
vals = [all_results[n]['BWT'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
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
    if name == 'DER-SA (ours)':
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
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Memory Usage (MB, lower = better)')
ax.set_title('Memory Usage', fontweight='bold')

# --- Time ---
ax = axes[1, 2]
vals = [all_results[n]['time'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], vals[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Training Time (s)')
ax.set_title('Training Time', fontweight='bold')

plt.suptitle('DER-SA vs Baselines -- All 6 Metrics', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/all_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — R2 evolution per task
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# R2 evolution: forgetting trajectory + average accuracy over time

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

focus_strats = ['Naive Fine-Tune', 'DER++', 'DER-SA (ours)', 'Joint Training']

# Plot 1: Task 0 R2 over time
ax = axes[0]
for name in focus_strats:
    if name in all_results and 'summary' in all_results[name]:
        r2_mat = all_results[name]['summary']['r2_matrix']
        task0_r2 = [r2_mat[i][0] for i in range(len(r2_mat)) if len(r2_mat[i]) > 0]
        ax.plot(range(len(task0_r2)), task0_r2, 'o-', label=name,
                color=colors.get(name, '#888'), linewidth=2, markersize=6)
ax.set_xlabel('After training on task #')
ax.set_ylabel('R2 on Task 0 (p100)')
ax.set_title('Forgetting Trajectory: Task 0 (p100)')
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
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Ablation Markdown
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""---
## Ablation Study

Test each DER-SA component individually to measure its contribution.
Each variant runs separately.
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Ablation Experiment
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Ablation: DER++ + each component alone, then all together

ablation_configs = {
    'DER++ (baseline)':     {'gamma_fkd': 0.0, 'gamma_jac': 0.0, 'tau': 2.0, 'use_derpp': True},
    'DER++ + ARW only':     {'gamma_fkd': 0.0, 'gamma_jac': 0.0, 'tau': 2.0, 'use_derpp': False},
    'DER++ + FKD only':     {'gamma_fkd': 0.3, 'gamma_jac': 0.0, 'tau': 999.0, 'use_derpp': False},
    'DER++ + JAC only':     {'gamma_fkd': 0.0, 'gamma_jac': 0.2, 'tau': 999.0, 'use_derpp': False},
    'DER++ + ARW + FKD':    {'gamma_fkd': 0.3, 'gamma_jac': 0.0, 'tau': 2.0, 'use_derpp': False},
    'DER++ + ARW + JAC':    {'gamma_fkd': 0.0, 'gamma_jac': 0.2, 'tau': 2.0, 'use_derpp': False},
    'DER++ + FKD + JAC':    {'gamma_fkd': 0.3, 'gamma_jac': 0.2, 'tau': 999.0, 'use_derpp': False},
    'DER-SA (full)':        {'gamma_fkd': 0.3, 'gamma_jac': 0.2, 'tau': 2.0, 'use_derpp': False},
}

ablation_results = {}

for name, cfg in ablation_configs.items():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print(f'\nAblation: {name}  (fkd={cfg["gamma_fkd"]}, jac={cfg["gamma_jac"]}, tau={cfg["tau"]})')

    if cfg.get('use_derpp'):
        learner = DERPlusPlus(make_model(), lr=LR, device=DEVICE, buffer_size=DER_BUFFER)
    else:
        learner = DERSA(make_model(), lr=LR, device=DEVICE,
                        buffer_size=DER_BUFFER, gamma_fkd=cfg['gamma_fkd'],
                        gamma_jac=cfg['gamma_jac'], tau=cfg['tau'])

    res = run_strategy(name, learner, task_list, verbose=False)
    ablation_results[name] = {
        'ACC': res['ACC'], 'F': res['F'],
        'BWT': res['BWT'], 'FWT': res['FWT'],
        'overall_r2': res['overall_r2'],
        'memory_mb': res['memory_mb'],
        'time': res['time'],
    }
    print(f'  -> ACC={res["ACC"]:.4f}  F={res["F"]:.4f}  BWT={res["BWT"]:+.4f}  '
          f'FWT={res["FWT"]:+.4f}  ({res["time"]:.0f}s)')

# Ablation table
print(f'\n{"="*100}')
print('ABLATION STUDY RESULTS')
print(f'{"="*100}')
print(f'{"Configuration":25s}  {"ACC":>8s}  {"F":>8s}  {"BWT":>8s}  '
      f'{"FWT":>8s}  {"Ovrl R2":>8s}  {"Time":>7s}')
print('-' * 85)
for name, d in ablation_results.items():
    print(f'{name:25s}  {d["ACC"]:8.4f}  {d["F"]:8.4f}  {d["BWT"]:+8.4f}  '
          f'{d["FWT"]:+8.4f}  {d["overall_r2"]:8.4f}  {d["time"]:6.0f}s')

with open(f'{RESULTS_DIR}/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2, default=str)
print(f'\nSaved to {RESULTS_DIR}/ablation_results.json')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Ablation Visualization
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Ablation Visualization: ACC + Forgetting side by side

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
names = list(ablation_results.keys())

# ACC
ax = axes[0]
acc_vals = [ablation_results[n]['ACC'] for n in names]
bar_colors = ['#e67e22' if 'baseline' in n else '#3498db' if 'full' in n else '#95a5a6' for n in names]
bars = ax.barh(names[::-1], acc_vals[::-1], color=bar_colors[::-1], edgecolor='white')
for bar, name in zip(bars, names[::-1]):
    if 'full' in name:
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
    elif 'baseline' in name:
        bar.set_edgecolor('#d35400'); bar.set_linewidth(2)
ax.set_xlabel('ACC (higher = better)')
ax.set_title('Ablation: Average Accuracy', fontweight='bold')
ax.axvline(acc_vals[0], color='#e67e22', linewidth=1.5, linestyle='--', alpha=0.7, label='DER++ baseline')
ax.legend()

# Forgetting
ax = axes[1]
f_vals = [ablation_results[n]['F'] for n in names]
bars = ax.barh(names[::-1], f_vals[::-1], color=bar_colors[::-1], edgecolor='white')
for bar, name in zip(bars, names[::-1]):
    if 'full' in name:
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
    elif 'baseline' in name:
        bar.set_edgecolor('#d35400'); bar.set_linewidth(2)
ax.set_xlabel('Forgetting (lower = better)')
ax.set_title('Ablation: Forgetting', fontweight='bold')
ax.axvline(f_vals[0], color='#e67e22', linewidth=1.5, linestyle='--', alpha=0.7, label='DER++ baseline')
ax.legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/ablation_chart.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Per-Joint analysis
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Per-Joint MSE comparison: DER++ vs DER-SA

joint_names = ['J1 (shoulder)', 'J2 (shoulder)', 'J3 (elbow)',
               'J4 (wrist)', 'J5 (wrist)', 'J6 (wrist)']

fig, ax = plt.subplots(figsize=(12, 5))

if 'DER++' in all_results and 'DER-SA (ours)' in all_results:
    dp_per_dim = all_results['DER++']['summary']['per_dim_matrix'][-1]
    sa_per_dim = all_results['DER-SA (ours)']['summary']['per_dim_matrix'][-1]

    dp_avg = np.mean(dp_per_dim, axis=0)
    sa_avg = np.mean(sa_per_dim, axis=0)

    x = np.arange(len(joint_names))
    w = 0.35
    ax.bar(x - w/2, dp_avg, w, label='DER++', color='#e67e22', alpha=0.85)
    ax.bar(x + w/2, sa_avg, w, label='DER-SA (ours)', color='#3498db', alpha=0.85)

    for i in range(len(joint_names)):
        imp = (dp_avg[i] - sa_avg[i]) / max(dp_avg[i], 1e-8) * 100
        ax.text(i, max(dp_avg[i], sa_avg[i]) + 0.01, f'{imp:+.1f}%',
                ha='center', fontsize=8, color='green' if imp > 0 else 'red')

    ax.set_xticks(x)
    ax.set_xticklabels(joint_names)
    ax.set_ylabel('Avg MSE (lower = better)')
    ax.set_title('Per-Joint MSE: DER++ vs DER-SA')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/per_joint_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Save config
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Save experiment config

config = {
    'experiment': 'der_sa_comparison',
    'date': time.strftime('%Y-%m-%d'),
    'metrics_used': ['ACC', 'F (Forgetting)', 'BWT', 'FWT', 'Memory (MB)', 'Time (s)'],
    'model': {'hidden': list(HIDDEN), 'dropout': DROPOUT, 'obs_dim': obs_dim, 'act_dim': act_dim},
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
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL — Summary
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""---
## Summary

*(to be filled after execution)*

### Metrics Overview

| Rank | Strategy | ACC | F | BWT | FWT | Memory (MB) | Time (s) |
|------|----------|-----|---|-----|-----|-------------|----------|
| 1 | ? | ? | ? | ? | ? | ? | ? |
| 2 | ? | ? | ? | ? | ? | ? | ? |
| 3 | ? | ? | ? | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? | ? | ? | ? |
| 5 | ? | ? | ? | ? | ? | ? | ? |
| 6 | ? | ? | ? | ? | ? | ? | ? |

### DER-SA Improvement over DER++
- ACC improvement: ?
- Forgetting reduction: ?
- BWT improvement: ?
- FWT improvement: ?
- Memory overhead: ?

### Ablation: Component Contributions
- ARW alone: ?
- FKD alone: ?
- JAC alone: ?
- Full DER-SA: ?
"""))

nb.cells = cells

# Write notebook
out_path = '/home/g0amer/Desktop/thesis/phd_project/notebooks/08_der_sa_experiment.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)

n_code = sum(1 for c in cells if c.cell_type == 'code')
n_md = sum(1 for c in cells if c.cell_type == 'markdown')
print(f'Notebook created: {out_path}')
print(f'  {len(cells)} cells ({n_code} code, {n_md} markdown)')
