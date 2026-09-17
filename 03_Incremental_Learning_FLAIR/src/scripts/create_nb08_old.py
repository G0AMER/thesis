#!/usr/bin/env python3
"""Generate notebook 08_der_sa_experiment.ipynb — DER-SA algorithm + comparison."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.0'}
})

cells = []

# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Markdown header
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""# Experiment 6 — DER-SA: Domain-Aware Replay for Shared Autonomy

**PhD Project:** Collaboration Humain-Robot : Apprentissage incrémental et adaptation comportementale  
**Author:** Ameur Gargouri  
**Notebook:** `08_der_sa_experiment.ipynb`

## Motivation

From our benchmark (Exp03/05), **DER++** is the best continual learning strategy for
shared autonomy, achieving **R²=0.597** (vs Joint Training R²=0.649). But it still
suffers from **systemic forgetting** (R² BWT = −0.15): all 6 joints lose accuracy
together as new participants are learned.

**Key observations driving our design:**
1. Forgetting is **systemic** — 14/15 joint pairs have forgetting correlation r > 0.7
2. The **elbow (J3)** is hardest, the **wrist (J6)** easiest — difficulty varies by joint
3. DER++ treats all replay samples equally, but some are more at risk of being forgotten
4. Standard MLP has no mechanism to separate shared vs participant-specific knowledge

## DER-SA: Three Novel Components

Building on DER++, we add:

1. **Adaptive Replay Weighting (ARW):** Weight buffer samples by forgetting risk:
   samples whose current prediction drifts most from stored logits get more replay
   weight. This focuses rehearsal where forgetting is worst.

2. **Joint-Aware Consistency Loss (JAC):** Exploit the systemic co-forgetting:
   add a loss term penalizing the cross-joint prediction covariance drift on
   replay samples. This stabilizes the joint coupling structure.

3. **Feature-Level Knowledge Distillation (FKD):** Distill from the penultimate
   hidden layer (not just output logits), preserving richer internal representations.

## Primary Metric
- **Overall R²** (pooled across all test samples) — must exceed DER++ baseline of 0.597

---
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — Imports
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""import os, sys, abc, copy, json, logging, random, time, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
from itertools import product

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
# CELL 3 — Metrics & Data Structures (from NB06, unchanged)
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Metrics & Data Structures (identical to previous experiments for fair comparison)

ACCURACY_THRESHOLD = 0.5  # threshold for accuracy: |error| < 0.5 (in normalized space)

@dataclass
class TaskResult:
    task_id: int; loss: float; mse: float; mae: float; r2: float
    n_samples: int = 0
    per_dim_mse: list = field(default_factory=list)
    threshold_accuracy: float = 0.0
    ss_res: float = 0.0
    ss_tot: float = 0.0
    n_correct: int = 0

@dataclass
class ContinualMetrics:
    n_tasks: int = 0
    accuracy_matrix: list = field(default_factory=list)
    r2_matrix: list = field(default_factory=list)
    acc_matrix: list = field(default_factory=list)
    task_names: list = field(default_factory=list)
    per_dim_matrix: list = field(default_factory=list)
    _last_results: list = field(default_factory=list)

    def record(self, trained_up_to, task_results):
        mse_row = [r.mse for r in task_results]
        r2_row = [r.r2 for r in task_results]
        acc_row = [r.threshold_accuracy for r in task_results]
        while len(self.accuracy_matrix) <= trained_up_to:
            self.accuracy_matrix.append([])
            self.r2_matrix.append([])
            self.acc_matrix.append([])
            self.per_dim_matrix.append([])
        self.accuracy_matrix[trained_up_to] = mse_row
        self.r2_matrix[trained_up_to] = r2_row
        self.acc_matrix[trained_up_to] = acc_row
        self.per_dim_matrix[trained_up_to] = [r.per_dim_mse for r in task_results]
        self._last_results = task_results

    @property
    def average_mse(self):
        if not self.accuracy_matrix: return float('nan')
        last = self.accuracy_matrix[-1]
        return float(np.mean(last)) if last else float('nan')

    @property
    def average_r2(self):
        if not self.r2_matrix: return float('nan')
        last = self.r2_matrix[-1]
        return float(np.mean(last)) if last else float('nan')

    @property
    def average_threshold_accuracy(self):
        if not self.acc_matrix: return float('nan')
        last = self.acc_matrix[-1]
        return float(np.mean(last)) if last else float('nan')

    @property
    def overall_r2(self):
        if not self._last_results: return float('nan')
        ss_res = sum(r.ss_res for r in self._last_results)
        ss_tot = sum(r.ss_tot for r in self._last_results)
        return 1 - ss_res / max(ss_tot, 1e-8)

    @property
    def overall_threshold_accuracy(self):
        if not self._last_results: return float('nan')
        total_correct = sum(r.n_correct for r in self._last_results)
        total_samples = sum(r.n_samples for r in self._last_results)
        return total_correct / max(total_samples, 1) * 100

    @property
    def backward_transfer(self):
        T = len(self.accuracy_matrix)
        if T < 2: return 0.0
        bwt, cnt = 0.0, 0
        for j in range(T - 1):
            if j < len(self.accuracy_matrix[T-1]) and j < len(self.accuracy_matrix[j]):
                bwt += self.accuracy_matrix[T-1][j] - self.accuracy_matrix[j][j]; cnt += 1
        return bwt / cnt if cnt > 0 else 0.0

    @property
    def r2_backward_transfer(self):
        T = len(self.r2_matrix)
        if T < 2: return 0.0
        bwt, cnt = 0.0, 0
        for j in range(T - 1):
            if j < len(self.r2_matrix[T-1]) and j < len(self.r2_matrix[j]):
                bwt += self.r2_matrix[T-1][j] - self.r2_matrix[j][j]; cnt += 1
        return bwt / cnt if cnt > 0 else 0.0

    @property
    def forward_transfer(self):
        T = len(self.accuracy_matrix)
        if T < 2: return 0.0
        fwt, cnt = 0.0, 0
        for j in range(1, T):
            if j < len(self.accuracy_matrix[j-1]) and j < len(self.accuracy_matrix[j]):
                fwt += self.accuracy_matrix[j-1][j] - self.accuracy_matrix[j][j]; cnt += 1
        return fwt / cnt if cnt > 0 else 0.0

    def summary_dict(self):
        return {'n_tasks': len(self.accuracy_matrix),
                'average_mse': self.average_mse,
                'average_r2': self.average_r2,
                'average_threshold_accuracy': self.average_threshold_accuracy,
                'overall_r2': self.overall_r2,
                'overall_threshold_accuracy': self.overall_threshold_accuracy,
                'backward_transfer': self.backward_transfer,
                'r2_backward_transfer': self.r2_backward_transfer,
                'forward_transfer': self.forward_transfer,
                'accuracy_matrix': self.accuracy_matrix,
                'r2_matrix': self.r2_matrix,
                'acc_matrix': self.acc_matrix,
                'task_names': self.task_names,
                'per_dim_matrix': self.per_dim_matrix}

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

print('Metrics & TaskData defined')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — Model (modified for feature extraction) + Base CL + Baselines
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Model (with feature extraction hook for FKD) & Base CL

class MLPPolicy(nn.Module):
    # MLP with accessible penultimate features for knowledge distillation
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
        self._feat_dim = d  # penultimate dimension

    def forward(self, x):
        return self.head(self.backbone(x))

    def forward_with_features(self, x):
        # Return (predictions, penultimate_features)
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
        history = {'train_loss': [], 'val_loss': []}
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
            avg = eloss / max(nb, 1)
            history['train_loss'].append(avg)
            if val_loader is not None:
                vl = self._eval_loss(val_loader)
                history['val_loss'].append(vl)
                if vl < best_val:
                    best_val = vl; best_state = copy.deepcopy(self.model.state_dict()); no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
        if best_state is not None: self.model.load_state_dict(best_state)
        self.on_task_end(task_id, train_loader)
        return history

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
        errors = (p - t).abs()
        correct_mask = (errors < ACCURACY_THRESHOLD).all(dim=1)
        within_threshold = correct_mask.float().mean().item() * 100
        n_correct = int(correct_mask.sum().item())
        self.model.train()
        return TaskResult(task_id, mse, mse, mae, r2, len(t), per_dim, within_threshold,
                          ss_res, ss_tot, n_correct)


# --- Baselines (identical to previous experiments) ---

class NaiveFineTune(ContinualLearner):
    def on_task_start(self, t, l): pass
    def compute_loss(self, obs, act, t): return F.mse_loss(self.model(obs), act)
    def on_task_end(self, t, l): pass

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
        return super().train_task(task_id, joint_loader, val_loader, epochs, patience, verbose)


class ReplayBuffer:
    # Reservoir-sampled replay buffer storing (obs, act, logits, features)
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


class DERPlusPlus(ContinualLearner):
    # Standard DER++ baseline (Buzzega et al., NeurIPS 2020)
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


class EWC(ContinualLearner):
    # Online EWC baseline (Schwarz et al., ICML 2018)
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


class AGEM(ContinualLearner):
    # A-GEM baseline (Chaudhry et al., ICLR 2019)
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
        history = {'train_loss': [], 'val_loss': []}
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
            avg = eloss / max(nb, 1)
            history['train_loss'].append(avg)
            if val_loader is not None:
                vl = self._eval_loss(val_loader)
                history['val_loss'].append(vl)
                if vl < best_val:
                    best_val = vl; best_state = copy.deepcopy(self.model.state_dict()); no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
        if best_state is not None: self.model.load_state_dict(best_state)
        self.on_task_end(task_id, train_loader)
        return history

print('Model & Baselines defined: Naive, Joint, DER++, Online EWC, A-GEM')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — DER-SA Algorithm
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# ═══════════════════════════════════════════════════════════════════
# DER-SA: Domain-Aware Replay for Shared Autonomy (OUR METHOD)
# ═══════════════════════════════════════════════════════════════════

class DERSA(ContinualLearner):
    # DER-SA: Dark Experience Replay with domain-aware enhancements
    # for continual shared autonomy.
    #
    # Three novel components on top of DER++:
    #   1. ARW: upweight replay samples whose logit drift is largest
    #   2. JAC: penalize cross-joint covariance drift on replay
    #   3. FKD: distill penultimate features, not just output logits
    #
    # Args: buffer_size, alpha (logit distil), beta (target replay),
    #       gamma_fkd (feature distil), gamma_jac (joint consistency), tau (softmax temp)
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
        self._ref_cov = None  # reference cross-joint covariance

    def on_task_start(self, task_id, train_loader):
        pass

    def on_task_end(self, task_id, train_loader):
        # Update reference covariance from buffer for JAC loss
        if len(self.buffer) > 100:
            sample = self.buffer.sample(min(len(self.buffer), 2000))
            bo = sample[0].to(self.device)
            with torch.no_grad():
                pred = self.model(bo)
            # Cross-joint covariance of predictions
            pred_centered = pred - pred.mean(0)
            self._ref_cov = (pred_centered.T @ pred_centered) / (pred.size(0) - 1)

    def compute_loss(self, obs, act, task_id):
        # --- Current task loss ---
        pred, feat = self.model.forward_with_features(obs)
        loss_current = F.mse_loss(pred, act)

        # Store to buffer with features
        with torch.no_grad():
            logits_snap = self.model(obs).detach()
            _, feat_snap = self.model.forward_with_features(obs)
        self.buffer.add(obs.detach(), act.detach(), logits_snap, feat_snap.detach())

        if len(self.buffer) == 0 or task_id == 0:
            return loss_current

        # --- Replay from buffer ---
        sample = self.buffer.sample(max(1, obs.size(0)))
        bo, ba, bl, bf = sample[0].to(self.device), sample[1].to(self.device), \
                         sample[2].to(self.device), sample[3].to(self.device)

        bp, bp_feat = self.model.forward_with_features(bo)

        # Component 1: Adaptive Replay Weighting (ARW)
        # Weight each replay sample by how much its prediction drifted from stored logits
        with torch.no_grad():
            drift = (bp - bl).pow(2).mean(dim=1)  # per-sample drift [N]
            weights = F.softmax(drift / self.tau, dim=0) * drift.size(0)  # normalized weights, mean ≈ 1

        # Weighted logit distillation (replaces uniform DER++ alpha loss)
        loss_logit = (weights.unsqueeze(1) * (bp - bl).pow(2)).mean()

        # Weighted target replay (replaces uniform DER++ beta loss)
        loss_target = (weights.unsqueeze(1) * (bp - ba).pow(2)).mean()

        # Component 2: Feature-level Knowledge Distillation (FKD)
        loss_fkd = F.mse_loss(bp_feat, bf)

        # Component 3: Joint-Aware Consistency (JAC)
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

print('DER-SA defined')
print('  Components: ARW (adaptive replay weighting) + FKD (feature distillation) + JAC (joint consistency)')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — Data Loading (identical to previous experiments)
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Data Loading -- identical to previous experiments for fair comparison

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
# CELL 7 — Setup & Build Tasks
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
DERSA_BUFFER    = 10000   # same buffer as DER++ for fair comparison
DERSA_ALPHA     = 0.5     # logit distillation (same base as DER++)
DERSA_BETA      = 0.5     # target replay (same base as DER++)
DERSA_GAMMA_FKD = 0.3     # feature distillation weight
DERSA_GAMMA_JAC = 0.2     # joint consistency weight
DERSA_TAU       = 2.0     # adaptive weighting temperature

N_TASKS = 10
obs_dim = 12
act_dim = 6

# Build tasks
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

def run_benchmark(name, learner, task_list, epochs=EPOCHS, patience=PATIENCE, bs=BS, verbose=False):
    metrics = ContinualMetrics(n_tasks=len(task_list),
                               task_names=[t.participant_id for t in task_list])
    t0 = time.time()
    for task in task_list:
        if verbose:
            print(f'  [{name}] Task {task.task_id}/{len(task_list)-1} ({task.participant_id})')
        learner.train_task(task.task_id, task.train_loader(bs),
                          task.val_loader(bs*2), epochs=epochs, patience=patience, verbose=False)
        results = [learner.evaluate_task(et.test_loader(bs*2), et.task_id) for et in task_list]
        metrics.record(task.task_id, results)
    elapsed = time.time() - t0
    return {'name': name, 'metrics': metrics, 'time': elapsed}

# Build and normalize
task_list = build_tasks(N_TASKS, participants)
norm_stats = normalize_tasks(task_list)

print(f'\nReady: {len(task_list)} tasks, {obs_dim}D -> {act_dim}D')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — Markdown: Run Experiments
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""---
## Run Full Comparison

Compare **DER-SA** against all baselines on 10 participants.
Same model architecture, same data, same seed — only the CL strategy differs.
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — Run Experiments
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Run all strategies

strategies = {
    'Naive Fine-Tune': lambda: NaiveFineTune(make_model(), lr=LR, device=DEVICE),
    'Joint Training':  lambda: JointTraining(make_model(), lr=LR, device=DEVICE),
    'DER++':           lambda: DERPlusPlus(make_model(), lr=LR, device=DEVICE, buffer_size=DER_BUFFER),
    'Online EWC':      lambda: EWC(make_model(), lr=LR, device=DEVICE,
                                    ewc_lambda=EWC_LAMBDA, fisher_samples=EWC_FISHER_N,
                                    online=True, gamma=OEWC_GAMMA),
    'A-GEM':           lambda: AGEM(make_model(), lr=LR, device=DEVICE, mem_per_task=AGEM_MEM),
    'DER-SA (ours)':   lambda: DERSA(make_model(), lr=LR, device=DEVICE,
                                      buffer_size=DERSA_BUFFER, alpha=DERSA_ALPHA, beta=DERSA_BETA,
                                      gamma_fkd=DERSA_GAMMA_FKD, gamma_jac=DERSA_GAMMA_JAC,
                                      tau=DERSA_TAU),
}

all_results = {}

for name, make_learner in strategies.items():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print(f'\n{"="*60}')
    print(f'  Running: {name}')
    print(f'{"="*60}')

    learner = make_learner()
    res = run_benchmark(name, learner, task_list, verbose=True)
    m = res['metrics']

    all_results[name] = {
        'overall_r2': m.overall_r2,
        'overall_accuracy': m.overall_threshold_accuracy,
        'avg_r2': m.average_r2,
        'avg_accuracy': m.average_threshold_accuracy,
        'avg_mse': m.average_mse,
        'bwt': m.backward_transfer,
        'r2_bwt': m.r2_backward_transfer,
        'fwt': m.forward_transfer,
        'time': res['time'],
        'summary': m.summary_dict(),
    }

    print(f'  -> Overall R²={m.overall_r2:.4f}  Acc={m.overall_threshold_accuracy:.1f}%  '
          f'MSE={m.average_mse:.4f}  R²-BWT={m.r2_backward_transfer:+.4f}  ({res["time"]:.0f}s)')

# Save
with open(f'{RESULTS_DIR}/full_results.json', 'w') as f:
    # Exclude non-serializable
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != 'summary'} for k, v in all_results.items()}
    json.dump(save_data, f, indent=2, default=str)

print(f'\nAll results saved to {RESULTS_DIR}/full_results.json')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — Results Table
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Results Summary — Ranked by Overall R²

print('='*105)
print('EXPERIMENT 6 — DER-SA vs BASELINES (Ranked by Overall R²)')
print('='*105)
print(f'{"Rank":>4s}  {"Strategy":20s}  {"Ovrl R²":>8s}  {"Ovrl Acc%":>9s}  '
      f'{"Avg R²":>7s}  {"R²-BWT":>8s}  {"MSE":>8s}  {"MSE-BWT":>8s}  {"Time":>8s}')
print('-' * 100)

ranked = sorted(all_results.items(), key=lambda x: x[1]['overall_r2'], reverse=True)

for rank, (name, d) in enumerate(ranked, 1):
    marker = ' ***' if name == 'DER-SA (ours)' else ''
    print(f'{rank:4d}  {name:20s}  {d["overall_r2"]:8.4f}  {d["overall_accuracy"]:8.1f}%  '
          f'{d["avg_r2"]:7.4f}  {d["r2_bwt"]:+8.4f}  '
          f'{d["avg_mse"]:8.4f}  {d["bwt"]:+8.4f}  {d["time"]:7.0f}s{marker}')

# Gap analysis: DER-SA vs DER++ and Joint
print(f'\n{"="*105}')
print('GAP ANALYSIS')
print(f'{"="*105}')
if 'DER-SA (ours)' in all_results and 'DER++' in all_results:
    sa = all_results['DER-SA (ours)']
    dp = all_results['DER++']
    jt = all_results.get('Joint Training', {})
    print(f'  DER-SA vs DER++:')
    print(f'    R² improvement:  {sa["overall_r2"] - dp["overall_r2"]:+.4f} ({(sa["overall_r2"] - dp["overall_r2"])/max(abs(dp["overall_r2"]),1e-8)*100:+.1f}%)')
    print(f'    Acc improvement: {sa["overall_accuracy"] - dp["overall_accuracy"]:+.1f}%')
    print(f'    BWT improvement: {sa["r2_bwt"] - dp["r2_bwt"]:+.4f} (less negative = less forgetting)')
    if jt:
        print(f'  DER-SA vs Joint Training:')
        print(f'    R² gap:          {sa["overall_r2"] - jt["overall_r2"]:+.4f} (negative = still below upper bound)')
        print(f'    Speed ratio:     {jt["time"]/max(sa["time"],1):.1f}x faster')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 11 — Visualization
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Visualization — Bar charts + R² evolution

colors = {
    'Naive Fine-Tune': '#e74c3c', 'Joint Training': '#2ecc71',
    'DER++': '#e67e22', 'Online EWC': '#9b59b6', 'A-GEM': '#34495e',
    'DER-SA (ours)': '#3498db'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# --- Plot 1: Overall R² bar chart ---
ax = axes[0, 0]
ranked_names = [n for n, _ in ranked]
ranked_r2 = [all_results[n]['overall_r2'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], ranked_r2[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Overall R² (higher = better)')
ax.set_title('*** Overall Accuracy (R²) — PRIMARY ***', fontweight='bold', fontsize=12)
ax.axvline(0, color='gray', linewidth=0.5)

# --- Plot 2: Threshold Accuracy ---
ax = axes[0, 1]
ranked_acc = [all_results[n]['overall_accuracy'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], ranked_acc[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Overall Threshold Accuracy (%)')
ax.set_title('Threshold Accuracy (|err| < 0.5 std)')

# --- Plot 3: R² BWT (forgetting) ---
ax = axes[1, 0]
ranked_bwt = [all_results[n]['r2_bwt'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], ranked_bwt[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('R² BWT (closer to 0 = less forgetting)')
ax.set_title('Backward Transfer (Forgetting)')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

# --- Plot 4: Time ---
ax = axes[1, 1]
ranked_time = [all_results[n]['time'] for n in ranked_names]
bars = ax.barh(ranked_names[::-1], ranked_time[::-1],
               color=[colors.get(n, '#888') for n in ranked_names[::-1]], edgecolor='white')
for bar, name in zip(bars, ranked_names[::-1]):
    if name == 'DER-SA (ours)':
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
ax.set_xlabel('Training Time (s)')
ax.set_title('Computational Cost')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/der_sa_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — R² evolution per task
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# R² evolution: how does accuracy on task 0 (p100) change as new tasks are learned?

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

focus_strats = ['Naive Fine-Tune', 'DER++', 'DER-SA (ours)', 'Joint Training']

# Plot 1: Task 0 R² over time
ax = axes[0]
for name in focus_strats:
    if name in all_results and 'summary' in all_results[name]:
        r2_mat = all_results[name]['summary']['r2_matrix']
        task0_r2 = [r2_mat[i][0] for i in range(len(r2_mat)) if len(r2_mat[i]) > 0]
        ax.plot(range(len(task0_r2)), task0_r2, 'o-', label=name,
                color=colors.get(name, '#888'), linewidth=2, markersize=6)
ax.set_xlabel('After training on task #')
ax.set_ylabel('R² on Task 0 (p100)')
ax.set_title('Forgetting Trajectory: Task 0 (p100)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Average R² across all seen tasks
ax = axes[1]
for name in focus_strats:
    if name in all_results and 'summary' in all_results[name]:
        r2_mat = all_results[name]['summary']['r2_matrix']
        avg_r2 = []
        for i in range(len(r2_mat)):
            if r2_mat[i]:
                avg_r2.append(np.mean(r2_mat[i][:i+1]))  # avg over tasks seen so far
        ax.plot(range(len(avg_r2)), avg_r2, 'o-', label=name,
                color=colors.get(name, '#888'), linewidth=2, markersize=6)
ax.set_xlabel('After training on task #')
ax.set_ylabel('Avg R² on all tasks seen so far')
ax.set_title('Average Accuracy Evolution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/r2_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 13 — Ablation Study
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""---
## Ablation Study

Test each DER-SA component individually to measure its contribution.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Ablation: DER++ + each component alone, then all together

ablation_configs = {
    'DER++ (baseline)':     {'gamma_fkd': 0.0, 'gamma_jac': 0.0, 'tau': 2.0},    # No new components
    'DER++ + ARW only':     {'gamma_fkd': 0.0, 'gamma_jac': 0.0, 'tau': 2.0},    # ARW is always on via tau
    'DER++ + FKD only':     {'gamma_fkd': 0.3, 'gamma_jac': 0.0, 'tau': 999.0},  # tau>>1 = uniform weights
    'DER++ + JAC only':     {'gamma_fkd': 0.0, 'gamma_jac': 0.2, 'tau': 999.0},
    'DER++ + ARW + FKD':    {'gamma_fkd': 0.3, 'gamma_jac': 0.0, 'tau': 2.0},
    'DER++ + ARW + JAC':    {'gamma_fkd': 0.0, 'gamma_jac': 0.2, 'tau': 2.0},
    'DER++ + FKD + JAC':    {'gamma_fkd': 0.3, 'gamma_jac': 0.2, 'tau': 999.0},
    'DER-SA (full)':        {'gamma_fkd': 0.3, 'gamma_jac': 0.2, 'tau': 2.0},
}

ablation_results = {}

for name, cfg in ablation_configs.items():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print(f'\nAblation: {name}  (fkd={cfg["gamma_fkd"]}, jac={cfg["gamma_jac"]}, tau={cfg["tau"]})')

    # For "DER++ (baseline)", use standard DER++
    if name == 'DER++ (baseline)':
        learner = DERPlusPlus(make_model(), lr=LR, device=DEVICE, buffer_size=DER_BUFFER)
    elif name == 'DER++ + ARW only':
        # ARW only = DER-SA with no FKD/JAC but with adaptive weighting
        learner = DERSA(make_model(), lr=LR, device=DEVICE,
                        buffer_size=DER_BUFFER, gamma_fkd=0.0, gamma_jac=0.0, tau=2.0)
    else:
        learner = DERSA(make_model(), lr=LR, device=DEVICE,
                        buffer_size=DER_BUFFER, gamma_fkd=cfg['gamma_fkd'],
                        gamma_jac=cfg['gamma_jac'], tau=cfg['tau'])

    res = run_benchmark(name, learner, task_list, verbose=False)
    m = res['metrics']
    ablation_results[name] = {
        'overall_r2': m.overall_r2,
        'overall_accuracy': m.overall_threshold_accuracy,
        'r2_bwt': m.r2_backward_transfer,
        'avg_mse': m.average_mse,
        'time': res['time'],
    }
    print(f'  -> R²={m.overall_r2:.4f}  Acc={m.overall_threshold_accuracy:.1f}%  '
          f'BWT={m.r2_backward_transfer:+.4f}  ({res["time"]:.0f}s)')

# Ablation table
print(f'\n{"="*90}')
print('ABLATION STUDY RESULTS')
print(f'{"="*90}')
print(f'{"Configuration":25s}  {"Ovrl R²":>8s}  {"Acc%":>6s}  {"R²-BWT":>8s}  {"MSE":>8s}  {"Time":>7s}')
print('-' * 75)
for name, d in ablation_results.items():
    print(f'{name:25s}  {d["overall_r2"]:8.4f}  {d["overall_accuracy"]:5.1f}%  '
          f'{d["r2_bwt"]:+8.4f}  {d["avg_mse"]:8.4f}  {d["time"]:6.0f}s')

# Save ablation
with open(f'{RESULTS_DIR}/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2, default=str)
print(f'\nSaved to {RESULTS_DIR}/ablation_results.json')
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — Ablation visualization
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Ablation Visualization

fig, ax = plt.subplots(figsize=(12, 6))
names = list(ablation_results.keys())
r2_vals = [ablation_results[n]['overall_r2'] for n in names]

bar_colors = ['#e67e22' if 'baseline' in n else '#3498db' if 'full' in n else '#95a5a6' for n in names]
bars = ax.barh(names[::-1], r2_vals[::-1], color=bar_colors[::-1], edgecolor='white')

# Highlight full DER-SA
for bar, name in zip(bars, names[::-1]):
    if 'full' in name:
        bar.set_edgecolor('#2c3e50'); bar.set_linewidth(2.5)
    elif 'baseline' in name:
        bar.set_edgecolor('#d35400'); bar.set_linewidth(2)

ax.set_xlabel('Overall R²')
ax.set_title('Ablation Study: Contribution of Each DER-SA Component', fontweight='bold')
ax.axvline(r2_vals[0], color='#e67e22', linewidth=1.5, linestyle='--', alpha=0.7, label='DER++ baseline')
ax.legend()
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/ablation_chart.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════
# CELL 15 — Per-Joint Analysis
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Per-Joint MSE comparison: DER++ vs DER-SA

joint_names = ['J1 (shoulder)', 'J2 (shoulder)', 'J3 (elbow)',
               'J4 (wrist)', 'J5 (wrist)', 'J6 (wrist)']

fig, ax = plt.subplots(figsize=(12, 5))

if 'DER++' in all_results and 'DER-SA (ours)' in all_results:
    dp_per_dim = all_results['DER++']['summary']['per_dim_matrix'][-1]
    sa_per_dim = all_results['DER-SA (ours)']['summary']['per_dim_matrix'][-1]

    # Average per-dim MSE across tasks
    dp_avg = np.mean(dp_per_dim, axis=0)
    sa_avg = np.mean(sa_per_dim, axis=0)

    x = np.arange(len(joint_names))
    w = 0.35
    ax.bar(x - w/2, dp_avg, w, label='DER++', color='#e67e22', alpha=0.85)
    ax.bar(x + w/2, sa_avg, w, label='DER-SA (ours)', color='#3498db', alpha=0.85)

    # Add improvement labels
    for i in range(len(joint_names)):
        imp = (dp_avg[i] - sa_avg[i]) / dp_avg[i] * 100
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
# CELL 16 — Save config
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell(r"""# Save experiment config

config = {
    'experiment': 'der_sa_comparison',
    'date': time.strftime('%Y-%m-%d'),
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
# CELL 17 — Summary
# ═══════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell(r"""---
## Summary

*(to be filled after execution)*

### Overall Results (Ranked by R²)

| Rank | Strategy | Overall R² | Acc% | R²-BWT | MSE | Time |
|------|----------|-----------|------|--------|-----|------|
| 1 | ? | ? | ? | ? | ? | ? |
| 2 | ? | ? | ? | ? | ? | ? |
| 3 | ? | ? | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? | ? | ? |
| 5 | ? | ? | ? | ? | ? | ? |
| 6 | ? | ? | ? | ? | ? | ? |

### DER-SA Improvement over DER++
- R² improvement: ?
- Accuracy improvement: ?
- Forgetting reduction (BWT): ?
- Speed overhead: ?

### Ablation: Component Contributions
- ARW alone: ?
- FKD alone: ?
- JAC alone: ?
- Full DER-SA: ?

### Key Findings
1. ?
2. ?
3. ?
"""))

nb.cells = cells

# Write notebook
out_path = '/home/g0amer/Desktop/thesis/phd_project/notebooks/08_der_sa_experiment.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)

print(f'Notebook created: {out_path}')
print(f'  {len(cells)} cells ({sum(1 for c in cells if c.cell_type == "code")} code, '
      f'{sum(1 for c in cells if c.cell_type == "markdown")} markdown)')
