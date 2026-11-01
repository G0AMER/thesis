"""Training loops: supervised, feature-alignment UDA (DANN / CORAL / MMD / PI-CDA) and FixMatch.

Model selection never looks at target labels: every run trains for a fixed number of steps
and the final weights are evaluated. This is the honest UDA protocol; picking the epoch with
the best target score would leak test labels.
"""
import math
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import worker_init_fn
from .losses import DomainDiscriminator, coral_loss, grad_reverse, mmd_loss, topology_consistency_loss

# Benign: with AMP, GradScaler skips the optimiser step when the first fp16 gradients overflow,
# so the scheduler is stepped once without an optimiser step. Nothing is mis-scheduled.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")


@dataclass
class TrainConfig:
    # data
    model: str = "ctrgcn"
    joint_set: str = "body19"
    features: str = "joint"
    window: int = 64
    canon_center: str = "first_frame"
    canon_yaw: str = "first_frame"
    canon_scale: str = "torso"
    source_aug: str = "weak"       # key of data.SOURCE_AUGS, used for InHARD-DT training clips
    freeze_left: bool = False      # apply data.freeze_left_arm to every clip, both domains
    # optimisation (CTR-GCN recipe, shortened for ~5k clips)
    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.05
    weight_decay: float = 4e-4
    momentum: float = 0.9
    warmup_epochs: int = 5
    dropout: float = 0.0
    num_workers: int = 4
    amp: bool = True
    device: str = "cuda"
    # adaptation weights
    lambda_mmd: float = 0.5
    lambda_topo: float = 0.1
    lambda_coral: float = 1.0
    lambda_dann: float = 1.0
    # FixMatch (initialised from the source-only checkpoint)
    fixmatch_threshold: float = 0.9
    lambda_fixmatch: float = 1.0
    adapt_epochs: int = 30
    adapt_lr: float = 0.01
    # few-shot fine-tuning
    fewshot_ks: tuple = (1, 5, 10)
    fewshot_steps: int = 300
    fewshot_lr: float = 0.01
    fewshot_inits: tuple = ("scratch", "source_only", "picda", "joint", "joint_fixmatch", "joint_proto")
    lambda_proto: float = 1.0  # joint_proto: class-conditional alignment of DT and real-shot prototypes
    # protocol
    n_folds: int = 4
    seeds: tuple = (0, 1, 2)
    save_features: bool = True
    log_every: int = 200
    extra: Dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["fewshot_ks"], d["seeds"] = list(self.fewshot_ks), list(self.seeds)
        d["fewshot_inits"] = list(self.fewshot_inits)
        return d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU (this will be slow)")
        return torch.device("cpu")
    return torch.device(cfg.device)


def make_loader(ds, cfg: TrainConfig, shuffle: bool, batch_size: Optional[int] = None):
    bs = min(batch_size or cfg.batch_size, len(ds))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle and len(ds) >= bs,
                      num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
                      persistent_workers=cfg.num_workers > 0)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def make_optimizer(params, lr: float, cfg: TrainConfig, total_steps: int, warmup_steps: int):
    opt = torch.optim.SGD(params, lr=lr, momentum=cfg.momentum, nesterov=True, weight_decay=cfg.weight_decay)

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


class _Stepper:
    """Shared AMP / optimiser / scheduler bookkeeping."""

    def __init__(self, params, lr, cfg, steps, warmup_steps, device):
        self.opt, self.sched = make_optimizer(params, lr, cfg, steps, warmup_steps)
        self.amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.device = device

    def autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.amp)

    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()


def _log(prefix, step, steps, t0, **vals):
    msg = " | ".join(f"{k}={v:.4f}" for k, v in vals.items())
    print(f"  [{prefix}] step {step}/{steps} | {msg} | {time.time() - t0:.0f}s", flush=True)


def train_supervised(model, ds, cfg: TrainConfig, device, epochs: Optional[int] = None,
                     steps: Optional[int] = None, lr: Optional[float] = None, tag: str = "sup"):
    loader = make_loader(ds, cfg, shuffle=True)
    if steps is None:
        steps = (epochs or cfg.epochs) * max(1, len(loader))
    warm = min(steps // 10, cfg.warmup_epochs * max(1, len(loader)))
    st = _Stepper(model.parameters(), lr or cfg.lr, cfg, steps, warm, device)
    model.train()
    it, t0 = infinite(loader), time.time()
    run_loss, run_acc = 0.0, 0.0
    for step in range(1, steps + 1):
        x, y, _ = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with st.autocast():
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float(), y)
        st.step(loss)
        run_loss += loss.item()
        run_acc += (logits.argmax(1) == y).float().mean().item()
        if step % cfg.log_every == 0 or step == steps:
            n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
            _log(tag, step, steps, t0, loss=run_loss / n, train_acc=run_acc / n)
            run_loss, run_acc = 0.0, 0.0
    return model


def train_uda(model, src_ds, tgt_ds, method: str, cfg: TrainConfig, device, tag: Optional[str] = None):
    """method in {'dann', 'coral', 'mmd', 'picda'}. tgt_ds must hide labels."""
    src_loader = make_loader(src_ds, cfg, shuffle=True)
    tgt_loader = make_loader(tgt_ds, cfg, shuffle=True)
    steps = cfg.epochs * len(src_loader)
    warm = cfg.warmup_epochs * len(src_loader)
    disc = DomainDiscriminator(model.feature_dim).to(device) if method == "dann" else None
    params = list(model.parameters()) + (list(disc.parameters()) if disc is not None else [])
    st = _Stepper(params, cfg.lr, cfg, steps, warm, device)
    if method == "picda":
        model.set_record_topology(True)
    model.train()
    if disc is not None:
        disc.train()
    s_it, t_it, t0 = infinite(src_loader), infinite(tgt_loader), time.time()
    acc = {"cls": 0.0, "align": 0.0}
    try:
        for step in range(1, steps + 1):
            xs, ys, _ = next(s_it)
            xt, _, _ = next(t_it)
            xs, ys, xt = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True), xt.to(device, non_blocking=True)
            ns = xs.size(0)
            with st.autocast():
                logits, feats = model(torch.cat([xs, xt]))
            logits, feats = logits.float(), feats.float()
            loss_cls = F.cross_entropy(logits[:ns], ys)
            if method == "mmd":
                align = cfg.lambda_mmd * mmd_loss(feats[:ns], feats[ns:])
            elif method == "coral":
                align = cfg.lambda_coral * coral_loss(feats[:ns], feats[ns:])
            elif method == "dann":
                p = step / steps
                coeff = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                d_logit = disc(grad_reverse(feats, coeff))
                d_true = torch.cat([torch.zeros(ns, device=device), torch.ones(feats.size(0) - ns, device=device)])
                align = cfg.lambda_dann * F.binary_cross_entropy_with_logits(d_logit, d_true)
            elif method == "picda":
                topos = model.pop_topologies()
                align = cfg.lambda_mmd * mmd_loss(feats[:ns], feats[ns:]) + \
                    cfg.lambda_topo * topology_consistency_loss(topos, ns)
            else:
                raise ValueError(method)
            st.step(loss_cls + align)
            acc["cls"] += loss_cls.item()
            acc["align"] += align.item()
            if step % cfg.log_every == 0 or step == steps:
                n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
                _log(tag or method, step, steps, t0, ce=acc["cls"] / n, align=acc["align"] / n)
                acc = {"cls": 0.0, "align": 0.0}
    finally:
        if method == "picda":
            model.set_record_topology(False)
    return model


def train_fixmatch(model, src_ds, tgt_two_view_ds, cfg: TrainConfig, device, tag: str = "fixmatch",
                   shots_ds=None, proto: bool = False):
    """Source CE + FixMatch consistency on target (Sohn et al., NeurIPS 2020).

    tgt_two_view_ds yields (weak, strong, -1, idx); pseudo-labels come from the weak view. It may be
    None (no unlabeled term). shots_ds, if given, adds a few labeled target clips to every
    supervised batch (semi-supervised domain adaptation: DT + k real shots + unlabeled real).
    proto adds a class-conditional alignment term: per class present in both, the squared distance
    between the batch means of L2-normalised DT features and real-shot features. The diagnostics
    showed the gap is a per-class motion shift, which marginal alignment (MMD/DANN) cannot fix.
    """
    src_loader = make_loader(src_ds, cfg, shuffle=True)
    steps = cfg.adapt_epochs * len(src_loader)
    st = _Stepper(model.parameters(), cfg.adapt_lr, cfg, steps, len(src_loader), device)
    model.train()
    s_it, t0 = infinite(src_loader), time.time()
    t_it = infinite(make_loader(tgt_two_view_ds, cfg, shuffle=True)) if tgt_two_view_ds is not None else None
    k_it = infinite(make_loader(shots_ds, cfg, shuffle=True, batch_size=32)) if shots_ds is not None else None
    acc = {"ce": 0.0, "unsup": 0.0, "mask": 0.0}
    for step in range(1, steps + 1):
        xs, ys, _ = next(s_it)
        n_src = xs.size(0)
        if k_it is not None:
            xk, yk, _ = next(k_it)
            xs, ys = torch.cat([xs, xk]), torch.cat([ys, yk])
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        ns = xs.size(0)
        unsup, mask = torch.zeros((), device=device), torch.zeros(1, device=device)
        if t_it is not None:
            xw, xstr, _, _ = next(t_it)
            xw, xstr = xw.to(device, non_blocking=True), xstr.to(device, non_blocking=True)
            # Pseudo-labels from the weak view in a separate no-grad pass: the graph then only
            # holds 2 x batch clips instead of 3 x, which fits a 6 GB laptop GPU.
            with torch.no_grad(), st.autocast():
                weak_logits, _ = model(xw)
            conf, pseudo = weak_logits.float().softmax(1).max(1)
            mask = (conf >= cfg.fixmatch_threshold).float()
            xs_all = torch.cat([xs, xstr])
        else:
            xs_all = xs
        with st.autocast():
            logits, feats = model(xs_all)
        logits = logits.float()
        ce = F.cross_entropy(logits[:ns], ys)
        if t_it is not None:
            unsup = (F.cross_entropy(logits[ns:], pseudo, reduction="none") * mask).mean()
        loss = ce + cfg.lambda_fixmatch * unsup
        if proto and ns > n_src:
            f = F.normalize(feats[:ns].float(), dim=1)
            fs, ysrc, fk, yk_ = f[:n_src], ys[:n_src], f[n_src:], ys[n_src:]
            terms = [((fs[ysrc == c].mean(0) - fk[yk_ == c].mean(0)) ** 2).sum()
                     for c in torch.unique(yk_) if (ysrc == c).any()]
            if terms:
                loss = loss + cfg.lambda_proto * torch.stack(terms).mean()
        st.step(loss)
        acc["ce"] += ce.item()
        acc["unsup"] += unsup.item()
        acc["mask"] += mask.mean().item()
        if step % cfg.log_every == 0 or step == steps:
            n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
            _log(tag, step, steps, t0, ce=acc["ce"] / n, unsup=acc["unsup"] / n, mask_rate=acc["mask"] / n)
            acc = {"ce": 0.0, "unsup": 0.0, "mask": 0.0}
    return model


@torch.no_grad()
def predict(model, ds, cfg: TrainConfig, device, return_features: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    if hasattr(model, "set_record_topology"):
        model.set_record_topology(False)
    loader = DataLoader(ds, batch_size=max(cfg.batch_size, 128), shuffle=False, num_workers=cfg.num_workers)
    idx, y_true, probs, feats = [], [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits, f = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu())
        idx.append(i)
        y_true.append(y)
        if return_features:
            feats.append(f.float().cpu())
    out = {"idx": torch.cat(idx).numpy(), "y_true": torch.cat(y_true).numpy(),
           "probs": torch.cat(probs).numpy()}
    out["y_pred"] = out["probs"].argmax(1)
    if return_features:
        out["features"] = torch.cat(feats).numpy()
    return out
