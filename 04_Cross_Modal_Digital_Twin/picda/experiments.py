"""Benchmark runner. Every run writes one JSON with its raw predictions, so all tables and
statistics are recomputed from predictions in notebook 04 (never typed in by hand).

Protocol (InHARD-DT -> InHARD):
  * Target subjects are split into `n_folds` subject-disjoint folds. For fold f, the other
    subjects are the target *training* pool (used unlabeled by UDA, labeled by the oracle)
    and fold f subjects are the *test* set, never seen in any form during training.
  * Source = all InHARD-DT clips. Training length is fixed; no target labels are used for
    model selection.

Methods (result folder names):
  source_only          train on DT, test on InHARD (once per seed, scored per fold)
  reverse_zero_shot    train on all InHARD, test on all DT (fold = all)
  dt_in_domain         DT subject-disjoint folds, DT->DT (gap reference)
  oracle               supervised on InHARD training subjects
  source_plus_target   supervised on DT + labeled InHARD training subjects
  dann | coral | mmd   classical UDA baselines
  picda                MMD + topology consistency (the legacy proposal, now tested honestly)
  fixmatch             source CE + FixMatch on target, initialised from source_only
  fewshot_k{K}_{init}  K labeled clips/class from InHARD training subjects,
                       init in {scratch, source_only, picda}
"""
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from . import config as C
from .data import (FEATURE_CHANNELS, NO_AUG, SOURCE_AUGS, STRONG_AUG, WEAK_AUG, Canon, SkeletonDataset,
                   few_shot_indices, freeze_left_arm, load_clips, load_index, load_splits, make_splits)
from .metrics import classification_metrics
from .models import build_model
from .skeleton import spatial_graph
from .train import (TrainConfig, get_device, predict, set_seed, train_fixmatch, train_supervised,
                    train_uda)

UDA_METHODS = ("dann", "coral", "mmd", "picda")
ALL_METHODS = ("source_only", "reverse_zero_shot", "dt_in_domain", "oracle", "source_plus_target",
               *UDA_METHODS, "fixmatch", "fewshot")
FEATURE_METHODS = {"source_only", "oracle", "dann", "coral", "mmd", "picda", "fixmatch"}


class Benchmark:
    def __init__(self, cfg: TrainConfig, out_dir):
        self.cfg = cfg
        self.out = Path(out_dir)
        for sub in ("results", "checkpoints", "features"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.device = get_device(cfg)

        index = load_index()
        self.dt = index[index.domain == C.DOMAIN_DT].reset_index(drop=True)
        self.real = index[index.domain == C.DOMAIN_REAL].reset_index(drop=True)
        canon = Canon(cfg.canon_center, cfg.canon_yaw, cfg.canon_scale)
        t0 = time.time()
        self.clips_dt = load_clips(self.dt, cfg.joint_set, canon)
        self.clips_real = load_clips(self.real, cfg.joint_set, canon)
        self.y_dt, self.y_real = self.dt.label.values, self.real.label.values
        print(f"Loaded {len(self.clips_dt)} DT clips ({self.dt.subject.nunique()} subjects) and "
              f"{len(self.clips_real)} InHARD clips ({self.real.subject.nunique()} subjects) "
              f"in {time.time() - t0:.0f}s")

        splits = load_splits() if C.SPLITS_JSON.exists() else None
        if splits is None or len(splits["real_test_folds"]) != cfg.n_folds:
            splits = make_splits(index, cfg.n_folds, cfg.n_folds)
        self.real_folds, self.dt_folds = splits["real_test_folds"], splits["dt_test_folds"]
        self.A = spatial_graph(cfg.joint_set)
        self.in_ch = FEATURE_CHANNELS[cfg.features]
        with open(self.out / "config.json", "w") as f:
            json.dump({"train_config": cfg.to_dict(), "splits": splits}, f, indent=1)

    # ------------------------------------------------------------------ helpers
    def _ds(self, domain, indices, views=(NO_AUG,), hide_labels=False):
        clips, labels = (self.clips_dt, self.y_dt) if domain == C.DOMAIN_DT else (self.clips_real, self.y_real)
        transform = freeze_left_arm if self.cfg.freeze_left else None
        return SkeletonDataset(clips, labels, indices, self.cfg.window, views, self.cfg.features,
                               self.cfg.joint_set, hide_labels, transform)

    def _train_ds(self, domain, indices):
        aug = SOURCE_AUGS[self.cfg.source_aug] if domain == C.DOMAIN_DT else WEAK_AUG
        return self._ds(domain, indices, views=(aug,))

    def _new_model(self):
        return build_model(self.cfg.model, C.NUM_CLASSES, self.A, self.in_ch, self.cfg.dropout).to(self.device)

    def _real_split(self, fold):
        test = self.real.subject.isin(set(self.real_folds[fold])).values
        return np.where(~test)[0], np.where(test)[0]

    def _dt_split(self, fold):
        test = self.dt.subject.isin(set(self.dt_folds[fold])).values
        return np.where(~test)[0], np.where(test)[0]

    def _path(self, method, fold, seed):
        return self.out / "results" / method / f"fold{fold}_seed{seed}.json"

    def _ckpt(self, name):
        return self.out / "checkpoints" / f"{name}.pt"

    def _save_result(self, method, fold, seed, domain, pred, extra=None, subset: Optional[np.ndarray] = None):
        df = self.real if domain == C.DOMAIN_REAL else self.dt
        idx, yt, yp = pred["idx"], pred["y_true"], pred["y_pred"]
        if subset is not None:
            keep = np.isin(idx, subset)
            idx, yt, yp = idx[keep], yt[keep], yp[keep]
        rows = df.iloc[idx]
        res = {
            "method": method, "fold": fold, "seed": seed, "test_domain": domain,
            "metrics": classification_metrics(yt, yp),
            "clip_id": rows.clip_id.tolist(), "subject": rows.subject.tolist(),
            "y_true": yt.tolist(), "y_pred": yp.tolist(),
            "extra": extra or {}, "config": self.cfg.to_dict(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        p = self._path(method, fold, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(res, f)
        m = res["metrics"]
        print(f"  -> {method} fold={fold} seed={seed}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"(n={m['n']})", flush=True)

    def _save_features(self, model, method, fold, seed, test_idx):
        if not (self.cfg.save_features and seed == self.cfg.seeds[0] and fold == 0 and method in FEATURE_METHODS):
            return
        src = predict(model, self._ds(C.DOMAIN_DT, np.arange(len(self.dt))), self.cfg, self.device, True)
        tgt = predict(model, self._ds(C.DOMAIN_REAL, test_idx), self.cfg, self.device, True)
        np.savez_compressed(self.out / "features" / f"{method}_fold{fold}_seed{seed}.npz",
                            feats_src=src["features"], y_src=src["y_true"], pred_src=src["y_pred"],
                            feats_tgt=tgt["features"], y_tgt=tgt["y_true"], pred_tgt=tgt["y_pred"],
                            subj_tgt=self.real.subject.values[tgt["idx"]])

    # ------------------------------------------------------------------ runs
    def run_source_only(self, seed, folds):
        ck = self._ckpt(f"source_only_seed{seed}")
        if all(self._path("source_only", f, seed).exists() for f in folds) and ck.exists():
            return
        set_seed(seed)
        model = self._new_model()
        t0 = time.time()
        if ck.exists():
            model.load_state_dict(torch.load(ck, map_location=self.device))
        else:
            train_supervised(model, self._train_ds(C.DOMAIN_DT, np.arange(len(self.dt))), self.cfg,
                             self.device, tag=f"source_only s{seed}")
            torch.save(model.state_dict(), ck)
        pred = predict(model, self._ds(C.DOMAIN_REAL, np.arange(len(self.real))), self.cfg, self.device)
        extra = {"train_time_s": time.time() - t0}
        self._save_result("source_only", "all", seed, C.DOMAIN_REAL, pred, extra)
        for f in folds:
            _, test = self._real_split(f)
            self._save_result("source_only", f, seed, C.DOMAIN_REAL, pred, extra, subset=test)
            self._save_features(model, "source_only", f, seed, test)

    def run_reverse(self, seed):
        if self._path("reverse_zero_shot", "all", seed).exists():
            return
        set_seed(seed)
        model = self._new_model()
        t0 = time.time()
        train_supervised(model, self._train_ds(C.DOMAIN_REAL, np.arange(len(self.real))), self.cfg,
                         self.device, tag=f"reverse s{seed}")
        pred = predict(model, self._ds(C.DOMAIN_DT, np.arange(len(self.dt))), self.cfg, self.device)
        self._save_result("reverse_zero_shot", "all", seed, C.DOMAIN_DT, pred, {"train_time_s": time.time() - t0})

    def run_dt_in_domain(self, seed, folds):
        for f in folds:
            if f >= len(self.dt_folds) or self._path("dt_in_domain", f, seed).exists():
                continue
            set_seed(seed)
            tr, te = self._dt_split(f)
            model = self._new_model()
            t0 = time.time()
            train_supervised(model, self._train_ds(C.DOMAIN_DT, tr), self.cfg, self.device, tag=f"dt_in_domain f{f} s{seed}")
            pred = predict(model, self._ds(C.DOMAIN_DT, te), self.cfg, self.device)
            self._save_result("dt_in_domain", f, seed, C.DOMAIN_DT, pred,
                              {"train_time_s": time.time() - t0, "test_subjects": self.dt_folds[f]})

    def run_fold(self, method, fold, seed):
        if self._path(method, fold, seed).exists():
            return
        set_seed(seed)
        tr, te = self._real_split(fold)
        model = self._new_model()
        t0 = time.time()
        tag = f"{method} f{fold} s{seed}"
        if method == "oracle":
            train_supervised(model, self._train_ds(C.DOMAIN_REAL, tr), self.cfg, self.device, tag=tag)
        elif method == "source_plus_target":
            ds = ConcatDataset([self._train_ds(C.DOMAIN_DT, np.arange(len(self.dt))),
                                self._train_ds(C.DOMAIN_REAL, tr)])
            train_supervised(model, ds, self.cfg, self.device, tag=tag)
        elif method in UDA_METHODS:
            src = self._train_ds(C.DOMAIN_DT, np.arange(len(self.dt)))
            tgt = self._ds(C.DOMAIN_REAL, tr, views=(WEAK_AUG,), hide_labels=True)
            train_uda(model, src, tgt, method, self.cfg, self.device, tag=tag)
            if method == "picda":
                torch.save(model.state_dict(), self._ckpt(f"picda_fold{fold}_seed{seed}"))
        elif method == "fixmatch":
            ck = self._ckpt(f"source_only_seed{seed}")
            if not ck.exists():
                print(f"  skip fixmatch f{fold} s{seed}: run source_only first")
                return
            model.load_state_dict(torch.load(ck, map_location=self.device))
            src = self._train_ds(C.DOMAIN_DT, np.arange(len(self.dt)))
            tgt = self._ds(C.DOMAIN_REAL, tr, views=(WEAK_AUG, STRONG_AUG), hide_labels=True)
            train_fixmatch(model, src, tgt, self.cfg, self.device, tag=tag)
        else:
            raise ValueError(method)
        pred = predict(model, self._ds(C.DOMAIN_REAL, te), self.cfg, self.device)
        self._save_result(method, fold, seed, C.DOMAIN_REAL, pred,
                          {"train_time_s": time.time() - t0, "test_subjects": self.real_folds[fold],
                           "n_target_train": int(len(tr))})
        self._save_features(model, method, fold, seed, te)

    def run_fewshot(self, fold, seed):
        tr, te = self._real_split(fold)
        for k in self.cfg.fewshot_ks:
            shots = few_shot_indices(self.y_real, tr, k, seed=seed * 1000 + fold * 10 + k)
            # joint*: DT + shots (+ FixMatch on unlabeled training subjects), from the source-only model
            for init in self.cfg.fewshot_inits:
                name = f"fewshot_k{k}_{init}"
                if self._path(name, fold, seed).exists():
                    continue
                set_seed(seed)
                model = self._new_model()
                lr = self.cfg.lr
                if init != "scratch":
                    ck = self._ckpt(f"picda_fold{fold}_seed{seed}" if init == "picda" else f"source_only_seed{seed}")
                    if not ck.exists():
                        print(f"  skip {name} f{fold} s{seed}: missing {ck.name}")
                        continue
                    model.load_state_dict(torch.load(ck, map_location=self.device))
                    lr = self.cfg.fewshot_lr
                t0 = time.time()
                if init.startswith("joint"):
                    tgt = (self._ds(C.DOMAIN_REAL, tr, views=(WEAK_AUG, STRONG_AUG), hide_labels=True)
                           if init == "joint_fixmatch" else None)
                    train_fixmatch(model, self._train_ds(C.DOMAIN_DT, np.arange(len(self.dt))), tgt, self.cfg,
                                   self.device, tag=f"{name} f{fold} s{seed}",
                                   shots_ds=self._train_ds(C.DOMAIN_REAL, shots), proto=init.startswith("joint_proto"))
                    if init.endswith("_ft"):  # stage 2: short fine-tune on the shots alone (helps at large k)
                        train_supervised(model, self._train_ds(C.DOMAIN_REAL, shots), self.cfg, self.device,
                                         steps=self.cfg.fewshot_steps, lr=self.cfg.fewshot_lr, tag=f"{name} ft")
                else:
                    train_supervised(model, self._train_ds(C.DOMAIN_REAL, shots), self.cfg, self.device,
                                     steps=self.cfg.fewshot_steps, lr=lr, tag=f"{name} f{fold} s{seed}")
                pred = predict(model, self._ds(C.DOMAIN_REAL, te), self.cfg, self.device)
                self._save_result(name, fold, seed, C.DOMAIN_REAL, pred,
                                  {"train_time_s": time.time() - t0, "n_shots": int(len(shots)),
                                   "shot_clip_ids": self.real.clip_id.values[shots].tolist()})

    def run(self, methods: Iterable[str] = ALL_METHODS, folds: Optional[Iterable[int]] = None,
            seeds: Optional[Iterable[int]] = None):
        methods = list(methods)
        unknown = set(methods) - set(ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown methods {unknown}")
        folds = list(range(self.cfg.n_folds)) if folds is None else list(folds)
        seeds = list(self.cfg.seeds) if seeds is None else list(seeds)
        t_all = time.time()
        self.failures = []
        for seed in seeds:
            print(f"\n================ seed {seed} ================", flush=True)
            if "source_only" in methods:
                self.run_source_only(seed, folds)
            if "reverse_zero_shot" in methods:
                self.run_reverse(seed)
            if "dt_in_domain" in methods:
                self.run_dt_in_domain(seed, folds)
            for fold in folds:
                for m in ("oracle", "source_plus_target", *UDA_METHODS, "fixmatch"):
                    if m in methods:
                        self._guarded(self.run_fold, m, fold, seed)
                if "fewshot" in methods:
                    self._guarded(self.run_fewshot, fold, seed)
        if self.failures:
            print(f"\n{len(self.failures)} run(s) FAILED and were skipped (re-run to retry):")
            for f in self.failures:
                print("   ", f)
        print(f"\nAll requested runs finished in {(time.time() - t_all) / 60:.1f} min. Results in {self.out}")

    def _guarded(self, fn, *args):
        """Run one experiment; on error log it, free GPU memory and continue with the next one."""
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 (an overnight run must not die on one failure)
            import traceback
            traceback.print_exc()
            self.failures.append(f"{fn.__name__}{args}: {type(e).__name__}: {e}"[:300])
            torch.cuda.empty_cache()
