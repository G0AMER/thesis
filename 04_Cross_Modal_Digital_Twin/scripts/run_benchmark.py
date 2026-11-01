"""Terminal entry point for long runs (same code path as notebook 03).

Example (from the project root, inside the venv):
    nohup python scripts/run_benchmark.py --out output_v2/benchmark_ctrgcn > bench.log 2>&1 &
    python scripts/run_benchmark.py --smoke            # 1-epoch end-to-end check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picda.config import OUTPUT_DIR  # noqa: E402
from picda.experiments import ALL_METHODS, Benchmark  # noqa: E402
from picda.train import TrainConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT_DIR / "benchmark_ctrgcn"))
    ap.add_argument("--model", default="ctrgcn", choices=["ctrgcn", "stgcn"])
    ap.add_argument("--features", default="joint", choices=["joint", "joint_motion", "joint_bone"])
    ap.add_argument("--yaw", default="first_frame", choices=["first_frame", "none"])
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fewshot-ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--fewshot-inits", nargs="+",
                    default=["scratch", "source_only", "picda", "joint", "joint_fixmatch", "joint_proto"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = TrainConfig(model=a.model, features=a.features, canon_yaw=a.yaw, epochs=a.epochs,
                      seeds=tuple(a.seeds), num_workers=a.workers,
                      fewshot_ks=tuple(a.fewshot_ks), fewshot_inits=tuple(a.fewshot_inits))
    out = a.out
    if a.smoke:
        cfg.epochs, cfg.adapt_epochs, cfg.fewshot_steps, cfg.warmup_epochs = 1, 1, 5, 0
        cfg.seeds, cfg.log_every = (0,), 20
        out = str(OUTPUT_DIR / "smoke")
        a.folds = [0]
    Benchmark(cfg, out).run(a.methods, folds=a.folds, seeds=cfg.seeds)


if __name__ == "__main__":
    main()
