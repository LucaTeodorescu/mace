"""
For each model in a multirun directory, find the best epoch (latest checkpoint)
and print the test_correlations_A values at that epoch.

Usage:
    python scripts/best_corr_A.py multirun/2026-03-23/01-39-09/
"""

import argparse
import json
import os
import re
from pathlib import Path


def get_best_epoch(run_dir: str) -> int | None:
    """Get the best epoch from checkpoint filenames (highest epoch = last improvement)."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(r"best_model_epoch_(\d+)\.pt")
    epochs = []
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def main():
    parser = argparse.ArgumentParser(description="Extract test corr_A at the best epoch for each run.")
    parser.add_argument("multirun_dir", type=str, help="Path to the multirun directory")
    args = parser.parse_args()

    multirun_dir = Path(args.multirun_dir)
    run_dirs = sorted(
        [d for d in multirun_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    for run_dir in run_dirs:
        best_epoch = get_best_epoch(str(run_dir))
        if best_epoch is None:
            print(f"Run {run_dir.name}: no checkpoints found")
            continue

        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"Run {run_dir.name}: no metrics.json found")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        epochs = metrics["epoch"]
        # epochs are 0-indexed in metrics but checkpoints use epoch+1
        epoch_idx = None
        for i, e in enumerate(epochs):
            if e + 1 == best_epoch:
                epoch_idx = i
                break

        if epoch_idx is None:
            # Try matching directly (in case epochs are already 1-indexed)
            for i, e in enumerate(epochs):
                if e == best_epoch:
                    epoch_idx = i
                    break

        if epoch_idx is None:
            print(f"Run {run_dir.name}: best epoch {best_epoch} not found in metrics.json")
            continue

        test_corr_A = metrics.get("test_correlations_A", {})
        timesteps = sorted(test_corr_A.keys(), key=lambda k: int(k.split("_")[1]))

        print(f"Run {run_dir.name} | best epoch: {best_epoch}")
        for ts in timesteps:
            val = test_corr_A[ts][epoch_idx]
            print(f"  {ts}: {val:.6f}")
        print()


if __name__ == "__main__":
    main()
