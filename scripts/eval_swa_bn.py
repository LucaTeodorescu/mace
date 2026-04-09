"""
Evaluate SWA checkpoints that crashed during BN recomputation.

Usage:
    # BN-only run:
    uv run python scripts/eval_swa_bn.py \
        --model_file multirun/2026-03-23/11-14-08/0/checkpoints/MACE_rmd17_ethanol_run-3.model \
        --swa_checkpoint multirun/2026-03-23/11-14-08/0/checkpoints/MACE_rmd17_ethanol_run-3_epoch-2002_swa.pt \
        --train_file data/rmd17/xyz_split01/rmd17_ethanol_train.xyz \
        --test_file data/rmd17/xyz_split01/rmd17_ethanol_test.xyz \
        --results_dir multirun/2026-03-23/11-14-08/0/results/MACE_rmd17_ethanol_run-3_train.txt

    # BN+dropout run:
    uv run python scripts/eval_swa_bn.py \
        --model_file multirun/2026-03-23/11-22-10/0/checkpoints/MACE_rmd17_ethanol_run-3.model \
        --swa_checkpoint multirun/2026-03-23/11-22-10/0/checkpoints/MACE_rmd17_ethanol_run-3_epoch-2231_swa.pt \
        --train_file data/rmd17/xyz_split01/rmd17_ethanol_train.xyz \
        --test_file data/rmd17/xyz_split01/rmd17_ethanol_test.xyz \
        --results_dir multirun/2026-03-23/11-22-10/0/results/MACE_rmd17_ethanol_run-3_train.txt
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path

import torch
from e3nn.nn import BatchNorm as E3nnBatchNorm

from mace import data
from mace.data import KeySpecification
from mace.tools import torch_geometric, utils
from mace.tools.tables_utils import create_error_table
from mace.modules.loss import WeightedEnergyForcesLoss

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@torch.no_grad()
def update_bn(loader, model, device):
    """Recompute BatchNorm running statistics (fixed: no force computation)."""
    e3nn_bns = [m for m in model.modules() if isinstance(m, E3nnBatchNorm)]
    saved_momenta = {m: m.momentum for m in e3nn_bns}
    for m in e3nn_bns:
        m.running_mean.zero_()
        m.running_var.fill_(1.0)

    pytorch_bns = [
        m for m in model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    saved_pytorch_momenta = {m: m.momentum for m in pytorch_bns}
    for m in pytorch_bns:
        m.running_mean.zero_()
        m.running_var.fill_(1.0)
        m.num_batches_tracked.zero_()

    if not e3nn_bns and not pytorch_bns:
        log.info("No BatchNorm layers found")
        return

    log.info(f"Found {len(e3nn_bns)} e3nn BN and {len(pytorch_bns)} PyTorch BN layers")

    was_training = model.training
    model.train()

    for i, batch in enumerate(loader):
        cma_momentum = 1.0 / (i + 1)
        for m in e3nn_bns:
            m.momentum = cma_momentum
        for m in pytorch_bns:
            m.momentum = cma_momentum

        batch = batch.to(device)
        batch_dict = batch.to_dict()
        model(batch_dict, training=True, compute_force=False)

    for m, mom in saved_momenta.items():
        m.momentum = mom
    for m, mom in saved_pytorch_momenta.items():
        m.momentum = mom

    model.train(was_training)
    log.info(f"BN statistics recomputed over {i + 1} batches")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWA checkpoint with BN fix")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to .model file (full model, stage one)")
    parser.add_argument("--swa_checkpoint", type=str, required=True,
                        help="Path to SWA .pt checkpoint")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training xyz file (for BN stat recomputation)")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Test xyz file for evaluation")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to _train.txt results file (for plotting)")
    parser.add_argument("--energy_weight", type=float, default=9.0,
                        help="Energy weight for loss (default: 9 = ethanol num_atoms)")
    parser.add_argument("--forces_weight", type=float, default=1000.0)
    parser.add_argument("--swa_start", type=int, default=2000,
                        help="SWA start epoch (for plot vertical line)")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)

    # Load full model (stage one) - contains the complete architecture
    log.info(f"Loading model from {args.model_file}")
    model = torch.load(args.model_file, map_location=device)

    # Load SWA state dict over it
    log.info(f"Loading SWA checkpoint from {args.swa_checkpoint}")
    swa_ckpt = torch.load(args.swa_checkpoint, map_location=device)
    swa_epoch = int(Path(args.swa_checkpoint).stem.split("epoch-")[1].split("_")[0])
    model.load_state_dict(swa_ckpt["model"])
    model.to(device)

    # Get z_table from model
    atomic_numbers = model.atomic_numbers.tolist()
    z_table = utils.AtomicNumberTable(zs=atomic_numbers)
    r_max = float(model.r_max)

    # Build key specification (rMD17 uses REF_energy / REF_forces)
    key_spec = KeySpecification.from_defaults()

    # Load train data for BN update
    log.info("Loading training data for BN recomputation")
    _, train_configs = data.load_from_xyz(
        file_path=args.train_file,
        key_specification=key_spec,
        config_type_weights={"Default": 1.0},
    )
    train_set = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max)
        for c in train_configs
    ]
    train_loader = torch_geometric.dataloader.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
    )

    # Recompute BN statistics
    update_bn(train_loader, model, device)

    # Load test data
    log.info("Loading test data")
    _, test_configs = data.load_from_xyz(
        file_path=args.test_file,
        key_specification=key_spec,
        config_type_weights={"Default": 1.0},
    )
    test_set = [
        data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max)
        for c in test_configs
    ]
    test_loader = torch_geometric.dataloader.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
    )

    # Build data loader dicts (matching MACE's naming convention)
    train_valid_data_loader = {"train_Default": train_loader}
    test_data_loader = {"Default_Default": test_loader}

    loss_fn = WeightedEnergyForcesLoss(
        energy_weight=args.energy_weight, forces_weight=args.forces_weight,
    )
    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": False,
        "dipoles": False,
    }

    # Evaluate
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    log.info("Evaluating SWA model on train set")
    table_train = create_error_table(
        table_type="TotalMAE",
        all_data_loaders=train_valid_data_loader,
        model=model,
        loss_fn=loss_fn,
        output_args=output_args,
        log_wandb=False,
        device=device,
        distributed=False,
    )
    log.info(f"Error-table on TRAIN (SWA stage two):\n{table_train}")

    log.info("Evaluating SWA model on test set")
    table_test = create_error_table(
        table_type="TotalMAE",
        all_data_loaders=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        output_args=output_args,
        log_wandb=False,
        device=device,
        distributed=False,
    )
    log.info(f"Error-table on TEST (SWA stage two):\n{table_test}")

    # Save stage-two model next to the original
    model_dir = Path(args.model_file).parent
    stem = Path(args.model_file).stem  # e.g. "MACE_rmd17_ethanol_run-3"
    stagetwo_path = model_dir / f"{stem}_stagetwo.model"
    log.info(f"Saving SWA model to {stagetwo_path}")
    torch.save(deepcopy(model).cpu(), stagetwo_path)

    # Plot if results_dir provided
    if args.results_dir:
        try:
            from mace.cli.visualise_train import TrainingPlotter

            plotter = TrainingPlotter(
                results_dir=args.results_dir,
                heads=["Default"],
                table_type="TotalMAE",
                train_valid_data=train_valid_data_loader,
                test_data=test_data_loader,
                output_args=output_args,
                device=device,
                plot_frequency=1,
                distributed=False,
                swa_start=args.swa_start,
            )
            plotter.plot(swa_epoch, model, rank=0)
            log.info("Plot saved")
        except Exception as e:
            log.warning(f"Plotting failed: {e}")

    log.info("Done")


if __name__ == "__main__":
    main()
