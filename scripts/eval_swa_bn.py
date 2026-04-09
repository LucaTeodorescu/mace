"""
Evaluate SWA checkpoints that crashed during BN recomputation.

Usage:
    # BN-only run:
    uv run python scripts/eval_swa_bn.py \
        --model_file multirun/2026-03-23/11-14-08/0/checkpoints/MACE_rmd17_ethanol_run-3.model \
        --swa_checkpoint multirun/2026-03-23/11-14-08/0/checkpoints/MACE_rmd17_ethanol_run-3_epoch-2002_swa.pt \
        --train_file data/rmd17/xyz_split01/rmd17_ethanol_train.xyz \
        --test_file data/rmd17/xyz_split01/rmd17_ethanol_test.xyz

    # BN+dropout run:
    uv run python scripts/eval_swa_bn.py \
        --model_file multirun/2026-03-23/11-22-10/0/checkpoints/MACE_rmd17_ethanol_run-3.model \
        --swa_checkpoint multirun/2026-03-23/11-22-10/0/checkpoints/MACE_rmd17_ethanol_run-3_epoch-2231_swa.pt \
        --train_file data/rmd17/xyz_split01/rmd17_ethanol_train.xyz \
        --test_file data/rmd17/xyz_split01/rmd17_ethanol_test.xyz
"""

from __future__ import annotations

import argparse
import logging

import torch
from e3nn.nn import BatchNorm as E3nnBatchNorm

from mace import data
from mace.data import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, utils
from mace.tools.default_keys import DefaultKeys
from mace.tools.tables_utils import create_error_table

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

    # Evaluate
    model.eval()
    log.info("Evaluating SWA model on test set")
    table = create_error_table(
        table_type="TotalMAE",
        all_data_loaders={"Default_Default": test_loader},
        model=model,
        loss_fn=None,
        output_args={
            "energy": True,
            "forces": True,
            "virials": False,
            "stress": False,
            "dipoles": False,
        },
        log_wandb=False,
        device=device,
        distributed=False,
    )
    log.info(f"Error-table on TEST (SWA stage two):\n{table}")


if __name__ == "__main__":
    main()
