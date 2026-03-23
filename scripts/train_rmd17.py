"""
Hydra-based rMD17 training script.

Prepares data (extract splits, npz→xyz) then launches mace_run_train.

Usage:
    # Single molecule
    python scripts/train_rmd17.py data.molecule=ethanol

    # Sweep on SLURM
    python scripts/train_rmd17.py -m hydra/launcher=submitit_slurm \
        data.molecule=ethanol,aspirin,toluene data.split=1,2,3
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# kcal/mol → eV
KCAL_MOL_TO_EV = 0.04336411531

# rMD17 molecules → atom counts (for energy_weight = num_atoms)
MOLECULES = {
    "aspirin": 21,
    "azobenzene": 24,
    "benzene": 12,
    "ethanol": 9,
    "malonaldehyde": 9,
    "naphthalene": 18,
    "paracetamol": 20,
    "salicylic_acid": 16,
    "toluene": 15,
    "uracil": 12,
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def extract_splits(data_dir: Path) -> None:
    """Extract split index CSVs from rmd17.tar.bz2 if not already present."""
    splits_dir = data_dir / "splits"
    if splits_dir.exists() and any(splits_dir.glob("index_train_*.csv")):
        return

    tar_path = data_dir / "rmd17.tar.bz2"
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Split indices not found and {tar_path} missing. "
            f"Cannot extract splits."
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting split indices from %s", tar_path.name)
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if basename.endswith(".csv"):
                dest = splits_dir / basename
                if not dest.exists():
                    member.name = basename
                    tar.extract(member, splits_dir)
                    log.info("  Extracted: %s", basename)


def load_split_indices(splits_dir: Path, split: int):
    train_file = splits_dir / f"index_train_{split:02d}.csv"
    test_file = splits_dir / f"index_test_{split:02d}.csv"

    if not train_file.exists():
        candidates = list(splits_dir.rglob(f"index_train_{split:02d}.csv"))
        if not candidates:
            raise FileNotFoundError(f"Train split not found: {train_file}")
        train_file = candidates[0]

    if not test_file.exists():
        candidates = list(splits_dir.rglob(f"index_test_{split:02d}.csv"))
        if not candidates:
            raise FileNotFoundError(f"Test split not found: {test_file}")
        test_file = candidates[0]

    return np.loadtxt(train_file, dtype=int), np.loadtxt(test_file, dtype=int)


def npz_to_xyz(npz_path: Path, xyz_path: Path, indices: np.ndarray) -> None:
    """Convert rMD17 .npz to extxyz with kcal/mol → eV conversion."""
    import ase
    import ase.io

    if xyz_path.exists():
        return

    data = np.load(npz_path)
    coords = data["coords"][indices]
    nuclear_charges = data["nuclear_charges"]
    energies = data["energies"][indices]
    forces = data["forces"][indices]

    log.info("Converting %d structures → %s", len(coords), xyz_path.name)
    xyz_path.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    for i in range(len(coords)):
        atoms = ase.Atoms(numbers=nuclear_charges, positions=coords[i])
        atoms.info["REF_energy"] = KCAL_MOL_TO_EV * float(energies[i])
        atoms.info["config_type"] = "Default"
        atoms.cell = np.zeros((3, 3))
        atoms.pbc = False
        atoms.arrays["REF_forces"] = KCAL_MOL_TO_EV * forces[i]
        atoms_list.append(atoms)

    ase.io.write(str(xyz_path), atoms_list, format="extxyz")


def prepare_molecule(data_dir: Path, molecule: str, split: int) -> tuple[Path, Path]:
    """Prepare train/test xyz files for a single molecule. Returns (train_path, test_path)."""
    extract_splits(data_dir)
    train_idx, test_idx = load_split_indices(data_dir / "splits", split)

    npz_path = data_dir / f"rmd17_{molecule}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Data not found: {npz_path}")

    xyz_dir = data_dir / f"xyz_split{split:02d}"
    train_xyz = xyz_dir / f"rmd17_{molecule}_train.xyz"
    test_xyz = xyz_dir / f"rmd17_{molecule}_test.xyz"

    npz_to_xyz(npz_path, train_xyz, train_idx)
    npz_to_xyz(npz_path, test_xyz, test_idx)

    return train_xyz, test_xyz


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_mace_command(cfg: DictConfig, train_file: Path, test_file: Path) -> list[str]:
    """Build mace_run_train CLI command from Hydra config."""
    mol = cfg.data.molecule
    n_atoms = MOLECULES[mol]
    model = cfg.model
    training = cfg.training

    energy_weight = n_atoms
    from hydra.core.hydra_config import HydraConfig
    out_dir = HydraConfig.get().runtime.output_dir

    cmd = [
        "mace_run_train",
        f"--name=MACE_rmd17_{mol}",
        f"--train_file={train_file}",
        f"--valid_fraction=0.05",
        f"--test_file={test_file}",
        # Loss
        f"--energy_weight={energy_weight}.0",
        f"--forces_weight={training.forces_weight}",
        '--config_type_weights={"Default":1.0}',
        f"--E0s={model.E0s}",
        # Architecture
        f"--model={model.model_class}",
        f"--interaction_first={model.interaction_type}",
        f"--interaction={model.interaction_type}",
        f"--num_interactions={model.num_interactions}",
        f"--max_ell={model.max_ell}",
        f"--hidden_irreps={str(model.hidden_irreps).replace(' ', '')}",
        f"--num_cutoff_basis={model.num_cutoff_basis}",
        f"--correlation={model.correlation}",
        f"--r_max={model.r_max}",
        f"--scaling={model.scaling}",
        # Optimizer
        f"--batch_size={training.batch_size}",
        f"--max_num_epochs={training.epochs}",
        f"--lr={training.lr}",
        f"--patience={training.patience}",
        f"--weight_decay={training.weight_decay}",
        f"--clip_grad={training.grad_clip}",
    ]

    # EMA
    if training.get("ema", False):
        cmd.append("--ema")
        cmd.append(f"--ema_decay={training.ema_decay}")
    if training.get("amsgrad", False):
        cmd.append("--amsgrad")

    # SWA
    if training.get("swa", False):
        cmd.extend([
            "--swa",
            f"--start_swa={training.swa_start}",
            f"--swa_lr={training.swa_lr}",
            f"--swa_forces_weight={training.swa_forces_weight}",
            f"--swa_energy_weight={energy_weight}.0",
        ])

    # BatchNorm
    if model.get("batchnorm", False):
        cmd.append("--batchnorm")
        cmd.append(f"--bn_momentum={model.get('bn_momentum', 0.5)}")

    # Dropout
    dropout_p = model.get("dropout_p", 0.0)
    if dropout_p > 0:
        cmd.append(f"--dropout_p={dropout_p}")

    # WandB
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        cmd.append("--wandb")
        if wandb_cfg.get("entity"):
            cmd.append(f"--wandb_entity={wandb_cfg.entity}")
        if wandb_cfg.get("project"):
            cmd.append(f"--wandb_project={wandb_cfg.project}")
        if wandb_cfg.get("group"):
            cmd.append(f"--wandb_group={wandb_cfg.group}")
        cmd.append(f"--wandb_name={cfg.experiment.name}")

    # Output
    cmd.extend([
        "--error_table=TotalMAE",
        "--default_dtype=float32",
        f"--device={training.device}",
        f"--seed={training.seed}",
        f"--work_dir={out_dir}",
    ])

    return cmd


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="rmd17", version_base=None)
def main(cfg: DictConfig):
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    mol = cfg.data.molecule
    if mol not in MOLECULES:
        raise ValueError(f"Unknown molecule: {mol}. Choose from: {list(MOLECULES.keys())}")

    data_dir = Path(cfg.data.root).resolve()
    split = cfg.data.split

    # Prepare data
    log.info("Preparing data: %s split %d", mol, split)
    train_file, test_file = prepare_molecule(data_dir, mol, split)

    # Build and run training command
    cmd = build_mace_command(cfg, train_file, test_file)
    log.info("Command:\n  %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        log.info("mace_run_train stdout:\n%s", result.stdout)
    if result.stderr:
        log.error("mace_run_train stderr:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"mace_run_train failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:] if result.stderr else '(empty)'}"
        )


if __name__ == "__main__":
    main()
