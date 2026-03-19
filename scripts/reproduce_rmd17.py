"""
Reproduce MACE rMD17 benchmark results.

Expects data already downloaded in data/rmd17/ with .npz files and rmd17.tar.bz2.
Split indices are extracted from the tar on first run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

# kcal/mol → eV conversion factor
KCAL_MOL_TO_EV = 0.04336411531

# rMD17 molecules with their atom counts (for energy_weight)
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


def extract_splits(data_dir: Path) -> None:
    """Extract split index CSVs from rmd17.tar.bz2 if not already present."""
    splits_dir = data_dir / "splits"
    if splits_dir.exists() and any(splits_dir.glob("index_train_*.csv")):
        print("  Split indices already extracted.")
        return

    tar_path = data_dir / "rmd17.tar.bz2"
    if not tar_path.exists():
        print(f"  ERROR: {tar_path} not found. Cannot extract split indices.")
        sys.exit(1)

    splits_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting split indices from {tar_path.name}...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if basename.endswith(".csv"):
                dest = splits_dir / basename
                if not dest.exists():
                    member.name = basename
                    tar.extract(member, splits_dir)
                    print(f"    Extracted: {basename}")


def load_split_indices(splits_dir: Path, split: int) -> tuple:
    """Load official train/test split indices.

    The rMD17 dataset provides 5 training splits (index_train_01..05.csv)
    and 1 test split (index_test_01.csv). Each contains 1000 indices.
    """
    train_idx_file = splits_dir / f"index_train_{split:02d}.csv"
    test_idx_file = splits_dir / f"index_test_{split:02d}.csv"

    if not train_idx_file.exists():
        candidates = list(splits_dir.rglob(f"index_train_{split:02d}.csv"))
        if candidates:
            train_idx_file = candidates[0]
        else:
            raise FileNotFoundError(
                f"Train split index not found: {train_idx_file}\n"
                f"Available files: {list(splits_dir.rglob('*.csv'))}"
            )

    if not test_idx_file.exists():
        candidates = list(splits_dir.rglob(f"index_test_{split:02d}.csv"))
        if candidates:
            test_idx_file = candidates[0]
        else:
            raise FileNotFoundError(
                f"Test split index not found: {test_idx_file}\n"
                f"Available files: {list(splits_dir.rglob('*.csv'))}"
            )

    train_indices = np.loadtxt(train_idx_file, dtype=int)
    test_indices = np.loadtxt(test_idx_file, dtype=int)

    return train_indices, test_indices


def npz_to_xyz(
    npz_path: Path,
    xyz_path: Path,
    indices: np.ndarray | None = None,
) -> None:
    """Convert rMD17 .npz to extxyz, applying kcal/mol → eV conversion."""
    import ase.io

    if xyz_path.exists():
        print(f"  Already exists: {xyz_path}")
        return

    data = np.load(npz_path)
    coords = data["coords"]
    nuclear_charges = data["nuclear_charges"]
    energies = data["energies"]
    forces = data["forces"]

    if indices is not None:
        coords = coords[indices]
        energies = energies[indices]
        forces = forces[indices]

    print(f"  Converting {len(coords)} structures → {xyz_path.name}")
    xyz_path.parent.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    for i in range(len(coords)):
        atoms = ase.Atoms(numbers=nuclear_charges, positions=coords[i])
        atoms.info["energy"] = KCAL_MOL_TO_EV * float(energies[i])
        atoms.info["config_type"] = "Default"
        atoms.cell = np.zeros((3, 3))
        atoms.pbc = False
        atoms.arrays["forces"] = KCAL_MOL_TO_EV * forces[i]
        atoms_list.append(atoms)

    ase.io.write(str(xyz_path), atoms_list, format="extxyz")


def prepare_data(data_dir: Path, split: int) -> None:
    """Extract splits and convert molecules from .npz to train/test .xyz files."""
    print(f"\n=== Preparing data (split {split:02d}) ===")

    extract_splits(data_dir)

    splits_dir = data_dir / "splits"
    train_indices, test_indices = load_split_indices(splits_dir, split)

    print(f"  Train indices: {len(train_indices)}, Test indices: {len(test_indices)}")

    xyz_dir = data_dir / f"xyz_split{split:02d}"

    for mol in MOLECULES:
        npz_path = data_dir / f"rmd17_{mol}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} not found, skipping {mol}")
            continue

        train_xyz = xyz_dir / f"rmd17_{mol}_train.xyz"
        test_xyz = xyz_dir / f"rmd17_{mol}_test.xyz"

        npz_to_xyz(npz_path, train_xyz, indices=train_indices)
        npz_to_xyz(npz_path, test_xyz, indices=test_indices)

    print(f"  Data ready in: {xyz_dir}")


def train_molecule(
    mol: str,
    data_dir: Path,
    split: int,
    device: str,
    work_dir: Path,
    seed: int,
    epochs: int,
    swa_start: int,
    batch_size: int,
) -> None:
    """Train MACE on a single rMD17 molecule using paper hyperparameters."""
    n_atoms = MOLECULES[mol]
    xyz_dir = data_dir / f"xyz_split{split:02d}"
    train_file = xyz_dir / f"rmd17_{mol}_train.xyz"
    test_file = xyz_dir / f"rmd17_{mol}_test.xyz"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")

    out_dir = work_dir / f"rmd17_{mol}_split{split:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters from the MACE paper and discussion #165
    # https://github.com/ACEsuit/mace/discussions/165
    cmd = [
        "mace_run_train",
        f"--name=MACE_rmd17_{mol}",
        f"--train_file={train_file}",
        "--valid_fraction=0.05",
        f"--test_file={test_file}",
        # Loss weights: energy_weight = num_atoms, forces_weight = 1000
        f"--energy_weight={n_atoms}.0",
        "--forces_weight=1000.0",
        '--config_type_weights={"Default":1.0}',
        "--E0s=average",
        # Model architecture
        "--model=ScaleShiftMACE",
        "--interaction_first=RealAgnosticResidualInteractionBlock",
        "--interaction=RealAgnosticResidualInteractionBlock",
        "--num_interactions=2",
        "--max_ell=3",
        "--hidden_irreps=256x0e + 256x1o + 256x2e",
        "--num_cutoff_basis=5",
        "--correlation=3",
        "--r_max=6.0",
        "--scaling=rms_forces_scaling",
        # Optimizer
        f"--batch_size={batch_size}",
        f"--max_num_epochs={epochs}",
        "--lr=0.01",
        "--patience=200",
        "--weight_decay=5e-7",
        "--ema",
        "--ema_decay=0.99",
        "--amsgrad",
        "--clip_grad=10",
        # SWA (Stochastic Weight Averaging — stage 2 training)
        "--swa",
        f"--start_swa={swa_start}",
        "--swa_lr=0.001",
        "--swa_forces_weight=1000.0",
        f"--swa_energy_weight={n_atoms}.0",
        # Output / misc
        "--error_table=TotalMAE",
        "--default_dtype=float32",
        f"--device={device}",
        f"--seed={seed}",
        f"--work_dir={out_dir}",
        "--restart_latest",
    ]

    print(f"\n{'='*60}")
    print(f"Training: {mol} ({n_atoms} atoms)")
    print(f"Output:   {out_dir}")
    print(f"Device:   {device}")
    print(f"{'='*60}")
    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce MACE rMD17 benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--molecule", "-m",
        type=str,
        default=None,
        choices=list(MOLECULES.keys()),
        help="Train a single molecule (default: all molecules)",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which official train/test split to use (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for training (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed (default: 3, matching the authors)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/rmd17"),
        help="Directory for data (default: data/rmd17)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/rmd17"),
        help="Directory for training outputs (default: results/rmd17)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3500,
        help="Max training epochs (default: 3500)",
    )
    parser.add_argument(
        "--swa-start",
        type=int,
        default=2000,
        help="Epoch to start SWA (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size (default: 5)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data (extract splits, convert to xyz), skip training",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip data preparation, assume data is ready",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    work_dir = args.work_dir.resolve()

    # Step 1: Prepare data (extract splits from tar, convert npz → xyz)
    if not args.train_only:
        prepare_data(data_dir, args.split)

    if args.prepare_only:
        print("\nData preparation complete. Use --train-only to skip this step next time.")
        return

    # Step 2: Train
    molecules = [args.molecule] if args.molecule else list(MOLECULES.keys())

    for mol in molecules:
        train_molecule(
            mol=mol,
            data_dir=data_dir,
            split=args.split,
            device=args.device,
            work_dir=work_dir,
            seed=args.seed,
            epochs=args.epochs,
            swa_start=args.swa_start,
            batch_size=args.batch_size,
        )

    print(f"\n{'='*60}")
    print("All done! Results are in:", work_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
