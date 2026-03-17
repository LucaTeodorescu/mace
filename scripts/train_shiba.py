import json
import logging
import os
import sys

import e3nn.nn as enn
import hydra
import hydra.utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from e3nn import o3

from mace import modules
from mace.modules.glass_models import MinimalMACE_glass


OmegaConf.register_new_resolver(
    "join", lambda lst, sep="-": sep.join(str(x) for x in lst)
)

disable_tqdm = not sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_cfg, device):
    """Build a MACE model from Hydra config."""
    hidden_irreps = o3.Irreps(model_cfg.hidden_irreps)
    interaction_cls = modules.interaction_classes[model_cfg.interaction_type]

    model = MinimalMACE_glass(
        r_max=model_cfg.r_max,
        num_bessel=model_cfg.num_bessel,
        num_polynomial_cutoff=model_cfg.num_polynomial_cutoff,
        max_ell=model_cfg.max_ell,
        interaction_cls=interaction_cls,
        interaction_cls_first=interaction_cls,
        num_interactions=model_cfg.num_interactions,
        num_elements=model_cfg.num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=model_cfg.avg_num_neighbors,
        correlation=model_cfg.correlation,
        gate=torch.nn.functional.silu,
        radial_MLP=list(model_cfg.radial_MLP),
        num_outputs=model_cfg.num_outputs,
        batchnorm=model_cfg.get("batchnorm", False),
        bn_momentum=model_cfg.get("bn_momentum", 0.5),
        dropout_p=model_cfg.get("dropout_p", 0.0),
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Data loading (Shiba glass)
# ---------------------------------------------------------------------------

def load_shiba_datasets(data_root, cfg_data, file_numbers, target_means=None, target_stds=None):
    """Load Shiba .pt files and optionally normalize per particle type."""
    from torch_geometric.data import Data

    all_data = []
    all_targets = []
    all_node_attrs = []

    for file_num in file_numbers:
        file_path = os.path.join(data_root, f"{cfg_data.pattern}_{file_num}_FIRE.pt")
        if not os.path.exists(file_path):
            logging.warning(f"{file_path} not found, skipping...")
            continue

        pt_data = torch.load(file_path)
        node_types = pt_data.x.squeeze()
        node_attrs = F.one_hot(node_types.long(), num_classes=2).float()

        data = Data(
            x=node_attrs,
            pos_th=pt_data.pos_th,
            edge_index_th=pt_data.edge_index_th,
            edge_attr_th=pt_data.edge_attr_th,
            y=pt_data.y,
        )
        all_data.append(data)
        all_targets.append(pt_data.y)
        all_node_attrs.append(node_attrs)

    if not all_data:
        raise RuntimeError(f"No data files found in {data_root}")

    # Compute or reuse normalization statistics
    if cfg_data.normalize:
        if target_means is not None and target_stds is not None:
            logging.info("Using pre-computed normalization statistics")
        else:
            all_targets_tensor = torch.cat(all_targets, dim=0)
            all_node_attrs_tensor = torch.cat(all_node_attrs, dim=0)
            particle_types = torch.argmax(all_node_attrs_tensor, dim=1)

            target_means = {}
            target_stds = {}
            for ptype in range(2):
                mask = particle_types == ptype
                if mask.sum() > 0:
                    type_targets = all_targets_tensor[mask]
                    target_means[ptype] = torch.mean(type_targets, dim=0)
                    target_stds[ptype] = torch.std(type_targets, dim=0).clamp(min=1e-8)
                    logging.info(
                        f"Type {ptype}: mean={target_means[ptype].tolist()}, "
                        f"std={target_stds[ptype].tolist()}, n={mask.sum()}"
                    )

        # Apply normalization
        for data in all_data:
            node_types = torch.argmax(data.x, dim=1)
            normalized_y = torch.zeros_like(data.y)
            for ptype in range(2):
                mask = node_types == ptype
                if mask.sum() > 0 and ptype in target_means:
                    normalized_y[mask] = (data.y[mask] - target_means[ptype]) / target_stds[ptype]
            data.y = normalized_y

    return all_data, target_means, target_stds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pearson_correlations(predictions, targets, node_attrs_list, time_indices):
    """Pearson correlation per particle type per timescale."""
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    node_attrs_concat = torch.cat(node_attrs_list, dim=0)
    particle_types = torch.argmax(node_attrs_concat, dim=1).cpu().numpy()

    correlations = {}
    for col_idx, t in enumerate(time_indices):
        for ptype in [0, 1]:
            mask = particle_types == ptype
            if np.sum(mask) > 0:
                pred_t = pred_np[mask, col_idx]
                tgt_t = target_np[mask, col_idx]
                if len(pred_t) > 1 and np.std(pred_t) > 0 and np.std(tgt_t) > 0:
                    corr, _ = pearsonr(pred_t, tgt_t)
                    correlations[f"t_{t}_type_{ptype}"] = corr
                else:
                    correlations[f"t_{t}_type_{ptype}"] = 0.0
    return correlations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_and_plot_metrics(metrics_history, epoch, save_dir, time_steps, experiment_name="train"):
    """Save training progress and correlation plots."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = metrics_history["epoch"]

    timestep = "t_6" if 6 in time_steps else f"t_{time_steps[0]}"

    # Plot 1: loss + correlation over epochs
    fig, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(epochs, metrics_history["train_correlations_A"][timestep], color="blue",
             linestyle="-", label=f"Train Corr ({timestep})", alpha=0.8, linewidth=2)
    ax1.plot(epochs, metrics_history["test_correlations_A"][timestep], color="blue",
             linestyle="--", label=f"Test Corr ({timestep})", alpha=0.8, linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel(f"Pearson Correlation - Particle A ({timestep})", color="blue", fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Training Progress - Epoch {epoch+1}", fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(epochs, metrics_history["train_loss"], color="red", linewidth=2,
             label="Train Loss", alpha=0.9)
    ax2.plot(epochs, metrics_history["test_loss"], color="darkred", linewidth=2,
             linestyle="--", label="Test Loss", alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Loss (MSE)", color="red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    plt.tight_layout()
    plot1_file = os.path.join(save_dir, f"training_progress_{experiment_name}.png")
    plt.savefig(plot1_file, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: correlation across time steps
    fig, ax = plt.subplots(figsize=(4, 3))
    train_corrs = [metrics_history["train_correlations_A"][f"t_{t}"][-1] for t in time_steps]
    test_corrs = [metrics_history["test_correlations_A"][f"t_{t}"][-1] for t in time_steps]
    ax.plot(time_steps, train_corrs, "o-", color="steelblue", linewidth=2, markersize=8, label="Train", alpha=0.8)
    ax.plot(time_steps, test_corrs, "s--", color="coral", linewidth=2, markersize=8, label="Test", alpha=0.8)
    ax.set_xlabel("Time Step Index", fontsize=12)
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title(f"Correlation over Time Steps - Particle A (Epoch {epoch+1})", fontsize=14)
    ax.set_xticks(time_steps)
    ax.set_xticklabels([f"t_{t}" for t in time_steps])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2_file = os.path.join(save_dir, f"correlation_timesteps_{experiment_name}.png")
    plt.savefig(plot2_file, dpi=200, bbox_inches="tight")
    plt.close()

    return plot1_file, plot2_file


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {device}")

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    time_indices = list(cfg.data.time_indices)
    data_root = os.path.join(hydra.utils.get_original_cwd(), cfg.data.root)

    # ---- Data ----
    train_file_numbers = list(range(cfg.data.train_start, cfg.data.train_end))
    test_file_numbers = list(range(cfg.data.test_start, cfg.data.test_end))

    logging.info("Loading training data...")
    train_data, y_mean, y_std = load_shiba_datasets(data_root, cfg.data, train_file_numbers)

    logging.info("Loading test data...")
    test_data, _, _ = load_shiba_datasets(
        data_root, cfg.data, test_file_numbers,
        target_means=y_mean, target_stds=y_std,
    )

    logging.info(f"Train: {len(train_data)} configs, Test: {len(test_data)} configs")

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = build_model(cfg.model, device)
    logging.info(f"Model: {cfg.model.interaction_type} | BN: {cfg.model.get('batchnorm', False)} | "
                 f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Hydra output directory ----
    exp_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    exp_name = cfg.experiment.name
    metrics_file = os.path.join(exp_dir, "metrics.json")

    config_file = os.path.join(exp_dir, "config.yaml")
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logging.info(f"Output directory: {exp_dir}")

    # ---- WandB ----
    if cfg.wandb.enabled:
        wandb.init(
            name=cfg.experiment.name if cfg.experiment.name != "default" else None,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.MSELoss()

    if cfg.training.scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg.training.scheduler_factor,
            patience=cfg.training.scheduler_patience,
            min_lr=cfg.training.min_lr,
        )
    else:
        scheduler = None

    # ---- BN debug hooks (only when wandb.debug is enabled) ----
    debug_mode = cfg.wandb.enabled and cfg.wandb.get("debug", False)
    bn_batch_stats = {}

    def _make_e3nn_bn_hook(layer_name):
        def hook(module, input, output):
            x = input[0]
            flat = x.reshape(-1, x.shape[-1])
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0)
            bn_batch_stats[f"debug/bn_batch_mean_abs/{layer_name}"] = batch_mean.abs().mean().item()
            bn_batch_stats[f"debug/bn_batch_var_min/{layer_name}"] = batch_var.min().item()
            bn_batch_stats[f"debug/bn_batch_var_mean/{layer_name}"] = batch_var.mean().item()
        return hook

    bn_hooks = []
    if debug_mode:
        for name, module in model.named_modules():
            if isinstance(module, enn.BatchNorm):
                bn_hooks.append(module.register_forward_hook(_make_e3nn_bn_hook(name)))

    # ---- Training loop ----
    best_test_loss = float("inf")
    nan_snapshot_saved = False
    global_step = 0
    debug_every_n_steps = 50

    metrics_history = {
        "epoch": [],
        "train_loss": [],
        "train_eval_loss": [],
        "test_loss": [],
        "learning_rate": [],
        "train_correlations_A": {f"t_{t}": [] for t in time_indices},
        "test_correlations_A": {f"t_{t}": [] for t in time_indices},
        "train_correlations_B": {f"t_{t}": [] for t in time_indices},
        "test_correlations_B": {f"t_{t}": [] for t in time_indices},
        "total_gradnorm": [],
    }

    for epoch in range(cfg.training.epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]", disable=disable_tqdm):
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            pred = outputs["propensities"]

            # NaN / explosion detection
            has_nan = torch.isnan(pred).any().item()
            pred_max = pred.abs().max().item()

            loss = criterion(pred, batch.y)
            loss_val = loss.item()

            if has_nan or loss_val > 1e6 or pred_max > 1e3:
                logging.warning(
                    f"[Epoch {epoch+1}] SPIKE - loss={loss_val:.2e}, pred_max={pred_max:.2e}, has_nan={has_nan}"
                )
                if debug_mode:
                    wandb.log({
                        "debug/spike_loss": loss_val,
                        "debug/spike_pred_max": pred_max,
                        "debug/spike_has_nan": int(has_nan),
                    })
                if has_nan and not nan_snapshot_saved:
                    nan_dir = os.path.join(exp_dir, "nan_snapshot")
                    os.makedirs(nan_dir, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "batch": batch.to("cpu"),
                        "pred": pred.detach().cpu(),
                        "epoch": epoch, "loss": loss_val,
                    }, os.path.join(nan_dir, "nan_snapshot.pt"))
                    nan_snapshot_saved = True
                if has_nan:
                    optimizer.zero_grad()
                    continue

            loss.backward()

            # Per-step debug logging (only in debug mode)
            if debug_mode and (global_step % debug_every_n_steps == 0):
                lr = optimizer.param_groups[0]["lr"]
                step_log = {}
                for pname, param in model.named_parameters():
                    if param.grad is not None:
                        grad_n = param.grad.norm().item()
                        ratio = lr * grad_n / (param.norm().item() + 1e-8)
                        step_log[f"debug/update_ratio/{pname}"] = ratio
                        step_log[f"debug/grad_norm/{pname}"] = grad_n
                step_log.update(bn_batch_stats)
                bn_batch_stats.clear()
                wandb.log(step_log)

            total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            global_step += 1

        train_loss /= len(train_data)

        # ---- Eval on train (no dropout, BN in eval mode) ----
        model.eval()
        train_eval_loss = 0.0
        train_predictions, train_targets, train_node_attrs = [], [], []
        n_train_eval_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                outputs = model(batch)
                pred = outputs["propensities"]
                loss = criterion(pred, batch.y)
                train_eval_loss += loss.item() * batch.num_graphs
                n_train_eval_samples += batch.num_graphs
                train_predictions.append(pred)
                train_targets.append(batch.y)
                train_node_attrs.append(batch.x)

        train_eval_loss /= n_train_eval_samples
        train_pred_cat = torch.cat(train_predictions, dim=0)
        train_targets_cat = torch.cat(train_targets, dim=0)
        train_correlations = compute_pearson_correlations(
            train_pred_cat, train_targets_cat, train_node_attrs, time_indices
        )

        # ---- Eval on test ----
        test_loss = 0.0
        test_predictions, test_targets, test_node_attrs = [], [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Test]", disable=disable_tqdm):
                batch = batch.to(device)
                outputs = model(batch)
                pred = outputs["propensities"]
                loss = criterion(pred, batch.y)
                test_loss += loss.item() * batch.num_graphs
                test_predictions.append(pred)
                test_targets.append(batch.y)
                test_node_attrs.append(batch.x)

        test_loss /= len(test_data)
        test_pred_cat = torch.cat(test_predictions, dim=0)
        test_targets_cat = torch.cat(test_targets, dim=0)
        test_correlations = compute_pearson_correlations(
            test_pred_cat, test_targets_cat, test_node_attrs, time_indices
        )

        # ---- Scheduler step ----
        if scheduler is not None:
            scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Logging ----
        timestep = "t_6" if 6 in time_indices else f"t_{time_indices[0]}"
        logging.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} - "
            f"Train: {train_eval_loss:.6f}, Test: {test_loss:.6f}, "
            f"Train Corr A {timestep}: {train_correlations.get(f'{timestep}_type_0', 0):.4f}, "
            f"Test Corr A {timestep}: {test_correlations.get(f'{timestep}_type_0', 0):.4f}, "
            f"LR: {current_lr:.2e}"
        )

        # ---- Metrics history ----
        metrics_history["epoch"].append(epoch)
        metrics_history["train_loss"].append(train_loss)
        metrics_history["train_eval_loss"].append(train_eval_loss)
        metrics_history["test_loss"].append(test_loss)
        metrics_history["learning_rate"].append(current_lr)
        metrics_history["total_gradnorm"].append(float(total_norm))
        for t in time_indices:
            metrics_history["train_correlations_A"][f"t_{t}"].append(
                train_correlations.get(f"t_{t}_type_0", 0.0))
            metrics_history["test_correlations_A"][f"t_{t}"].append(
                test_correlations.get(f"t_{t}_type_0", 0.0))
            metrics_history["train_correlations_B"][f"t_{t}"].append(
                train_correlations.get(f"t_{t}_type_1", 0.0))
            metrics_history["test_correlations_B"][f"t_{t}"].append(
                test_correlations.get(f"t_{t}_type_1", 0.0))

        # ---- Save metrics JSON ----
        with open(metrics_file, "w") as f:
            json_data = {
                "epoch": metrics_history["epoch"],
                "train_loss": [float(x) for x in metrics_history["train_loss"]],
                "test_loss": [float(x) for x in metrics_history["test_loss"]],
                "train_eval_loss": [float(x) for x in metrics_history["train_eval_loss"]],
                "learning_rate": [float(x) for x in metrics_history["learning_rate"]],
                "train_correlations_A": {k: [float(x) for x in v] for k, v in metrics_history["train_correlations_A"].items()},
                "test_correlations_A": {k: [float(x) for x in v] for k, v in metrics_history["test_correlations_A"].items()},
                "train_correlations_B": {k: [float(x) for x in v] for k, v in metrics_history["train_correlations_B"].items()},
                "test_correlations_B": {k: [float(x) for x in v] for k, v in metrics_history["test_correlations_B"].items()},
                "total_gradnorm": [float(x) for x in metrics_history["total_gradnorm"]],
            }
            json.dump(json_data, f, indent=2)

        # ---- WandB epoch logging ----
        if cfg.wandb.enabled:
            log_dict = {
                "train/loss": train_loss,
                "train/eval_loss": train_eval_loss,
                "test/loss": test_loss,
                "lr": current_lr,
                "grad_norm": float(total_norm),
            }
            for t in time_indices:
                log_dict[f"train/corr_A/t_{t}"] = train_correlations.get(f"t_{t}_type_0", 0.0)
                log_dict[f"test/corr_A/t_{t}"] = test_correlations.get(f"t_{t}_type_0", 0.0)
                log_dict[f"train/corr_B/t_{t}"] = train_correlations.get(f"t_{t}_type_1", 0.0)
                log_dict[f"test/corr_B/t_{t}"] = test_correlations.get(f"t_{t}_type_1", 0.0)
            # Log BN running statistics (only in debug mode)
            if debug_mode:
                for bname, module in model.named_modules():
                    if isinstance(module, enn.BatchNorm):
                        if hasattr(module, "running_var") and module.running_var is not None:
                            log_dict[f"debug/bn_running_var_min/{bname}"] = module.running_var.min().item()
                            log_dict[f"debug/bn_running_var_mean/{bname}"] = module.running_var.mean().item()
            wandb.log(log_dict)

        # ---- Checkpoint best model ----
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            ckpt_dir = os.path.join(exp_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_loss": test_loss,
                "test_correlations": test_correlations,
            }, ckpt_path)
            logging.info(f"> New best model saved! {ckpt_path}")

        # ---- Plots ----
        save_and_plot_metrics(
            metrics_history, epoch,
            save_dir=exp_dir,
            experiment_name=exp_name,
            time_steps=time_indices,
        )

    logging.info(f"Training complete. Best test loss: {best_test_loss:.6f}")

    for h in bn_hooks:
        h.remove()

    if cfg.wandb.enabled:
        wandb.finish()

    return model


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
