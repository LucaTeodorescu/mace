import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from e3nn import o3
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mace import modules
from mace.modules.glass_models import MinimalMACE_glass
from mace.modules.blocks import (
    InteractionBlock,
)
from mace.tools import MetricsLogger, CheckpointHandler
from mace import data
from scipy.stats import pearsonr

def parse_args():
    parser = argparse.ArgumentParser(description='Train MACE model for glass propensity prediction')
    
    # Model parameters
    parser.add_argument('--r_max', type=float, default=2.0, help='Maximum distance for interactions')
    parser.add_argument('--num_bessel', type=int, default=10, help='Number of Bessel functions')
    parser.add_argument('--num_polynomial_cutoff', type=int, default=5, help='Number of polynomial cutoff functions')
    parser.add_argument('--max_ell', type=int, default=2, help='Maximum angular momentum')
    parser.add_argument('--num_interactions', type=int, default=2, help='Number of interaction blocks')
    parser.add_argument('--hidden_irreps', type=str, default="16x0e + 16x1o + 16x1e", help='Hidden irreps string')
    parser.add_argument('--correlation', type=int, default=3, help='Body order (nu)')
    parser.add_argument('--num_elements', type=int, default=2, help='Number of element types')
    parser.add_argument('--interaction_type', type=str, default="LucaInteractionBlock", help='Interaction block type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default="data/T_044/processed", help='Data directory')
    parser.add_argument('--train_start', type=int, default=1, help='Start file number for training')
    parser.add_argument('--train_end', type=int, default=401, help='End file number for training (exclusive)')
    parser.add_argument('--test_start', type=int, default=401, help='Start file number for testing')
    parser.add_argument('--test_end', type=int, default=501, help='End file number for testing (exclusive)')
    
    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="propensity_train", help='Experiment name')
    parser.add_argument('--results_dir', type=str, default="experiments", help='Results directory')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_type', type=str, default="None", help='LR scheduler type (ReduceLROnPlateau or None)')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='LR scheduler factor')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='LR scheduler patience')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Model params
    r_max = args.r_max
    num_bessel = args.num_bessel
    num_polynomial_cutoff = args.num_polynomial_cutoff
    max_ell = args.max_ell
    num_interactions = args.num_interactions
    hidden_irreps = o3.Irreps(args.hidden_irreps)
    correlation = args.correlation
    num_elements = args.num_elements
    num_workers = args.num_workers
    experiment_name = args.experiment_name
    
    # Training params
    batch_size = args.batch_size
    lr = args.lr
    max_epochs = args.max_epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Paths
    
    timestamp = datetime.now().strftime("%Y%m%d")
    folder_path = f"{args.results_dir}/{timestamp}_{experiment_name}"
    os.makedirs(folder_path, exist_ok=True)
    data_dir = args.data_dir
    metrics_file = f"{folder_path}/metrics_history_{timestamp}_{experiment_name}.json"
    
    # Split file numbers: 1-400 for train, 401-500 for test
    train_file_numbers = list(range(args.train_start, args.train_end))
    test_file_numbers = list(range(args.test_start, args.test_end))
    
    # Log all experiment parameters
    experiment_config = {
        "model_params": {
            "r_max": r_max,
            "num_bessel": num_bessel,
            "num_polynomial_cutoff": num_polynomial_cutoff,
            "max_ell": max_ell,
            "num_interactions": num_interactions,
            "hidden_irreps": str(hidden_irreps),
            "correlation": correlation,
            "num_elements": num_elements,
        },
        "training_params": {
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_epochs": max_epochs,
            "device": str(device),
            "num_workers": num_workers,
        },
        "scheduler_params": {
            "type": args.scheduler_type,
            "factor": args.scheduler_factor,
            "patience": args.scheduler_patience,
            "min_lr": args.min_lr,
        },
        "data_params": {
            "train_files": f"{args.train_start}-{args.train_end-1}",
            "test_files": f"{args.test_start}-{args.test_end-1}",
            "data_dir": data_dir,
        }
    }
    
    logging.basicConfig(level=logging.INFO)
    
    config_file = f"{folder_path}/experiment_config_{timestamp}_{experiment_name}.json"
    with open(config_file, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    logging.info(f"Experiment config saved to {config_file}")
    
    logging.info("Loading training data...")
    train_data, train_target_means, train_target_stds = load_multiple_shiba_datasets(
        data_dir, "isoconfig_N4096T0.44", train_file_numbers, normalize=True, experiment_name=experiment_name, folder_path=folder_path
    )
    
    logging.info("Loading validation data...")
    valid_data, _, _ = load_multiple_shiba_datasets(
        data_dir, "isoconfig_N4096T0.44", test_file_numbers, normalize=True, experiment_name=experiment_name, folder_path=folder_path,
        target_means=train_target_means, target_stds=train_target_stds
    )
    logging.info(f"Training on {len(train_data)} files, validating on {len(valid_data)} files")

    checkpoint_dir = f"{folder_path}/checkpoints_{timestamp}_{experiment_name}"
    
    logger = MetricsLogger(directory=folder_path, tag=f"propensity_train_{timestamp}_{experiment_name}")
    
    logging.info("Loading data...")
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        prefetch_factor=1,  # Instead of default 2
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        prefetch_factor=1,  # Instead of default 2
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    
    logging.info("Creating model...")
    
    model = MinimalMACE_glass(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        interaction_cls=modules.interaction_classes[args.interaction_type],
        interaction_cls_first=modules.interaction_classes[args.interaction_type],
        num_interactions=num_interactions,
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=50,
        correlation=correlation,
        gate=F.silu,
        num_outputs=10,
    ).to(device)
    
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = Adam(model.parameters(), lr=lr)
    if args.scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience)
    else:
        scheduler = None
        
    loss_fn = torch.nn.MSELoss()
    
    best_valid_loss = float('inf')
    
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'valid_loss': [],
        'learning_rate': [],  # Add learning rate tracking
        'train_correlations_A': {f't_{t}': [] for t in range(10)},
        'valid_correlations_A': {f't_{t}': [] for t in range(10)},
        'train_correlations_B': {f't_{t}': [] for t in range(10)},
        'valid_correlations_B': {f't_{t}': [] for t in range(10)},
        # 'train_correlations_A': {'t_6': []},  # Only t_6
        # 'valid_correlations_A': {'t_6': []},  # Only t_6
        # 'train_correlations_B': {'t_6': []},  # Only t_6
        # 'valid_correlations_B': {'t_6': []},  # Only t_6
    }
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        train_node_attrs = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            predictions = outputs["propensities"]
            
            # Loss computation
            loss = loss_fn(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.append(predictions)
            train_targets.append(batch.y)
            train_node_attrs.append(batch.x)
        
        avg_train_loss = train_loss / len(train_loader)
        
        train_pred_concat = torch.cat(train_predictions, dim=0)
        train_target_concat = torch.cat(train_targets, dim=0)
        train_correlations = compute_pearson_correlation(train_pred_concat, 
                                                         train_target_concat, 
                                                         train_node_attrs)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        valid_predictions = []
        valid_targets = []
        valid_node_attrs = []
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                outputs = model(batch)
                predictions = outputs["propensities"]
                loss = loss_fn(predictions, batch.y)
                # loss = loss_fn(predictions, batch.y[:, 6:7])
                valid_loss += loss.item()
                
                valid_predictions.append(predictions)
                valid_targets.append(batch.y)
                valid_node_attrs.append(batch.x)
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        if scheduler is not None:
            scheduler.step(avg_valid_loss)
            
            # Get current learning rate from scheduler
            current_lr = scheduler.optimizer.param_groups[0]['lr']
        else:
            current_lr = lr
        
        valid_pred_concat = torch.cat(valid_predictions, dim=0)
        valid_target_concat = torch.cat(valid_targets, dim=0)
        valid_correlations = compute_pearson_correlation(valid_pred_concat, 
                                                         valid_target_concat, 
                                                         valid_node_attrs)
        
        # Logging
        log_msg = (
            f"Epoch {epoch+1}/{max_epochs} - "
            # f"Train Loss: {avg_train_loss:.6f}, Train Corr of A at t_1 and t_10: "
            # f"{train_correlations['t_1_type_0']:.4f}, {train_correlations['t_9_type_0']:.4f}, "
            # f"Valid Loss: {avg_valid_loss:.6f}, Valid Corr of A at t_1 and t_10: "
            # f"{valid_correlations['t_1_type_0']:.4f}, {valid_correlations['t_9_type_0']:.4f}"
            f"Train Loss: {avg_train_loss:.6f}, Train Corr of A at t_6: "
            f"{train_correlations['t_6_type_0']:.4f}, "
            f"Valid Loss: {avg_valid_loss:.6f}, Valid Corr of A at t_6: "
            f"{valid_correlations['t_6_type_0']:.4f}, "
            f"LR: {current_lr:.2e}"
        )
        
        logging.info(log_msg)
        
        metrics = {
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "learning_rate": current_lr,
        }
        
        for key, val in train_correlations.items():
            metrics[f"train_corr_{key}"] = val
        for key, val in valid_correlations.items():
            metrics[f"valid_corr_{key}"] = val
        
        logger.log(metrics)

        # metrics history for plot and json storing
        metrics_history['valid_loss'].append(avg_valid_loss)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['epoch'].append(epoch)
        metrics_history['learning_rate'].append(current_lr)
        for t in range(10):
            metrics_history['train_correlations_A'][f't_{t}'].append(
                train_correlations.get(f't_{t}_type_0', 0.0)
            )
            metrics_history['valid_correlations_A'][f't_{t}'].append(
                valid_correlations.get(f't_{t}_type_0', 0.0)
            )
            metrics_history['train_correlations_B'][f't_{t}'].append(
                train_correlations.get(f't_{t}_type_1', 0.0)
            )
            metrics_history['valid_correlations_B'][f't_{t}'].append(
                valid_correlations.get(f't_{t}_type_1', 0.0)
            )
            
        # JSON file update
        with open(metrics_file, 'w') as f:
            json_data = {
                'epoch': metrics_history['epoch'],
                'train_loss': [float(x) for x in metrics_history['train_loss']],
                'valid_loss': [float(x) for x in metrics_history['valid_loss']],
                'learning_rate': [float(x) for x in metrics_history['learning_rate']],
                'train_correlations_A': {k: [float(x) for x in v] for k, v in metrics_history['train_correlations_A'].items()},
                'valid_correlations_A': {k: [float(x) for x in v] for k, v in metrics_history['valid_correlations_A'].items()},
                'train_correlations_B': {k: [float(x) for x in v] for k, v in metrics_history['train_correlations_B'].items()},
                'valid_correlations_B': {k: [float(x) for x in v] for k, v in metrics_history['valid_correlations_B'].items()},
            }
            json.dump(json_data, f, indent=2)
        
        # Create and save plot
        plot_file = save_and_plot_metrics(metrics_history, epoch, folder_path, experiment_name, timestamp)
        
        # Save checkpoint
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            
            # Create checkpoint directory if it doesn't exist
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
                'train_correlation_A_1': train_correlations['t_1_type_0'],
                'valid_correlation_A_1': valid_correlations['t_1_type_0'],
                'train_correlation_A_6': train_correlations['t_6_type_0'],
                'valid_correlation_A_6': valid_correlations['t_6_type_0'],
            }, checkpoint_path)

            logging.info(f"Saved new best model to {checkpoint_path}")

    logging.info(f"Files updated during run: {metrics_file} and {plot_file}")
    model_path = Path("./models") / f"propensity_model_{timestamp}_{experiment_name}.pt"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved final model to {model_path}")


def load_shiba_dataset(file_path):
    """Custom loader for .pt files"""
    pt_data = torch.load(file_path)
    
    from torch_geometric.data import Data
    import torch.nn.functional as F
    
    data_list = []
    
    # Convert integer node types to one-hot encoding for binary system
    node_types = pt_data.x.squeeze()  # Remove extra dimension: [4096, 1] -> [4096]
    node_attrs = F.one_hot(node_types.long(), num_classes=2).float()  # [4096, 2]
    
    data = Data(
        x=node_attrs,  # One-hot encoding
        pos_th=pt_data.pos_th,
        edge_index_th=pt_data.edge_index_th,
        edge_attr_th=pt_data.edge_attr_th,
        y=pt_data.y,
    )
    
    data_list.append(data)
    
    return data_list

def load_multiple_shiba_datasets(data_dir, pattern, file_numbers, normalize=True, stats_file=None, experiment_name="propensity_train", folder_path="results", target_means=None, target_stds=None):
    """Load multiple .pt files and combine them with per-particle-type normalization
    
    Args:
        data_dir: Directory containing the data files
        pattern: File pattern to match
        file_numbers: List of file numbers to load
        normalize: Whether to apply normalization
        stats_file: Path to save/load statistics file
        experiment_name: Name for the experiment
        folder_path: Path to save results
        target_means: Pre-computed target means (if None, will compute from data)
        target_stds: Pre-computed target stds (if None, will compute from data)
    """
    import glob
    from torch_geometric.data import Data
    import torch.nn.functional as F
    
    all_data = []
    all_targets = []
    all_node_attrs = []  # Store node attributes to identify particle types
    
    for file_num in file_numbers:
        file_path = os.path.join(data_dir, f"isoconfig_N4096T0.44_{file_num}_FIRE.pt")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        print(f"Loading {file_path}...")
        pt_data = torch.load(file_path)
        
        # Convert to proper format
        node_types = pt_data.x.squeeze()
        node_attrs = F.one_hot(node_types.long(), num_classes=2).float()
        
        data = Data(
            x=node_attrs,  # One-hot encoding
            pos_th=pt_data.pos_th,
            edge_index_th=pt_data.edge_index_th,
            edge_attr_th=pt_data.edge_attr_th,
            y=pt_data.y,  # Keep original targets for now
        )
        
        all_data.append(data)
        all_targets.append(pt_data.y)
        all_node_attrs.append(node_attrs)
    
    if normalize:
        # If pre-computed statistics are provided, use them
        if target_means is not None and target_stds is not None:
            print("Using pre-computed normalization statistics")
            for particle_type in range(2):
                if particle_type in target_means:
                    print(f"Particle type {particle_type} - Using pre-computed mean: {target_means[particle_type]}")
                    print(f"Particle type {particle_type} - Using pre-computed std: {target_stds[particle_type]}")
        else:
            # Compute statistics from current data (should only be done for training data)
            print("Computing normalization statistics from current data")
            # Concatenate all data
            all_targets_tensor = torch.cat(all_targets, dim=0)  # [total_nodes, 10]
            all_node_attrs_tensor = torch.cat(all_node_attrs, dim=0)  # [total_nodes, 2]
            
            # Get particle types (0 for type A, 1 for type B)
            particle_types = torch.argmax(all_node_attrs_tensor, dim=1)  # [total_nodes]
            
            # Compute per-particle-type statistics
            target_means = {}
            target_stds = {}
            
            for particle_type in range(2):  # 0 and 1 for binary system
                mask = (particle_types == particle_type)
                if mask.sum() > 0:  # If there are particles of this type
                    type_targets = all_targets_tensor[mask]  # [n_type_particles, 10]
                    
                    target_mean = torch.mean(type_targets, dim=0)  # Mean for each time step
                    target_std = torch.std(type_targets, dim=0)   # Std for each time step
                    
                    # Avoid division by zero
                    target_std = torch.where(target_std < 1e-8, torch.ones_like(target_std), target_std)
                    
                    target_means[particle_type] = target_mean
                    target_stds[particle_type] = target_std
                    
                    print(f"Particle type {particle_type} - Target mean: {target_mean}")
                    print(f"Particle type {particle_type} - Target std: {target_std}")
                    print(f"Particle type {particle_type} - Number of particles: {mask.sum()}")
            
            # Save statistics for later use
            if stats_file is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                stats_file = f"{folder_path}/target_statistics_per_particle_{timestamp}_{experiment_name}.json"
            
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            stats = {
                "target_means": {str(k): v.tolist() for k, v in target_means.items()},
                "target_stds": {str(k): v.tolist() for k, v in target_stds.items()},
                "num_samples": len(all_data),
                "particle_type_counts": {str(k): int((particle_types == k).sum()) for k in range(2)}
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved per-particle normalization statistics to {stats_file}")
        
        # Apply per-particle-type normalization to all data
        for data in all_data:
            node_types = torch.argmax(data.x, dim=1)  # Get particle types for this data
            normalized_y = torch.zeros_like(data.y)
            
            for particle_type in range(2):
                mask = (node_types == particle_type)
                if mask.sum() > 0 and particle_type in target_means:
                    # Normalize targets for this particle type
                    normalized_y[mask] = (data.y[mask] - target_means[particle_type]) / target_stds[particle_type]
            
            data.y = normalized_y
    
    return all_data, target_means if normalize else None, target_stds if normalize else None

def compute_pearson_correlation(predictions, targets, node_attrs_list):
    """Calculate Pearson correlation coefficient between predictions and targets"""
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    correlations = {}
    node_attrs_concat = torch.cat(node_attrs_list, dim=0) # [nodes * samples, 2]
    particle_types = torch.argmax(node_attrs_concat, dim=1).cpu().numpy() # [nodes * samples] 0s or 1s
    
    for t in range(10):
        for particle_type in [0, 1]:
            mask = particle_types == particle_type
            if np.sum(mask) > 0:
                pred_t_type = pred_np[mask, t]
                target_t_type = target_np[mask, t]
                
                if len(pred_t_type) > 1 and np.std(pred_t_type) > 0 and np.std(target_t_type) > 0:
                    corr, _ = pearsonr(pred_t_type, target_t_type)
                    correlations[f't_{t}_type_{particle_type}'] = corr
                else:
                    correlations[f't_{t}_type_{particle_type}'] = 0.0
    
    return correlations
    
    
def save_and_plot_metrics(metrics_history, epoch, save_dir="./results", experiment_name="propensity_train", timestamp=None):
    """Save metrics and create plot with correlations and loss"""
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = metrics_history['epoch']
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot correlations on primary y-axis
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Different colors for each timestep
    
    # for t, color in enumerate(colors):
    #     valid_corr = metrics_history['valid_correlations_A'][f't_{t}']
    #     ax1.plot(epochs, valid_corr, color=color, linestyle='--', 
    #             label=f'Valid t_{t}', alpha=0.8, linewidth=1.5)
    # Plot correlations on primary y-axis - only t_6
    valid_corr = metrics_history['valid_correlations_A']['t_6']
    ax1.plot(epochs, valid_corr, color='blue', linestyle='--', 
            label='Valid t_6', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Pearson Correlation - Particle A', color='black')
    ax1.set_ylabel('Pearson Correlation - Particle A (t_6)', color='black')
    ax1.set_ylim(0, 0.8)  # Set y-axis range from 0 to 0.8
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    # ax1.set_title(f'Training Progress - Epoch {epoch+1}')
    ax1.set_title(f'Training Progress - Epoch {epoch+1} (t_6 only)')
    
    # Create second y-axis for loss
    ax2 = ax1.twinx()
    
    ax2.plot(epochs, metrics_history['train_loss'], color='red', linewidth=2, 
            label='Train Loss', alpha=0.9)
    ax2.plot(epochs, metrics_history['valid_loss'], color='darkred', linewidth=2, 
            linestyle='--', label='Valid Loss', alpha=0.9)
    
    ax2.set_yscale('log')
    
    ax2.set_ylabel('Loss (MSE)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Put correlation legend outside the plot
    ax1.legend(lines1, labels1, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)
    ax2.legend(lines2, labels2, bbox_to_anchor=(1.15, 0.2), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_dir, f"training_progress_{timestamp}_{experiment_name}.png")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return plot_file


if __name__ == "__main__":
    print("Current Directory")
    print(os.getcwd())
    

    main()