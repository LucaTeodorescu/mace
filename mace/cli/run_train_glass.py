import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from e3nn import o3
from pathlib import Path
import json

from mace.modules.glass_models import MinimalMACE
from mace.modules.blocks import (
    InteractionBlock,
)
from mace.tools import MetricsLogger, CheckpointHandler
from mace import data

def main():
    # Model params
    r_max = 5.0
    num_bessel = 8
    num_polynomial_cutoff = 5
    max_ell = 2
    num_interactions = 2
    hidden_irreps = o3.Irreps("32x0e + 16x1o")
    correlation = 3
    num_elements = 2 
    
    # Training params
    batch_size = 32
    lr = 1e-3
    max_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    train_file = "path/to/your/train_data.pt"
    valid_file = "path/to/your/valid_data.pt"
    checkpoint_dir = "./checkpoints"
    
    logging.basicConfig(level=logging.INFO)
    logger = MetricsLogger(directory="./results", tag="propensity_train")
    
    logging.info("Loading data...")
    
    train_data = load_shiba_dataset(train_file)
    valid_data = load_shiba_dataset(valid_file)
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    
    logging.info("Creating model...")
    
    model = MinimalMACE(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        interaction_cls=InteractionBlock,
        interaction_cls_first=InteractionBlock,
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
    loss_fn = torch.nn.MSELoss()
    
    checkpoint_handler = CheckpointHandler(
        directory=checkpoint_dir,
        tag="propensity",
        keep=5,  # Keep last 5 checkpoints
    )
    
    best_valid_loss = float('inf')
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
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
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                outputs = model(batch)
                predictions = outputs["propensities"]
                loss = loss_fn(predictions, batch.y)
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        # Logging
        logging.info(
            f"Epoch {epoch+1}/{max_epochs} - "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Valid Loss: {avg_valid_loss:.6f}"
        )
        
        logger.log(
            {"train_loss": avg_train_loss, "valid_loss": avg_valid_loss},
            step=epoch,
        )
        
        # Save checkpoint
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            checkpoint_handler.save(
                state={"model": model, "optimizer": optimizer},
                epochs=epoch,
                keep_last=True,
            )
            logging.info("Saved new best model")
    
    model_path = Path("./models") / "propensity_model.pt"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved final model to {model_path}")


def load_shiba_dataset(file_path):
    """Custom loader for .pt files"""
    pt_data = torch.load(file_path, weights_only=False)
    
    from torch_geometric.data import Data
    
    data_list = []
    
    data = Data(
        x=pt_data.x,
        pos=pt_data.pos_th,
        edge_index=pt_data.edge_index_th,
        edge_attr=pt_data.edge_attr_th,
        y=pt_data.y,
    )
    
    data_list.append(data)
    
    return data_list


if __name__ == "__main__":
    main()