#!/usr/bin/env python3
"""
Test script to verify that LucaInteractionBlock can be imported and instantiated correctly.
"""

import torch
from e3nn import o3
from mace.modules import LucaInteractionBlock

def test_luca_interaction_block():
    """Test that LucaInteractionBlock can be instantiated with proper parameters."""
    
    # Define test parameters
    node_attrs_irreps = o3.Irreps([(2, (0, 1))])  # 2 scalars for binary system
    node_feats_irreps = o3.Irreps([(16, (0, 1))])  # 16 scalars
    edge_attrs_irreps = o3.Irreps.spherical_harmonics(2)  # up to l=2
    edge_feats_irreps = o3.Irreps("10x0e")  # 10 scalars for radial features
    target_irreps = o3.Irreps("16x0e + 16x1o + 16x1e")  # mixed irreps
    hidden_irreps = o3.Irreps("16x0e + 16x1o + 16x1e")
    avg_num_neighbors = 50.0
    radial_MLP = [32, 32]
    
    try:
        # Create the interaction block
        interaction_block = LucaInteractionBlock(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=target_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        
        print("LucaInteractionBlock created successfully!")
        print(f"   - Has batch_norm_up: {hasattr(interaction_block, 'batch_norm_up')}")
        print(f"   - Has batch_norm_out: {hasattr(interaction_block, 'batch_norm_out')}")
        print(f"   - Has batch_norm_skip: {hasattr(interaction_block, 'batch_norm_skip')}")
        
        # Test forward pass with dummy data
        batch_size = 4
        num_nodes = 100
        
        # Create dummy data
        node_attrs = torch.randn(num_nodes, 2)  # One-hot encoded node types
        node_feats = torch.randn(num_nodes, node_feats_irreps.dim)
        edge_attrs = torch.randn(200, edge_attrs_irreps.dim)  # Random edge attributes
        edge_feats = torch.randn(200, edge_feats_irreps.dim)  # Random edge features
        edge_index = torch.randint(0, num_nodes, (2, 200))  # Random edge connections
        
        # Forward pass
        with torch.no_grad():
            output, skip_connection = interaction_block(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
        
        print("Forward pass successful!")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Skip connection shape: {skip_connection.shape}")
        print(f"   - Expected output shape: ({num_nodes}, {hidden_irreps.dim})")
        print(f"   - Expected skip shape: ({num_nodes}, {hidden_irreps.dim})")
        
        return True
        
    except Exception as e:
        print(f"Error creating or testing LucaInteractionBlock: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing LucaInteractionBlock...")
    success = test_luca_interaction_block()
    if success:
        print("\nAll tests passed! LucaInteractionBlock is ready to use.")
    else:
        print("\nTests failed. Please check the implementation.")
