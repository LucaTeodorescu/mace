###########################################################################################
# Implementation of MACE models for Glasses
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import nn as e3nn_nn
from e3nn import o3
from e3nn.util.jit import compile_mode

# from mace.modules.embeddings import GenericJointEmbedding
# from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    NormLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_fixed_charge_dipole,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)


@compile_mode("script")
class MinimalMACE_glass(torch.nn.Module):
    """
    Minimal MACE model for glass propensity prediction.
    
    This model is designed for predicting time-dependent propensity values
    for glass systems with multiple particle types.
    
    Args:
        r_max: Maximum interaction radius
        num_bessel: Number of Bessel basis functions
        num_polynomial_cutoff: Number of polynomial cutoff functions
        max_ell: Maximum spherical harmonic degree
        interaction_cls: Interaction block class for subsequent layers
        interaction_cls_first: Interaction block class for first layer
        num_interactions: Number of interaction blocks
        num_elements: Number of particle types (default: 2 for binary system)
        hidden_irreps: Hidden irreps for the model
        MLP_irreps: MLP irreps (unused in this minimal version)
        avg_num_neighbors: Average number of neighbors for normalization
        correlation: Correlation order (body order)
        gate: Activation function
        radial_MLP: Radial MLP architecture
        num_outputs: Number of time steps to predict (default: 10)
    """
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int = 2,  # Default for binary system
        hidden_irreps: o3.Irreps = None,
        MLP_irreps: o3.Irreps = None,  # Unused in minimal version
        avg_num_neighbors: float = 50.0,
        correlation: Union[int, List[int]] = 3,
        gate: Optional[Callable] = None,
        radial_MLP: Optional[List[int]] = None,
        num_outputs: int = 10,  # 10 time steps
        batchnorm: bool = False,
        bn_momentum: float = 0.5,
        dropout_p: float = 0.0,
        readout_type: str = "linear",  # "linear" (o3.Linear) or "norm" (eqnet-style norm + nn.Linear)
    ):
        super().__init__()
        
        # Validate inputs
        if num_elements <= 0:
            raise ValueError(f"num_elements must be positive, got {num_elements}")
        if num_outputs <= 0:
            raise ValueError(f"num_outputs must be positive, got {num_outputs}")
        if r_max <= 0:
            raise ValueError(f"r_max must be positive, got {r_max}")

        # Basic setup
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Embeddings
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        
        self.num_elements = num_elements
        self.num_outputs = num_outputs
        self.r_max = r_max

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        # Spherical harmonics for angular features
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interaction blocks
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()

        if radial_MLP is None:
            radial_MLP = [32, 32,]

        # Build interactions and products
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()

        # First interaction
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            dropout_p=dropout_p,
        )
        self.interactions.append(inter)

        prod = EquivariantProductBasisBlock(
            node_feats_irreps=interaction_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=False,  # No self-connection for first layer
        )
        self.products.append(prod)

        # Rest of interactions
        for i in range(1, num_interactions):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                dropout_p=dropout_p,
            )
            self.interactions.append(inter)

            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps,
                correlation=correlation[i],
                num_elements=num_elements,
                use_sc=True,  # Use self-connection after first layer
            )
            self.products.append(prod)

        # Optional BatchNorm before each interaction block
        self.use_batchnorm = batchnorm
        self.layer_norms = torch.nn.ModuleList()
        if batchnorm:
            # First interaction takes node_feats_irreps (scalars only)
            self.layer_norms.append(
                e3nn_nn.BatchNorm(node_feats_irreps, momentum=bn_momentum)
            )
            # Subsequent interactions take hidden_irreps (after product output)
            for _ in range(1, num_interactions):
                self.layer_norms.append(
                    e3nn_nn.BatchNorm(hidden_irreps, momentum=bn_momentum)
                )

        # Readout
        self.readout_type = readout_type
        self.propensity_readouts = torch.nn.ModuleList()

        for idtype in range(num_elements):
            if readout_type == "norm":
                self.propensity_readouts.append(NormLinearReadoutBlock(
                    hidden_irreps,
                    o3.Irreps(f"{num_outputs}x0e"),
                ))
            else:
                self.propensity_readouts.append(LinearReadoutBlock(
                    hidden_irreps,
                    o3.Irreps(f"{num_outputs}x0e"),
                ))
            
        
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            data: Dictionary containing:
                - pos_th: Atomic positions [n_atoms, 3]
                - x: Node attributes (one-hot encoded) [n_atoms, num_elements]
                - edge_index_th: Edge indices [2, n_edges]
                - batch: Batch indices [n_atoms] (optional)
                
        Returns:
            Dictionary containing:
                - propensities: Predicted propensity values [n_atoms, num_outputs]
                - node_feats: Final node features [n_atoms, hidden_dim]
        """
        positions = data["pos_th"]
        node_attrs = data["x"]
        edge_index = data["edge_index_th"]
        atom_types = torch.argmax(node_attrs, dim=1)
        batch = data.get("batch", torch.zeros(positions.shape[0], dtype=torch.long))

        # Compute edge vectors and lengths
    
        with torch.no_grad():
            edge_src, edge_dst = edge_index
            vectors = positions[edge_dst] - positions[edge_src]
            lengths = torch.linalg.norm(vectors, dim=1, keepdim=True)

        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        
        with torch.no_grad():
            edge_attrs = self.spherical_harmonics(vectors)
            edge_feats, cutoff = self.radial_embedding(lengths, node_attrs, edge_index, None)

        # Message passing
        layers = zip(self.interactions, self.products)
        for i, (interaction, product) in enumerate(layers):
            if self.use_batchnorm:
                node_feats = self.layer_norms[i](node_feats)

            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )

        propensities = torch.zeros(node_feats.shape[0], self.num_outputs, device=node_feats.device)
        
        for element_type in range(self.num_elements):
            mask = (atom_types == element_type)
            
            if mask.sum() > 0:  # If there are atoms of this type
                element_propensities = self.propensity_readouts[element_type](node_feats[mask])
                propensities[mask] = element_propensities

        return {
            "propensities": propensities,
            "node_feats": node_feats, 
        }
