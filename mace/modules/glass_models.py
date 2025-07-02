###########################################################################################
# Implementation of MACE models for Glasses
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

# from mace.modules.embeddings import GenericJointEmbedding
# from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_sum

from .blocks import (
    # AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    # LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    # NonLinearDipoleReadoutBlock,
    # NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    # ScaleShiftBlock,
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
class MinimalMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,  # (particle A and B)
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        num_outputs: int = 10,  # 10 time steps?
    ):
        super().__init__()

        # Basic setup
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Embeddings
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])

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
            radial_MLP = [64, 64, 64]

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

        # Readout
        self.propensity_readout = LinearReadoutBlock(
            hidden_irreps,
            o3.Irreps(f"{num_outputs}x0e"),  # 10 scalar outputs per node
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = data["pos_th"]
        node_attrs = data["x"]
        edge_index = data["edge_index_th"]
        batch = data.get("batch", torch.zeros(positions.shape[0], dtype=torch.long))

        # Compute edge vectors and lengths
        edge_src, edge_dst = edge_index
        vectors = positions[edge_dst] - positions[edge_src]  # Assuming no PBC for now
        lengths = torch.linalg.norm(vectors, dim=1)

        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Message passing
        for interaction, product in zip(self.interactions, self.products):
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

        # Predict
        propensities = self.propensity_readout(node_feats)  # [n_nodes, 10]

        return {
            "propensities": propensities,
            "node_feats": node_feats, 
        }
