# -*- coding: utf-8 -*-
"""
Contrastive utilities for semantic alignment between latent drug–protein representations
and heterogeneous-graph-derived features.

Implements an InfoNCE loss over a batch of paired embeddings.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(z_proj: torch.Tensor,
                  hg_proj: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss between two sets of projected vectors.

    Args:
        z_proj: Tensor[B, D]   – projections of [z || ctx] or similar.
        hg_proj: Tensor[B, D]  – projections of heterogeneous-graph embeddings.
        temperature: softmax temperature (default: 0.07).

    Each index i in z_proj is considered a positive pair with index i in hg_proj.
    Other pairs in the batch are treated as negatives.
    """
    if z_proj.ndim != 2 or hg_proj.ndim != 2:
        raise ValueError("info_nce_loss expects 2D tensors: [B, D].")

    if z_proj.size(0) != hg_proj.size(0):
        raise ValueError("Batch size mismatch between z_proj and hg_proj.")

    # L2-normalize so dot products become cosine similarities
    z_norm = F.normalize(z_proj, dim=1)
    hg_norm = F.normalize(hg_proj, dim=1)

    # Similarity matrix [B, B]
    sim = torch.matmul(z_norm, hg_norm.t()) / temperature

    # Positives are on the diagonal
    targets = torch.arange(sim.size(0), device=sim.device)

    # z as anchor, heterogeneous-graph (HG) as candidate
    loss_i = F.cross_entropy(sim, targets)
    # heterogeneous-graph (HG) as anchor, z as candidate (transpose)
    loss_j = F.cross_entropy(sim.t(), targets)

    return 0.5 * (loss_i + loss_j)
"""Contrastive learning utilities (InfoNCE) used by PAFDTA."""

