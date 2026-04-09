# FILE: src/models/pafdta.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pafdta_backbone import DrugEncoderVAE, ModalWeightFusion, MixedExpertFusion
from src.utils.contrastive import info_nce_loss


class LinearAttention(nn.Module):
    """Same idea as PAFDTA's linear attention pooling."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_second = nn.Linear(hidden_dim, heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, L, D]
        mask: [B, L] (bool) or [B, heads, L] (float/bool)
        return: [B, D]
        """
        B, L, D = x.shape
        sent_att = torch.tanh(self.linear_first(x))          # [B, L, H]
        sent_att = self.linear_second(sent_att)              # [B, L, heads]
        sent_att = sent_att.transpose(1, 2)                  # [B, heads, L]

        if mask.dim() == 2:
            m = mask.unsqueeze(1).expand(-1, self.heads, -1) # [B, heads, L]
        else:
            m = mask

        # mask: True=valid, False=pad
        if m.dtype != torch.bool:
            m_bool = m > 0.5
        else:
            m_bool = m

        minus_inf = torch.full_like(sent_att, -9e15)
        e = torch.where(m_bool, sent_att, minus_inf)
        att = self.softmax(e)                                # [B, heads, L]
        sent_emb = torch.matmul(att, x)                      # [B, heads, D]
        avg = torch.sum(sent_emb, dim=1) / float(self.heads) # [B, D]
        return avg


class PAFDTA(nn.Module):
    def __init__(
        self,
        protein_feat_dim: int = 640,
        latent_dim: int = 384,
        n_heads: int = 6,
        dropout: float = 0.25,
        tf_layers: int = 2,
        tf_ff_mult: int = 4,
        linear_attn_heads: int = 8,
        linear_attn_hidden: int = 64,
        use_hg: bool = True,
        hg_dim: int = 6,
        hg_weight: float = 0.35,
        lambda_align: float = 0.05,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_hg = bool(use_hg) and (hg_dim > 0)
        self.hg_dim = hg_dim
        self.hg_weight = float(hg_weight)
        self.lambda_align = float(lambda_align)
        self.temperature = float(temperature)

        # Drug encoder (keeps v5 multi-scale CNN local/global + VAE)
        self.drug_enc = DrugEncoderVAE(latent_dim=latent_dim)

        # Protein projection to latent_dim
        self.prot_proj = nn.Linear(protein_feat_dim, latent_dim) if protein_feat_dim != latent_dim else nn.Identity()

        # Transformer encoders (token-level)
        ff_dim = latent_dim * tf_ff_mult
        enc_layer_drug = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout
        )
        enc_layer_prot = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.drug_tf = nn.TransformerEncoder(enc_layer_drug, num_layers=tf_layers)
        self.prot_tf = nn.TransformerEncoder(enc_layer_prot, num_layers=tf_layers)

        # Linear attention pooling (PAFDTA-style)
        self.drug_attn = LinearAttention(latent_dim, hidden_dim=linear_attn_hidden, heads=linear_attn_heads)
        self.prot_attn = LinearAttention(latent_dim, hidden_dim=linear_attn_hidden, heads=linear_attn_heads)
        self.inter_attn = LinearAttention(latent_dim, hidden_dim=linear_attn_hidden, heads=linear_attn_heads)

        # HG branch + fusion (keeps v5)
        if self.use_hg:
            self.hg_bn = nn.BatchNorm1d(hg_dim, affine=True, momentum=0.1)
            self.hg_proj = nn.Linear(hg_dim, latent_dim)
            self.hg_drop = nn.Dropout(dropout)

            self.modal_fusion = ModalWeightFusion(latent_dim)
            self.mixed_fusion = MixedExpertFusion(latent_dim, dropout=dropout)

            # projection heads for InfoNCE
            self.z_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
            self.hg_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )

        # Regression head
        in_dim = latent_dim * (4 if self.use_hg else 3)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        drugs: List[str],
        prot_feats: torch.Tensor,   # [B, L, C]
        prot_mask: torch.Tensor,    # [B, L] bool (True=valid)
        hg: Optional[torch.Tensor] = None,  # [B, K]
    ):
        # ---- drug ----
        if self.use_hg:
            z, mu, logvar, drug_tokens, drug_mask = self.drug_enc(drugs, return_seq=True)
        else:
            z, mu, logvar, drug_tokens, drug_mask = self.drug_enc(drugs, return_seq=True)

        # ---- protein ----
        ctx_tokens = self.prot_proj(prot_feats.float())   # [B, L, D]

        # Transformer encoders need key_padding_mask: True indicates PAD
        drug_pad = ~drug_mask
        prot_pad = ~prot_mask
        # TransformerEncoder (older PyTorch) expects [L, B, D]
        drug_tokens_t = drug_tokens.transpose(0, 1)  # [Nd, B, D]
        ctx_tokens_t  = ctx_tokens.transpose(0, 1)   # [L,  B, D]

        drug_tokens_t = self.drug_tf(drug_tokens_t, src_key_padding_mask=drug_pad)  # [Nd, B, D]
        ctx_tokens_t  = self.prot_tf(ctx_tokens_t,  src_key_padding_mask=prot_pad)  # [L,  B, D]

        drug_tokens = drug_tokens_t.transpose(0, 1)  # [B, Nd, D]
        ctx_tokens  = ctx_tokens_t.transpose(0, 1)   # [B, L,  D]
        # PAFDTA-style pooled reps
        xd_attn = self.drug_attn(drug_tokens, drug_mask)   # [B, D]
        xt_attn = self.prot_attn(ctx_tokens,  prot_mask)   # [B, D]

        cat_tokens = torch.cat([ctx_tokens, drug_tokens], dim=1)                # [B, L+Nd, D]
        cat_mask   = torch.cat([prot_mask, drug_mask], dim=1)                   # [B, L+Nd]
        cat_attn   = self.inter_attn(cat_tokens, cat_mask)                      # [B, D]

        # ---- HG + fusion + contrastive ----
        info_loss = None
        if self.use_hg and (hg is not None):
            hg = hg.float()
            hg_norm = self.hg_bn(hg)
            hg_feat = self.hg_drop(self.hg_proj(hg_norm))                       # [B, D]

            fused_modal = self.modal_fusion(z, xt_attn, hg_feat)                # [B, D]
            fused_feat  = self.mixed_fusion(z, xt_attn, hg_feat, fused_modal)   # [B, D]
            fused_feat  = self.hg_weight * fused_feat

            # InfoNCE: align drug latent z with HG
            z_proj = self.z_head(z)
            hg_proj = self.hg_head(hg_feat)
            info_loss = info_nce_loss(z_proj, hg_proj, temperature=self.temperature)

            rep = torch.cat([xd_attn, cat_attn, xt_attn, fused_feat], dim=-1)   # [B, 4D]
            pred = self.mlp(rep).squeeze(-1)
            return pred, mu, logvar, info_loss
        else:
            rep = torch.cat([xd_attn, cat_attn, xt_attn], dim=-1)               # [B, 3D]
            pred = self.mlp(rep).squeeze(-1)
            return pred, mu, logvar
"""PAFDTA model definition.

Defines the PAFDTA architecture that fuses drug and protein representations and optionally incorporates heterogeneous-graph path-score features."""

