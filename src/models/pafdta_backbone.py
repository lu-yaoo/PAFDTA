# FILE: src/models/pafdta_backbone.py
# -*- coding: utf-8 -*-
"""
  PyTorch   protein tokenVAE  :
- protein :  MultiheadAttention(  batch_first / norm_first)
- drug :  SMILES + GRU   VAE( drug )
- heterogeneous graph :BN +   + Dropout +  (  [z, ctx]   gate)+
- regression :MLP
forward(drugs, prot_feats, prot_mask, hg_feats=None) -> (pred, mu, logvar)

 :
- drugs: List[str](SMILES)  LongTensor[B, T];
- prot_feats: FloatTensor[B, L, C],C=protein_feat_dim( :640)
- prot_mask:  BoolTensor[B, L],True= ,False=pad
- hg_feats:   FloatTensor[B, hg_dim]( )

 :  torch<=1.8/1.9/1.10   batch_first
"""
import string
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem


# ---------------------------
#
class DrugEncoderVAE(nn.Module):
    """
      GIN   VAE  :
    -  :List[str] SMILES( );
    -  :  RDKit   SMILES  ( = , = ),
       / ( , , ),
        GIN-style   +  drug  z,
        VAE   (z, mu, logvar).
     :
      *  ,  List[str]  ;
          LongTensor  .
    """
    def __init__(
        self,
        latent_dim: int = 384,
        num_gnn_layers: int = 4,
        pool: str = "sum",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_gnn_layers = num_gnn_layers
        assert pool in ["mean", "sum"], "pool must be 'mean' or 'sum'"
        self.pool = pool

        #  /
        self.max_atomic_num = 60          # 1..60  ,>60   one-hot
        self.num_hybridization = 6        # SP, SP2, SP3, SP3D, SP3D2, OTHER
        self.num_formal_charge = 7        # -3..+3
        self.num_h_count = 6              # 0..4, 5+
        self.num_degree = 6               # 0..4, 5+
        self.atom_feat_dim = (
            self.max_atomic_num + 1 +     # atomic number
            1 +                           # is_aromatic
            1 +                           # in_ring
            self.num_hybridization +
            self.num_formal_charge +
            self.num_h_count +
            self.num_degree
        )
        #  : / /  + aromatic + conjugated + in_ring
        self.bond_feat_dim = 3 + 1 + 1 + 1

        #   latent_dim
        self.atom_fc = nn.Linear(self.atom_feat_dim, self.latent_dim)

        # GIN  ( )
        self.gnn_layers = nn.ModuleList([
            _SimpleGINLayer(self.latent_dim, self.bond_feat_dim)
            for _ in range(self.num_gnn_layers)
        ])

        # VAE
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # --------- Multi-scale 1D CNN branch (depth path) + gating fusion ---------
        #   hidden_dim (latent_dim)   3  , ,
        #  .
        self.ms_num_scales = 3
        assert self.latent_dim % self.ms_num_scales == 0, "latent_dim   3  "
        group_dim = self.latent_dim // self.ms_num_scales

        #   kernel size   Conv1d, " " (3/5/7)
        self.ms_convs = nn.ModuleList([
            nn.Conv1d(group_dim, group_dim, kernel_size=1, padding=1),
            nn.Conv1d(group_dim, group_dim, kernel_size=3, padding=2),
            nn.Conv1d(group_dim, group_dim, kernel_size=5, padding=3),
        ])

        #   +    :  [global, local]
        self.fuse_gate_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # 1D CNN  (  max-pooling)
        self.local_attn_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.Tanh(),
            nn.Linear(self.latent_dim // 2, 1),
        )

    # ---------- RDKit featurization ----------
    def _atom_features(self, atom: "rdchem.Atom") -> torch.Tensor:
        # atomic number one-hot (1..max_atomic_num, else last)
        z = atom.GetAtomicNum()
        atomic = torch.zeros(self.max_atomic_num + 1, dtype=torch.float32)
        if z <= self.max_atomic_num and z > 0:
            atomic[z - 1] = 1.0
        else:
            atomic[-1] = 1.0

        # aromaticity & ring
        is_aromatic = torch.tensor([1.0 if atom.GetIsAromatic() else 0.0], dtype=torch.float32)
        in_ring = torch.tensor([1.0 if atom.IsInRing() else 0.0], dtype=torch.float32)

        # hybridization
        hyb = atom.GetHybridization()
        hyb_vec = torch.zeros(self.num_hybridization, dtype=torch.float32)
        hyb_map = {
            rdchem.HybridizationType.SP: 0,
            rdchem.HybridizationType.SP2: 1,
            rdchem.HybridizationType.SP3: 2,
            rdchem.HybridizationType.SP3D: 3,
            rdchem.HybridizationType.SP3D2: 4,
        }
        idx = hyb_map.get(hyb, 5)
        hyb_vec[idx] = 1.0

        # formal charge: -3..+3
        charge = atom.GetFormalCharge()
        charge = max(-3, min(3, int(charge)))
        charge_vec = torch.zeros(self.num_formal_charge, dtype=torch.float32)
        charge_vec[charge + 3] = 1.0

        # num H: 0..4, 5+
        hcount = atom.GetTotalNumHs()
        h_idx = int(hcount) if hcount <= 4 else 5
        h_vec = torch.zeros(self.num_h_count, dtype=torch.float32)
        h_vec[h_idx] = 1.0

        # degree: 0..4, 5+
        deg = atom.GetDegree()
        d_idx = int(deg) if deg <= 4 else 5
        d_vec = torch.zeros(self.num_degree, dtype=torch.float32)
        d_vec[d_idx] = 1.0

        return torch.cat([atomic, is_aromatic, in_ring, hyb_vec, charge_vec, h_vec, d_vec], dim=0)

    def _bond_features(self, bond: "rdchem.Bond") -> torch.Tensor:
        if bond is None:
            return torch.zeros(self.bond_feat_dim, dtype=torch.float32)

        bt = bond.GetBondType()
        bond_type = torch.zeros(3, dtype=torch.float32)
        if bt == rdchem.BondType.SINGLE:
            bond_type[0] = 1.0
        elif bt == rdchem.BondType.DOUBLE:
            bond_type[1] = 1.0
        elif bt == rdchem.BondType.TRIPLE:
            bond_type[2] = 1.0

        aromatic = torch.tensor([1.0 if bond.GetIsAromatic() else 0.0], dtype=torch.float32)
        conjugated = torch.tensor([1.0 if bond.GetIsConjugated() else 0.0], dtype=torch.float32)
        in_ring = torch.tensor([1.0 if bond.IsInRing() else 0.0], dtype=torch.float32)
        return torch.cat([bond_type, aromatic, conjugated, in_ring], dim=0)


    def _smiles_to_graph_cpu(self, smi: str):
        """
          CPU  ,  cache.  CPU  .
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            #   (CPU)
            x = torch.zeros(1, self.atom_feat_dim, dtype=torch.float32)
            edge_index = torch.zeros(0, 2, dtype=torch.long)
            edge_attr = torch.zeros(0, self.bond_feat_dim, dtype=torch.float32)
            return x, edge_index, edge_attr

        num_atoms = mol.GetNumAtoms()
        atom_feats = [self._atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
        x = torch.stack(atom_feats, dim=0)  # [N, F] on CPU

        rows, cols, bond_feats = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = self._bond_features(bond)
            #  :i->j, j->i
            rows.extend([i, j])
            cols.extend([j, i])
            bond_feats.extend([bf, bf])

        if len(rows) == 0:
            edge_index = torch.zeros(0, 2, dtype=torch.long)
            edge_attr = torch.zeros(0, self.bond_feat_dim, dtype=torch.float32)
        else:
            edge_index = torch.tensor(list(zip(rows, cols)), dtype=torch.long)
            edge_attr = torch.stack(bond_feats, dim=0)

        return x, edge_index, edge_attr

    def _get_graph_from_cache(self, smi: str):
        """
          CPU cache  ; .
        """
        if not hasattr(self, "graph_cache"):
            self.graph_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        if smi in self.graph_cache:
            x_cpu, edge_index_cpu, edge_attr_cpu = self.graph_cache[smi]
        else:
            x_cpu, edge_index_cpu, edge_attr_cpu = self._smiles_to_graph_cpu(smi)
            self.graph_cache[smi] = (x_cpu, edge_index_cpu, edge_attr_cpu)

        return x_cpu, edge_index_cpu, edge_attr_cpu

    def _smiles_to_graph(self, smi: str, device: torch.device):
        """
         :  device  .
        """
        x_cpu, edge_index_cpu, edge_attr_cpu = self._get_graph_from_cache(smi)
        x = x_cpu.to(device)
        edge_index = edge_index_cpu.to(device)
        edge_attr = edge_attr_cpu.to(device)
        return x, edge_index, edge_attr

    # ---------- VAE forward ----------
    def forward(self, drugs, return_seq: bool = False):
        """
         :
          - drugs: List[str] SMILES( )
         :
          - z, mu, logvar : [B, latent_dim]
        """
        if torch.is_tensor(drugs):
            raise ValueError("  DrugEncoderVAE   List[str] SMILES  .")

        batch_size = len(drugs)
        device = next(self.parameters()).device

        hs = []
        lens = []
        mus = []
        logvars = []
        for smi in drugs:
            # 1)   + GIN  :  h_i
            x, edge_index, edge_attr = self._smiles_to_graph(smi, device)   # [N, F], [E, 2], [E, Fe]
            h = self.atom_fc(x)                                             # [N, D]
            for layer in self.gnn_layers:
                h = layer(h, edge_index, edge_attr)                         # [N, D]

            if return_seq:
                hs.append(h)  # [N, D]
                lens.append(int(h.size(0)))
            # 2)  (width path)
            if self.pool == "sum":
                global_feat = h.sum(dim=0)
            else:
                global_feat = h.mean(dim=0)                                 # [D]

            # 3)   1D CNN +  (depth path)
            #      [N, D]   N  " ",  3   kernel=3/5/7   Conv1d,
            #      max-pooling.
            if h.size(0) == 1:
                #  ,
                local_feat = global_feat
            else:
                seq = h.unsqueeze(0).transpose(1, 2)                         # [1, D, N]
                D = seq.size(1)
                group_dim = D // self.ms_num_scales
                #   3  ,  kernel   Conv1d
                chunks = torch.split(seq, group_dim, dim=1)
                conv_outs = []
                for chunk, conv in zip(chunks, self.ms_convs):
                    hi = F.relu(conv(chunk), inplace=True)                   # [1, group_dim, N]
                    conv_outs.append(hi)
                conv_seq = torch.cat(conv_outs, dim=1)                       # [1, D, N]

                #  :
                conv_seq_T = conv_seq.transpose(1, 2)                        # [1, N, D]
                attn_logits = self.local_attn_mlp(conv_seq_T)                # [1, N, 1]
                attn_weights = torch.softmax(attn_logits, dim=1)             # [1, N, 1]
                local_feat = (conv_seq_T * attn_weights).sum(dim=1).squeeze(0)  # [D]

            # 4)  :  global/local
            fuse_inp = torch.cat([global_feat, local_feat], dim=-1)          # [2D]
            gate = torch.sigmoid(self.fuse_gate_mlp(fuse_inp))               # [D]
            fused = gate * local_feat + (1.0 - gate) * global_feat           # [D]

            # 5) VAE  : regression mu/logvar
            mu = self.fc_mu(fused)
            logvar = self.fc_logvar(fused)
            mus.append(mu)
            logvars.append(logvar)

        mu = torch.stack(mus, dim=0)              # [B, D]
        logvar = torch.stack(logvars, dim=0)      # [B, D]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                        # reparameterization trick


        if not return_seq:
            return z, mu, logvar

        # pad atom-level sequences (hs) to [B, Nmax, D] and create mask [B, Nmax]
        if len(hs) == 0:
            # should not happen; fallback to single-token sequence of z
            hs_padded = z.unsqueeze(1)
            hs_mask = torch.ones(z.size(0), 1, dtype=torch.bool, device=z.device)
        else:
            max_n = max(lens) if len(lens) > 0 else 1
            hs_padded = torch.zeros(z.size(0), max_n, z.size(1), dtype=z.dtype, device=z.device)
            hs_mask = torch.zeros(z.size(0), max_n, dtype=torch.bool, device=z.device)
            for bi, h in enumerate(hs):
                n = h.size(0)
                hs_padded[bi, :n, :] = h
                hs_mask[bi, :n] = True
        return z, mu, logvar, hs_padded, hs_mask


class _SimpleGINLayer(nn.Module):
    """
      PyTorch   GIN-style  :
      -  :h_u + W_e(e_{uv})
      -  :
      -  :MLP(h_v + sum_u msg_{u->v})
      torch_scatter/pyG,  Python  .
    """
    def __init__(self, hidden_dim: int, edge_feat_dim: int):
        super().__init__()
        self.edge_fc = nn.Linear(edge_feat_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,           # [N, D]
        edge_index: torch.Tensor,  # [E, 2]
        edge_attr: torch.Tensor,   # [E, Fe]
    ) -> torch.Tensor:
        N, D = h.shape
        if edge_index.numel() == 0:
            return self.mlp(h)

        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        msg = h[src] + self.edge_fc(edge_attr)    # [E, D]

        agg = torch.zeros_like(h)
        #   index_add_   Python  ,
        agg.index_add_(0, dst, msg)

        out = h + agg
        out = self.mlp(out)
        return out
#-------
# ( )  :
# ---------------------------
class Generator(nn.Module):
    """  z   MLP  (  WGAN-GP  ,  )"""
    def __init__(self, latent_dim=384, noise_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, noise):
        return self.net(noise)


class Discriminator(nn.Module):
    """WGAN   critic,  z  ( , )"""
    def __init__(self, latent_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, z):
        return self.net(z).view(-1)


# ---------------------------
# Main model
# ---------------------------


class ModalWeightFusion(nn.Module):
    """
     :
        z, ctx, hg (  [B, D]), Generate ,
       .
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def forward(self, z: torch.Tensor, ctx: torch.Tensor, hg: torch.Tensor) -> torch.Tensor:
        # z, ctx, hg: [B, D]
        #   summary   [B, 3]
        s = torch.stack(
            [
                z.mean(dim=-1),
                ctx.mean(dim=-1),
                hg.mean(dim=-1),
            ],
            dim=-1,
        )  # [B, 3]

        alpha = torch.softmax(self.gate(s), dim=-1)  # [B, 3]
        z_w = alpha[:, 0:1]
        ctx_w = alpha[:, 1:2]
        hg_w = alpha[:, 2:3]

        fused = z_w * z + ctx_w * ctx + hg_w * hg
        return fused


class MixedExpertFusion(nn.Module):
    """
     :
      - Expert1:  , ;
      - Expert2:   MLP  , ;
      - Expert3:   (z*ctx, z*hg, ctx*hg).
        gate  Generate 3  , .
    """
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        in_dim = dim * 4  # concat [z, ctx, hg, fused_modal] -> [B, 4D]

        # Expert 1:
        self.exp1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Expert 2:   MLP
        self.exp2 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Expert 3:
        pair_dim = dim * 3  # [z*ctx, z*hg, ctx*hg]
        self.exp3 = nn.Sequential(
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # gate:
        self.gate = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

    def forward(
        self,
        z: torch.Tensor,
        ctx: torch.Tensor,
        hg: torch.Tensor,
        fused_modal: torch.Tensor,
    ) -> torch.Tensor:
        base = torch.cat([z, ctx, hg, fused_modal], dim=-1)  # [B, 4D]

        f1 = self.exp1(base)  # [B, D]
        f2 = self.exp2(base)  # [B, D]

        #
        z_ctx = z * ctx
        z_hg = z * hg
        ctx_hg = ctx * hg
        pair = torch.cat([z_ctx, z_hg, ctx_hg], dim=-1)  # [B, 3D]
        f3 = self.exp3(pair)  # [B, D]

        gate_logits = self.gate(base)      # [B, 3]
        alpha = torch.softmax(gate_logits, dim=-1)  # [B, 3]

        fused = (
            alpha[:, 0:1] * f1 +
            alpha[:, 1:2] * f2 +
            alpha[:, 2:3] * f3
        )
        return fused


class PAFDTABackbone(nn.Module):
    """
    drug  VAE +   protein token token   +  (AAP) +   heterogeneous graph  + regression
    -  :  MultiheadAttention(  batch_first), /  [L, B, D]
    """
    def __init__(
        self,
        protein_feat_dim: int,      # ESM per-token hidden dim( :640)
        latent_dim: int = 384,
        n_heads: int = 6,
        dropout: float = 0.25,
        use_hg: bool = False,
        hg_dim: int = 0,
        hg_weight: float = 0.35,    #   heterogeneous graph(HG)  ,
        lambda_align: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_hg = bool(use_hg)
        self.hg_dim = int(hg_dim)
        self.hg_weight = float(hg_weight)
        self.lambda_align = float(lambda_align)

        # drug  VAE
        self.drug_encoder = DrugEncoderVAE(latent_dim=self.latent_dim)

        # protein :  ESM token   latent_dim,
        self.proj = (
            nn.Linear(protein_feat_dim, self.latent_dim)
            if protein_feat_dim != self.latent_dim else nn.Identity()
        )

        #  :  batch_first;  [L, B, D]
        self.attn = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=n_heads, dropout=dropout)

        # heterogeneous graph
        if self.use_hg and self.hg_dim > 0:
            self.hg_bn = nn.BatchNorm1d(self.hg_dim, affine=True, momentum=0.1)
            self.hg_proj = nn.Linear(self.hg_dim, self.latent_dim)
            self.hg_drop = nn.Dropout(dropout)
            # gate   [z, ctx] ->  (0,1)
            self.hg_gate = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, 1),
                nn.Sigmoid(),
            )
            #  (  + use_hg   InfoNCE  )
            self.align_zctx = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.align_hg = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.hg_bn = None
            self.hg_proj = None
            self.hg_drop = None
            self.hg_gate = None
            self.align_zctx = None
            self.align_hg = None

        #  :  +
        self.modal_fusion = ModalWeightFusion(self.latent_dim)
        self.moe_fusion = MixedExpertFusion(self.latent_dim, dropout=dropout)

        # regression MLP:  latent_dim
        fusion_dim = self.latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim // 2, 1),
        )

# ( ) ,  train_esm.py
        self.generator = Generator(latent_dim=self.latent_dim)
        self.discriminator = Discriminator(latent_dim=self.latent_dim)

    # --------- helper: attention pooling ----------
    def _attend(self, z: torch.Tensor, prot_feats: torch.Tensor, prot_mask: torch.Tensor) -> torch.Tensor:
        """
         drug  z   query, protein token  ,  ctx.
         :
          z:          [B, D]
          prot_feats: [B, L, D] (  latent_dim)
          prot_mask:  [B, L]    (True= ,False=pad)
         :
          ctx:        [B, D]
        """
        B, L, D = prot_feats.shape

        # q: [1, B, D];k,v: [L, B, D]
        q = z.unsqueeze(1).transpose(0, 1)         # [1, B, D]
        k = prot_feats.transpose(0, 1)             # [L, B, D]
        v = k

        # MultiheadAttention   key_padding_mask   True= ,  mask
        key_padding_mask = ~prot_mask.bool()       # [B, L]

        ctx, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)  # ctx: [1, B, D]
        ctx = ctx.squeeze(0)                        # [B, D]

        #
        ctx = F.layer_norm(ctx, (D,))
        return ctx

    # --------- forward ----------
    def forward(
        self,
        drugs,                        # List[str]   LongTensor[B, T]
        prot_feats: torch.Tensor,     # [B, L, C]
        prot_mask: torch.Tensor,      # [B, L](True= )
        hg_feats: Optional[torch.Tensor] = None,  # [B, hg_dim]   None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         :pred[B], mu[B, D], logvar[B, D]
        """
        # drug :VAE
        z, mu, logvar = self.drug_encoder(drugs)               # [B, D], [B, D], [B, D]

        # protein :token -> latent_dim,
        k = self.proj(prot_feats)                              # [B, L, D]
        ctx = self._attend(z, k, prot_mask)                    # [B, D]

        # heterogeneous graph ( )
        align_zctx = None
        align_hg = None

        if self.use_hg and (self.hg_proj is not None) and (hg_feats is not None):
            # BN   [B, C]
            kx = self.hg_bn(hg_feats)                          # [B, hg_dim]
            kx = self.hg_proj(kx)                              # [B, D]
            kx = self.hg_drop(kx)

            # gate: [B,1],  [z, ctx]
            g = self.hg_gate(torch.cat([z, ctx], dim=-1))      # [B, 1] in (0, 1)
            hg_vec = kx * g * self.hg_weight                   # [B, D]

            #  (  InfoNCE  )
            if self.training and self.lambda_align > 0 and (self.align_zctx is not None) and (self.align_hg is not None):
                align_zctx = self.align_zctx(torch.cat([z, ctx], dim=-1))  # [B, D]
                align_hg = self.align_hg(kx)                               # [B, D]
        else:
            #  heterogeneous graph , ,
            hg_vec = torch.zeros_like(z)

        #  : ,
        fused_modal = self.modal_fusion(z, ctx, hg_vec)                    # [B, D]
        fused = self.moe_fusion(z, ctx, hg_vec, fused_modal)              # [B, D]

        pred = self.mlp(fused).squeeze(-1)                                # [B]

        #   + heterogeneous graph :
        if self.training and self.use_hg and self.lambda_align > 0 and (align_zctx is not None) and (align_hg is not None):
            return pred, mu, logvar, align_zctx, align_hg

        return pred, mu, logvar

# --------- ( ) GAN   ----------
    def forward_with_z(
        self,
        drugs,
        prot_feats: torch.Tensor,
        prot_mask: torch.Tensor,
        hg_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         :pred, z, mu, logvar
        """
        z, mu, logvar = self.drug_encoder(drugs)               # [B, D]
        k = self.proj(prot_feats)                              # [B, L, D]
        ctx = self._attend(z, k, prot_mask)                    # [B, D]

        if self.use_hg and (self.hg_proj is not None) and (hg_feats is not None):
            kx = self.hg_bn(hg_feats)
            kx = self.hg_proj(kx)
            kx = self.hg_drop(kx)
            g = self.hg_gate(torch.cat([z, ctx], dim=-1))
            hg_vec = kx * g * self.hg_weight
        else:
            hg_vec = torch.zeros_like(z)

        fused_modal = self.modal_fusion(z, ctx, hg_vec)
        fused = self.moe_fusion(z, ctx, hg_vec, fused_modal)

        pred = self.mlp(fused).squeeze(-1)
        return pred, z, mu, logvar
"""Core building blocks for PAFDTA.

Includes a drug variational encoder, protein token attention pooling, optional heterogeneous-graph feature projection/fusion, and a regression head."""

