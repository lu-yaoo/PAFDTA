# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm
import ast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.pafdta_dataset import PAFDTADataset, pad_collate_fn
from src.models.pafdta import PAFDTA
from src.utils.common import to_device
from src.utils.metrics import mse as mse_np, r2 as r2_np, aupr_from_scores, concordance_index, pearson_spearman
from src.utils.contrastive import info_nce_loss


# --------------------  :  --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------  : Load  --------------------
def _as_float(x, default=None):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(ast.literal_eval(x)) if any(ch.isalpha() for ch in x) else float(x)
        except Exception:
            try:
                return float(x)
            except Exception:
                return default
    return default

def _as_int(x, default=None):
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        try:
            return int(ast.literal_eval(x))
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return default
    return default

def _as_bool(x, default=None):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    return default


# --------------------  ( ) +  ( ) --------------------
# ======   train_esm.py  ,  TargetScaler   ======
class TargetScaler:
    """
     :y' = transform(y)
     :inverse_to_base(y') ->  (  y_unit   nM/M   base=neg_log10,  pKd)
    y_unit: "nM" / "M" / "pKd" / "auto"
    base  : "none" / "neg_log10"
    """
    def __init__(self, base: str = "neg_log10", standardize: bool = True,
                 log_floor: float = 1e-12, y_unit: str = "auto"):
        self.base = base
        self.standardize = standardize
        self.log_floor = float(log_floor)
        self.y_unit = (y_unit or "auto").lower()
        self.unit_scale = 1.0     #   M   -log10:nM -> 1e-9;M -> 1.0;pKd ->
        self.mean_ = None
        self.std_ = None

    def _infer_unit_scale(self, y_np: np.ndarray):
        """ .  y_unit  ."""
        if self.y_unit in ("pkd",):
            self.unit_scale = None  #   pKd
            return
        if self.y_unit in ("nm", "nmol", "nmolar", "nanomolar", "nmolarity", "nm ", "nm.", "nmol/l", "nm/l", "nm/l", "nM".lower()):
            self.unit_scale = 1e-9
            return
        if self.y_unit in ("m", "mol", "molar", "mol/l", "molarity"):
            self.unit_scale = 1.0
            return
        # auto  :  > 1,  nM; (<1e-3)  M;  3~12   pKd
        med, mx = float(np.nanmedian(y_np)), float(np.nanmax(y_np))
        if 2.5 <= med <= 12 and mx <= 20:
            #   pKd
            self.y_unit = "pkd"
            self.unit_scale = None
        elif med > 1e-2 or mx > 1.0:
            #   nM
            self.y_unit = "nM"
            self.unit_scale = 1e-9
        else:
            self.y_unit = "M"
            self.unit_scale = 1.0

    @property
    def output_scale_name(self):
        """inverse_to_base  :'pKd'   'Kd'"""
        if self.base == "neg_log10":
            return "pKd"      #   nM   M,  -log10(… in M)   pKd
        if self.y_unit == "pkd":
            return "pKd"
        return "Kd"

    def _base_forward(self, y_t: torch.Tensor) -> torch.Tensor:
        """
         " ":
          -   y_unit   nM/M   base=neg_log10:  unit_scale   M,  -log10   pKd
          -   y_unit   pKd:base=none  ;  base=neg_log10, (  log)
        """
        if self.y_unit == "pkd":
            return y_t  # Already pKd
        if self.base == "neg_log10":
            y_in_M = y_t * (self.unit_scale if self.unit_scale is not None else 1.0)
            return -torch.log10(torch.clamp(y_in_M, min=self.log_floor))
        return y_t

    def fit(self, train_dataset):
        ys = []
        for (lig_key, prot_key, lig_idx, prot_idx) in train_dataset.samples:
            val = train_dataset._lookup_label(lig_key, prot_key, lig_idx, prot_idx)
            ys.append(val)
        y_np = np.asarray(ys, dtype=float)
        self._infer_unit_scale(y_np)
        y_t = torch.tensor(y_np, dtype=torch.float32)
        yb = self._base_forward(y_t)
        if self.standardize:
            self.mean_ = float(yb.mean().item())
            self.std_ = float(yb.std(unbiased=False).item()) or 1.0
            print(f"[TargetTransform] base={self.base}, unit={self.y_unit}, "
                  f"standardize=True, mean={self.mean_:.6f}, std={self.std_:.6f}")
        else:
            print(f"[TargetTransform] base={self.base}, unit={self.y_unit}, standardize=False")

    def transform(self, y_t: torch.Tensor) -> torch.Tensor:
        yb = self._base_forward(y_t)
        if self.standardize:
            return (yb - self.mean_) / self.std_
        return yb

    def inverse_to_base(self, y_np: np.ndarray) -> np.ndarray:
        """
         " ":
          -   output_scale_name=='pKd':  pKd
          -   Kd( ).  pKd  .
        """
        y_np = np.asarray(y_np, dtype=float)
        if self.standardize:
            y_np = y_np * self.std_ + self.mean_
        # base='neg_log10'   y_unit='pKd'  ,y_np   pKd
        return y_np

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["davis", "kiba"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--feature_dir", type=str, required=True)
    p.add_argument("--config", type=str, default="./configs/PAFDTA_train.yaml")
    p.add_argument("--fold", type=int, default=0, help="Fold index for cross-validation (e.g., 0-4).")
    p.add_argument("--save_path", type=str, default="./checkpoints/pafdta.pth")
    return p.parse_args()


# --------------------  ( ) --------------------
# ======   train_esm.py  ,  evaluate   ======
def evaluate(model, loader, device, scaler: TargetScaler, cfg, dataset_name: str):
    model.eval()
    y_true_t, y_pred_t = [], []
    with torch.no_grad():
        for batch in loader:
            moved = to_device(batch, device)
            if len(moved) == 5:
                drugs, feats, mask, y, hg = moved
                pred, _, _ = model(drugs, feats, mask, hg)
            else:
                drugs, feats, mask, y = moved
                pred, _, _ = model(drugs, feats, mask)
            y_true_t.append(y.detach().cpu().numpy())
            y_pred_t.append(pred.detach().cpu().numpy())

    y_true_t = np.concatenate(y_true_t, axis=0)
    y_pred_t = np.concatenate(y_pred_t, axis=0)

    #  " "
    y_true_base = scaler.inverse_to_base(y_true_t)
    y_pred_base = scaler.inverse_to_base(y_pred_t)

    #  : " "
    m = mse_np(y_true_base, y_pred_base)
    r2 = r2_np(y_true_base, y_pred_base)
    ci = concordance_index(y_true_base, y_pred_base)

    # AUPR: dataset  pKd  (davis=7.0, kiba=12.1)
    if scaler.output_scale_name == "pKd":
        ds = (dataset_name or "").strip().lower()
        if ds == "davis":
            cutoff_pkd = 7.0
        elif ds == "kiba":
            cutoff_pkd = 12.1
        else:
            cutoff_pkd = _as_float(cfg.get("aupr_pkd_cutoff", 7.0), 7.0)
        y_bin = (y_true_base >= cutoff_pkd).astype(int)
        scores = y_pred_base  #  " "
    else:
        #   pKd( ),  Kd
        kd_cut = _as_float(cfg.get("aupr_kd_cutoff", 30e-9), 30e-9)
        y_bin = (y_true_base <= kd_cut).astype(int)
        scores = -y_pred_base  # Kd smaller indicates" ",negate

    try:
        aupr = aupr_from_scores(y_bin, scores)
    except Exception:
        aupr = float("nan")

    try:
        pr, sr = pearson_spearman(y_true_base, y_pred_base)
    except Exception:
        pr, sr = float("nan"), float("nan")

    return {"MSE": m, "R2": r2, "CI": ci, "AUPR": aupr, "Pearson": pr, "Spearman": sr}

# -------------------- Main pipeline --------------------
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # Type casting
    seed        = _as_int(cfg.get("seed", 42), 42)
    batch_size  = _as_int(cfg.get("batch_size", 32), 32)
    epochs      = _as_int(cfg.get("epochs", 20), 20)
    lr          = _as_float(cfg.get("lr", 1e-3), 1e-3)
    weight_decay = _as_float(cfg.get("weight_decay", 0.0), 0.0)
    latent_dim  = _as_int(cfg.get("latent_dim", 256), 256)
    n_heads     = _as_int(cfg.get("n_heads", 4), 4)
    dropout     = _as_float(cfg.get("dropout", 0.1), 0.1)
    protein_fd  = _as_int(cfg.get("protein_feat_dim", 640), 640)
    # heterogeneous graph(HG)
    cfg_use_hg = _as_bool(cfg.get("use_hg", True), True)
    hg_weight  = _as_float(cfg.get("hg_weight", 0.35), 0.35)
    beta_kl     = _as_float(cfg.get("beta_kl", 1e-3), 1e-3)
    max_gn      = _as_float(cfg.get("max_grad_norm", 5.0), 5.0)
    lambda_align = _as_float(cfg.get("lambda_align", 0.05), 0.05)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = PAFDTADataset(args.dataset, args.data_root, args.feature_dir, split="train", fold_id=args.fold)
    valid_set = PAFDTADataset(args.dataset, args.data_root, args.feature_dir, split="valid", fold_id=args.fold)
    test_set  = PAFDTADataset(args.dataset, args.data_root, args.feature_dir, split="test",  fold_id=args.fold)
    print(f"[Config] use_hg={cfg_use_hg} | detected_hg_dim={getattr(train_set, 'hg_dim', 0)} | hg_weight={hg_weight}")


    #  ( ) +  ( )
    scaler = TargetScaler(
        base=cfg.get("target_base", "neg_log10"),
        standardize=_as_bool(cfg.get("standardize", True), True),
        log_floor=_as_float(cfg.get("log_floor", 1e-9), 1e-9),
    )
    scaler.fit(train_set)
    train_set.target_transform = scaler.transform
    valid_set.target_transform = scaler.transform
    test_set.target_transform  = scaler.transform
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

    #
    # heterogeneous graph(HG) features (meta-path PathScores): auto-detect unless overridden
    use_hg_cfg = cfg.get("use_hg", None)
    if use_hg_cfg is None:
        use_hg = cfg_use_hg and ((getattr(train_set, "hg_dim", 0) or 0) > 0)
    else:
        use_hg = bool(use_hg_cfg)
    hg_dim = getattr(train_set, "hg_dim", 0) if use_hg else 0

    model = PAFDTA(latent_dim=latent_dim,
                            protein_feat_dim=protein_fd,
                            n_heads=n_heads,
                            dropout=dropout,
                            use_hg=use_hg,
                            hg_dim=hg_dim,
                            hg_weight=hg_weight,
                            lambda_align=lambda_align).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_mse = float("inf")
    best_metrics = None
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} - Training"):
            moved = to_device(batch, device)
            if len(moved) == 5:
                drugs, feats, mask, y, hg = moved
                if lambda_align > 0 and use_hg:
                    out = model(drugs, feats, mask, hg)
                    if len(out) == 5:
                        pred, mu, logvar, z_align, hg_align = out
                        info_loss = info_nce_loss(z_align, hg_align)
                    elif len(out) == 4:
                        pred, mu, logvar, info_loss = out
                    elif len(out) == 3:
                        pred, mu, logvar = out
                        info_loss = 0.0
                    else:
                        raise model(drugs, feats, mask, hg)
                else:
                    pred, mu, logvar = model(drugs, feats, mask, hg)
                    info_loss = torch.tensor(0.0, device=device)
            else:
                drugs, feats, mask, y = moved
                pred, mu, logvar = model(drugs, feats, mask)
                info_loss = torch.tensor(0.0, device=device)

            # regression (MSE) " "( )
            loss_pred = F.mse_loss(pred, y)

            # VAE KL(beta  );  GAN,
            kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / mu.size(0)
            loss = loss_pred + beta_kl * kl + lambda_align * info_loss

            opt.zero_grad()
            loss.backward()
            if max_gn is not None and max_gn > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
            opt.step()

            total += loss_pred.item() * y.size(0)
            n += y.size(0)

        tr_mse_train_scale = total / n  #   MSE( )

        # ——  : " "  —— #
        metrics = evaluate(model, valid_loader, device, scaler, cfg, args.dataset)
        val_mse = metrics["MSE"]

        print(
            f"[Epoch {ep}] Train(MSE@train-scale): {tr_mse_train_scale:.6f} | "
            f"Valid(MSE/CI/R2/AUPR): {metrics['MSE']:.4f} / {metrics['CI']:.4f} / {metrics['R2']:.4f} / {metrics['AUPR']:.4f} "
            f"| (Pearson={metrics['Pearson']:.4f}, Spearman={metrics['Spearman']:.4f})"
        )

        if val_mse < best_mse:
            best_mse = val_mse
            best_metrics = dict(metrics)
            torch.save(model.state_dict(), args.save_path)

    if best_metrics is not None:
        print(
            f"[Done] Best Valid metrics (on base scale): "
            f"MSE={best_metrics['MSE']:.6f}, CI={best_metrics['CI']:.4f}, "
            f"R2={best_metrics['R2']:.4f}, AUPR={best_metrics['AUPR']:.4f} | "
            f"Pearson={best_metrics['Pearson']:.4f}, Spearman={best_metrics['Spearman']:.4f}; "
            f"model saved to {args.save_path}"
        )
    else:
        print(f"[Done] Best Valid MSE (on base scale): {best_mse:.6f}, model saved to {args.save_path}")


    # ——   test_fold   best checkpoint ——
    if best_metrics is not None:
        try:
            model.load_state_dict(torch.load(args.save_path, map_location=device))
            model.eval()
            final_metrics = evaluate(model, test_loader, device, scaler, cfg, args.dataset)
            print("[Final] Test(MSE/CI/R2/AUPR): {:.4f} / {:.4f} / {:.4f} / {:.4f} | (Pearson={:.4f}, Spearman={:.4f})".format(
                final_metrics["MSE"], final_metrics["CI"], final_metrics["R2"], final_metrics["AUPR"],
                final_metrics["Pearson"], final_metrics["Spearman"],
            ))
        except Exception as e:
            print(f"[Final] Failed to evaluate on test_fold: {e}")


if __name__ == "__main__":

    main()