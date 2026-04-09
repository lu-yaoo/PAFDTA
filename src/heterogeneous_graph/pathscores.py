# FILE: src/heterogeneous_graph/pathscores.py
# -*- coding: utf-8 -*-
"""
Compute meta-path PathScores features for DTA using only the provided Davis/KIBA files,
following Affinity2Vec (Sci Rep 2022) meta-paths up to length 3:
C1: D-D-T, C2: D-T-T, C3: D-D-D-T, C4: D-T-T-T, C5: D-D-T-T, C6: D-T-D-T.

We build probability transition matrices from:
- DD similarity (nD x nD)
- TT similarity (nT x nT)
- DT affinity (nD x nT) preprocessed to [0,1] (pKd for Davis; inverted KIBA for stronger=larger)

IMPORTANT: To avoid data leakage, DT is constructed from TRAIN pairs only (others set to 0).
"""

import os
from typing import Dict, Tuple, List, Optional
import numpy as np

# ---------------------------
# Robust text readers/utils
# ---------------------------

def _read_matrix_txt(path: str, sep: str = None) -> np.ndarray:
    """
    Robust reader for dense matrices stored in text files.
    - sep=None => split on any whitespace (recommended)
    - Tolerates extra header row/col (n x (n+1)) / ((n+1) x n)
    - Ensures 2D even when single column is read
    """
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(sep) if sep is not None else line.split()
            parts = [p for p in parts if p != ""]
            data.append(parts)

    if len(data) == 0:
        return np.zeros((0, 0), dtype=float)

    # Try convert to float
    try:
        M = np.array(data, dtype=float)
    except Exception:
        #  FILE ,  float,  0
        rows = []
        for row in data:
            conv = []
            for x in row:
                try:
                    conv.append(float(x))
                except Exception:
                    conv.append(0.0)
            rows.append(conv)
        maxlen = max(len(r) for r in rows)
        rows = [r + [0.0] * (maxlen - len(r)) for r in rows]
        M = np.array(rows, dtype=float)

    if M.ndim == 1:
        M = M.reshape(-1, 1)

    # Strip possible header-like extra col/row
    if M.shape[1] == M.shape[0] + 1:
        M = M[:, 1:]
    if M.shape[0] == M.shape[1] + 1:
        M = M[1:, :]

    # Replace non-finite
    M[~np.isfinite(M)] = 0.0
    return M


def _minmax_01(M: np.ndarray, invert: bool = False) -> np.ndarray:
    A = M.astype(float).copy()
    finite = np.isfinite(A)
    if not np.any(finite):
        return np.zeros_like(A, dtype=float)
    lo = float(np.nanmin(A[finite]))
    hi = float(np.nanmax(A[finite]))
    if hi <= lo:
        A[finite] = 0.0
    else:
        A[finite] = (A[finite] - lo) / (hi - lo)
    if invert:
        A[finite] = 1.0 - A[finite]
    A[~finite] = 0.0
    return A


def _row_normalize(M: np.ndarray) -> np.ndarray:
    """
    Row-normalize to make a row-stochastic matrix.
    Rows whose sum <= 0 stay zero.
    This implementation avoids shape/broadcasting issues.
    """
    import numpy as np

    A = np.asarray(M, dtype=float)
    # clip negatives to zero ( / )
    A[A < 0] = 0.0

    # Row sums with keepdims -> shape (n, 1)
    rs = A.sum(axis=1, keepdims=True)

    # Divide each row by its sum; where=rs>0 ensures zero-rows remain zero
    # Broadcasting: (n, m) / (n, 1) -> (n, m)
    np.divide(A, rs, out=A, where=rs > 0)

    return A


def _build_DT_from_train(Y: np.ndarray, train_pairs: List[Tuple[int, int]], dataset: str) -> np.ndarray:
    """
    Build D->T strengths in [0,1] using only training pairs.
    - Davis: If values look like Kd(nM) -> convert to pKd = -log10(M) ≈ 9 - log10(nM), then min-max.
    - KIBA : Lower score = stronger -> inverse min-max.
    """
    nD, nT = Y.shape
    DT = np.zeros((nD, nT), dtype=float)

    # Gather train values
    vals = []
    for li, pi in train_pairs:
        if 0 <= li < nD and 0 <= pi < nT:
            v = Y[li, pi]
            if np.isfinite(v):
                vals.append(v)
    vals = np.array(vals, dtype=float) if len(vals) else np.array([], dtype=float)

    if dataset == "davis":
        if vals.size == 0:
            return DT
        # Heuristic check Kd(nM)
        med = float(np.median(vals))
        if med > 100:  # Heuristic threshold:  nM
            # Convert:nM -> M -> pKd
            DT_m = np.zeros_like(DT)
            mask = DT > 0
            Kd_M = Y * 1e-9
            DT_m[mask] = -np.log10(np.clip(Kd_M[mask], 1e-12, None))
            W = _minmax_01(DT_m, invert=False)
        else:
            # Already pKd /
            DT_p = np.zeros_like(DT)
            mask = DT > 0
            DT_p[mask] = Y[mask]
            W = _minmax_01(DT_p, invert=False)
    else:
        # KIBA: = ,  min-max
        if vals.size == 0:
            return DT
        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        W = np.zeros_like(DT)
        mask = np.isfinite(Y)
        W[mask] = 1.0 - (Y[mask] - mn) / (mx - mn + 1e-12)
        W[~np.isfinite(W)] = 0.0
        W = np.clip(W, 0.0, 1.0)

    # Write training edges only
    out = np.zeros_like(DT)
    for li, pi in train_pairs:
        if 0 <= li < nD and 0 <= pi < nT and np.isfinite(W[li, pi]):
            out[li, pi] = float(W[li, pi])
    return out


# ---------------------------
# Main API
# ---------------------------

def compute_pathscores_all(data_dir: str,
                           dataset: str,
                           train_pairs: List[Tuple[int, int]],
                           out_npz_path: str) -> Dict[str, object]:
    """
    data_dir   : path that contains ligands_* / proteins.txt / similarities / Y
    dataset    : "davis" | "kiba"
    train_pairs: (drug_idx, target_idx) list from TRAIN split
    out_npz_path: output file
    """
    dataset = dataset.lower()
    if dataset not in ("davis", "kiba"):
        raise ValueError(f"Unsupported dataset: {dataset}")

    # ---- Load DD, TT, Y with robust separators ----
    if dataset == "davis":
        dd_path = os.path.join(data_dir, "drug-drug_similarities_2D.txt")
        tt_path = os.path.join(data_dir, "target-target_similarities_WS.txt")
        y_path  = os.path.join(data_dir, "Y")

        DD = _read_matrix_txt(dd_path, sep=None)
        TT = _read_matrix_txt(tt_path, sep=None)

        Y = None
        if os.path.exists(y_path):
            try:
                import pickle
                with open(y_path, "rb") as f:
                    Y = pickle.load(f, encoding="latin1")
                Y = np.array(Y, dtype=float)
            except Exception:
                Y = None
        if Y is None:
            # Fallback: Load
            Y = _read_matrix_txt(y_path, sep=None)

    else:  # KIBA
        tt_path = os.path.join(data_dir, "kiba_target_sim.txt")
        y_path  = os.path.join(data_dir, "kiba_binding_affinity_v2.txt")
        dd_path = os.path.join(data_dir, "kiba_drug_sim.txt")  #

        TT = _read_matrix_txt(tt_path, sep=None)
        Y  = _read_matrix_txt(y_path,  sep=None)

        #   DD,  D-D  : ;  Y
        if os.path.exists(dd_path):
            DD = _read_matrix_txt(dd_path, sep=None)
        else:
            DD = np.eye(Y.shape[0], dtype=float)

    # ---- Shape sanity & orient to (nD, nT) ----
    nD_DD, nT_TT = DD.shape[0], TT.shape[0]
    if Y.shape == (nT_TT, nD_DD):
        Y = Y.T
    elif Y.shape != (nD_DD, nT_TT):
        #  : / , ;
        if Y.shape[0] == nD_DD and Y.shape[1] >= nT_TT:
            Y = Y[:, :nT_TT]
        elif Y.shape[1] == nT_TT and Y.shape[0] >= nD_DD:
            Y = Y[:nD_DD, :]
        elif Y.shape[0] == nT_TT and Y.shape[1] >= nD_DD:
            Y = Y.T
            if Y.shape[0] >= nD_DD and Y.shape[1] >= nT_TT:
                Y = Y[:nD_DD, :nT_TT]
        else:
            raise ValueError(f"Unexpected Y shape {Y.shape} for (nD, nT)=({nD_DD},{nT_TT})")

    # ---- Normalize graphs ----
    DD_01 = _minmax_01(DD, invert=False)
    TT_01 = _minmax_01(TT, invert=False)
    # remove self-loops
    if DD_01.shape[0] == DD_01.shape[1]:
        np.fill_diagonal(DD_01, 0.0)
    if TT_01.shape[0] == TT_01.shape[1]:
        np.fill_diagonal(TT_01, 0.0)

    # ---- Build transitions ----
    P_DD = _row_normalize(DD_01)
    P_TT = _row_normalize(TT_01)

    DT_strength = _build_DT_from_train(Y, train_pairs, dataset)
    P_DT = _row_normalize(DT_strength)
    P_TD = _row_normalize(DT_strength.T)

    # ---- Meta-path products ----
    # Use float32 to save memory
    P_DD = P_DD.astype(np.float32, copy=False)
    P_TT = P_TT.astype(np.float32, copy=False)
    P_DT = P_DT.astype(np.float32, copy=False)
    P_TD = P_TD.astype(np.float32, copy=False)

    A1 = P_DD @ P_DT           # DDT
    A2 = P_DT @ P_TT           # DTT
    A3 = P_DD @ P_DD @ P_DT    # DDDT
    A4 = P_DT @ P_TT @ P_TT    # DTTT
    A5 = P_DD @ P_DT @ P_TT    # DDTT
    A6 = P_DT @ P_TD @ P_DT    # DTDT

    X = np.stack([A1, A2, A3, A4, A5, A6], axis=-1).astype(np.float32)
    paths = ["DDT", "DTT", "DDDT", "DTTT", "DDTT", "DTDT"]

    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(out_npz_path, X=X, paths=np.array(paths, dtype=object))

    return {"X_shape": X.shape, "paths": paths, "out": out_npz_path}
"""Utilities for loading/processing heterogeneous-graph path-score features."""

