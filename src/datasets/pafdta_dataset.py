# FILE:src/datasets/pafdta_dataset.py
# -*- coding: utf-8 -*-
import os
import ast
from typing import List, Tuple, Dict, Any, Optional, Callable

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import re
from src.utils.common import (
    md5_hash,
    load_json,
    load_pickle,
    normalize_key_dict,
    maybe_to_int,
    is_probable_smiles,
)


def _to_scalar_float(v):
    #   (label, extra)  / ,
    if isinstance(v, (tuple, list)) and len(v) > 0:
        v = v[0]

    # torch.Tensor
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().reshape(-1)[0])

    # numpy
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0])
    if isinstance(v, (np.floating, np.integer)):
        return float(v)

    #
    if isinstance(v, (float, int)):
        return float(v)

    #  :
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("nan", "none", ""):
            return float("nan")
        try:
            return float(s)
        except Exception:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if m:
                return float(m.group(0))
            #  , ,
            raise TypeError(f"Label y is non-numeric: {repr(v)}")

    # Fallback
    return float(v)
# -----------------   -----------------
def _pick_first_exist(paths: List[str]) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    return ""

def _resolve_data_dir(data_root: str, dataset: str) -> str:
    """ directory:./data/   ./data/<dataset>/"""
    cands = [data_root, os.path.join(data_root, dataset)]
    for d in cands:
        if os.path.isdir(d):
            #   proteins.txt   folds directory
            if os.path.isfile(os.path.join(d, "proteins.txt")) or os.path.isdir(os.path.join(d, "folds")):
                return d
    #   data_root
    return data_root

# -----------------  FILE -----------------
def _auto_find_ligands_file(data_dir: str, dataset: str) -> str:
    # 1)
    guess = _pick_first_exist([
        os.path.join(data_dir, f"ligands_{dataset}.txt"),
        os.path.join(data_dir, "ligands.txt"),
        os.path.join(data_dir, "ligands_smiles.txt"),
        os.path.join(data_dir, "ligands_can.txt"),
        os.path.join(data_dir, "ligands_iso.txt"),
    ])
    if guess:
        return guess
    # 2)  directory, "  SMILES   JSON  "
    for fname in sorted(os.listdir(data_dir)):
        if fname == "proteins.txt":
            continue
        if not (fname.endswith(".txt") or fname.endswith(".json")):
            continue
        fpath = os.path.join(data_dir, fname)
        try:
            obj = load_json(fpath)
        except Exception:
            continue
        if isinstance(obj, dict) and obj:
            vals = list(obj.values())

            def _unwrap(v):
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    for key in ("smiles", "SMILES", "can", "canonical_smiles"):
                        if key in v and isinstance(v[key], str):
                            return v[key]
                return None

            hits, check_n = 0, min(20, len(vals))
            for i in range(check_n):
                s = _unwrap(vals[i])
                if s and is_probable_smiles(s):
                    hits += 1
            if hits >= max(1, check_n // 3):
                return fpath
    return ""

# -----------------  FILE:Load ( ) -----------------
def _list_fold_candidates(data_dir: str) -> List[str]:
    fold_dir = os.path.join(data_dir, "folds")
    if not os.path.isdir(fold_dir):
        return []  #  directory,  Y Fallback
    cands: List[str] = []
    for root, _, files in os.walk(fold_dir):
        for fn in files:
            if fn.endswith(".txt") or fn.endswith(".json"):
                cands.append(os.path.join(root, fn))
    return sorted(cands)

def _parse_pairs_object(obj: Any, split: Optional[str] = None) -> List[Tuple[Any, Any]]:
    """
      [(lig, prot), ...]:
      - [[lig, prot], ...]
      - [{"lig":..,"prot":..}, ...](  drug/target/d,t  )
      - {"train":[...], "test":[...]}   {fold0:{train:[...]}}
      - {"pairs":[...]} / {"data":[...]} / {"items":[...]} / {"indices":[...]} / {"idx":[...]}
    """
    def _coerce_pair(p):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return p[0], p[1]
        if isinstance(p, dict):
            for lk in ("lig", "drug", "d"):
                for pk in ("prot", "target", "t"):
                    if lk in p and pk in p:
                        return p[lk], p[pk]
        return None

    def _parse_any(x) -> List[Tuple[Any, Any]]:
        #
        if isinstance(x, list):
            out = []
            for p in x:
                pair = _coerce_pair(p)
                if pair:
                    out.append(pair)
            if out:
                return out
        #
        if isinstance(x, dict):
            #   split
            if split and split in x and isinstance(x[split], (list, dict)):
                got = _parse_any(x[split])
                if got:
                    return got
            #
            for key in ("train", "valid", "val", "test"):
                if key in x and isinstance(x[key], (list, dict)):
                    got = _parse_any(x[key])
                    if got:
                        return got
            for key in ("pairs", "data", "items", "indices", "idx"):
                if key in x and isinstance(x[key], (list, dict)):
                    got = _parse_any(x[key])
                    if got:
                        return got
            #   value
            for v in x.values():
                got = _parse_any(v)
                if got:
                    return got
        return []

    return _parse_any(obj)

def _parse_pairs_lines(txt: str) -> List[Tuple[Any, Any]]:
    """
     :
      - "i j" / "i,j"
      - "(i, j)" / "[i, j]"
      -  ( )
    """
    pairs: List[Tuple[Any, Any]] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        #
        while len(line) > 0 and (line[0] in "[(" and line[-1] in "])"):
            line = line[1:-1].strip()
        #
        line = line.replace(",", " ").replace("\t", " ")
        toks = [t for t in line.split() if t not in (":", ";", "|")]
        if len(toks) >= 2:
            a, b = toks[0], toks[1]
            a = int(a) if isinstance(a, str) and a.isdigit() else a
            b = int(b) if isinstance(b, str) and b.isdigit() else b
            pairs.append((a, b))
    return pairs

def _read_pairs_from_file(fp: str, split: Optional[str]) -> List[Tuple[Any, Any]]:
    """
    Load FILE, :
      1) JSON
      2) Python  (ast.literal_eval)
      3)  ( / )
    """
    # JSON
    try:
        obj = load_json(fp)
        pairs = _parse_pairs_object(obj, split)
        if pairs:
            return pairs
    except Exception:
        pass
    # Python
    try:
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read()
        obj = ast.literal_eval(txt)
        pairs = _parse_pairs_object(obj, split)
        if pairs:
            return pairs
    except Exception:
        pass
    #
    try:
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read()
        pairs = _parse_pairs_lines(txt)
        if pairs:
            return pairs
    except Exception:
        pass
    return []



def _read_fold_indices(fp: str):
    with open(fp, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    return ast.literal_eval(txt)


def _pairs_from_folds_setting1(Y: Any, data_dir: str, split: str, fold_id: int = 0) -> List[Tuple[int, int]]:

    folds_dir = os.path.join(data_dir, 'folds')
    train_fp = os.path.join(folds_dir, 'train_fold_setting1.txt')
    test_fp  = os.path.join(folds_dir, 'test_fold_setting1.txt')
    if not (os.path.isfile(train_fp) and os.path.isfile(test_fp)):
        return []

    split_l = (split or '').strip().lower()
    fid = max(0, min(int(fold_id), 4))

    try:
        train_obj = _read_fold_indices(train_fp)
        test_obj  = _read_fold_indices(test_fp)
    except Exception as e:
        print(f"[PAFDTA] Warning: failed to parse folds: {e}")
        return []
    if split_l in ('train', 'training'):
        if not (isinstance(train_obj, list) and len(train_obj) >= 5 and isinstance(train_obj[0], list)):
            return []
        #  :  fold_id   4
        idx_list = []
        for j, fold in enumerate(train_obj[:5]):
            if j == fid:
                continue
            idx_list.extend(fold)
    elif split_l in ('valid', 'val', 'dev'):
        if not (isinstance(train_obj, list) and len(train_obj) >= 5 and isinstance(train_obj[0], list)):
            return []
        #  :  fold_id
        idx_list = train_obj[fid]
    elif split_l == 'test':
        if isinstance(test_obj, list) and len(test_obj) > 0 and isinstance(test_obj[0], list):
            idx_list = test_obj[fid]
        else:
            idx_list = test_obj
    else:
        return []

    if not isinstance(idx_list, list):
        return []

    # Y -> numpy
    if isinstance(Y, torch.Tensor):
        Y_np = Y.detach().cpu().numpy()
    elif isinstance(Y, np.ndarray):
        Y_np = Y
    elif isinstance(Y, (list, tuple)):
        Y_np = np.asarray(Y, dtype=float)
    else:
        # folds_setting1  ,dict
        return []

    #  ( )
    if np.isnan(Y_np).any():
        rows, cols = np.where(~np.isnan(Y_np))
    else:
        ii, jj = np.meshgrid(np.arange(Y_np.shape[0]), np.arange(Y_np.shape[1]), indexing='ij')
        rows, cols = ii.reshape(-1), jj.reshape(-1)

    n_valid = int(rows.shape[0])
    #  :fold indices   [0, n_valid)
    bad = [int(i) for i in idx_list[:50] if not (isinstance(i, int) or (isinstance(i, str) and str(i).isdigit()))]
    if bad:
        print(f"[PAFDTA] Warning: fold indices contain non-int values (show up to 50): {bad}")

    pairs: List[Tuple[int, int]] = []
    for i in idx_list:
        if isinstance(i, str) and i.isdigit():
            i = int(i)
        if not isinstance(i, int):
            continue
        if i < 0 or i >= n_valid:
            raise IndexError(f"[PAFDTA] fold index out of range: {i} (n_valid={n_valid})")
        pairs.append((int(rows[i]), int(cols[i])))

    print(f"[PAFDTA] Using folds_setting1: split={split_l}, fold_id={fid}, pairs={len(pairs)}, n_valid={n_valid}")
    return pairs
def _ordered_keys(d: Dict[Any, Any], is_str_keys: bool) -> List[Any]:
    """ : ,  JSON  ."""
    keys = list(d.keys())
    if not keys:
        return keys
    if all((isinstance(k, int) or (isinstance(k, str) and k.isdigit())) for k in keys):
        return sorted([int(k) if isinstance(k, str) and k.isdigit() else k for k in keys])
    return keys

def _build_index_maps(keys: List[Any]) -> Dict[Any, int]:
    return {k: i for i, k in enumerate(keys)}

def _coerce_to_key_or_index(x, key_is_str: bool, keys: List[Any], key2idx: Dict[Any, int]) -> Tuple[Optional[Any], Optional[int]]:
    """
      x( " /ID"," "," ") :
      -   -> (key, idx)
      -   int ->   keys[pos]
    """
    #
    if x in key2idx:
        return x, key2idx[x]
    #  /  ->
    xi = None
    if isinstance(x, str) and x.isdigit():
        xi = int(x)
    elif isinstance(x, int):
        xi = x
    if isinstance(xi, int) and 0 <= xi < len(keys):
        return keys[xi], xi
    #  :  int,
    if key_is_str is False and isinstance(x, str) and x.isdigit():
        xi = int(x)
        if xi in key2idx:
            return xi, key2idx[xi]
    return None, None

def _score_fold_file(pairs_raw: List[Tuple[Any, Any]],
                     lig_keys_order: List[Any], prot_keys_order: List[Any],
                     lig_key_is_str: bool, prot_key_is_str: bool) -> Tuple[int, List[Tuple[Any, Any]]]:
    """ ,  ( ,  )."""
    lig_key2idx = _build_index_maps(lig_keys_order)
    prot_key2idx = _build_index_maps(prot_keys_order)
    ok = 0
    bad: List[Tuple[Any, Any]] = []
    for p in pairs_raw:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            bad.append(p if isinstance(p, tuple) else (str(p), ""))
            continue
        lig_raw, prot_raw = p[0], p[1]
        lig_raw = maybe_to_int(lig_raw)
        prot_raw = maybe_to_int(prot_raw)
        lig_key, _ = _coerce_to_key_or_index(lig_raw, lig_key_is_str, lig_keys_order, lig_key2idx)
        prot_key, _ = _coerce_to_key_or_index(prot_raw, prot_key_is_str, prot_keys_order, prot_key2idx)
        if lig_key is None or prot_key is None:
            if len(bad) < 20:
                bad.append((lig_raw, prot_raw))
        else:
            ok += 1
    return ok, bad

def _select_best_fold_file(data_dir: str,
                           ligands: Dict[Any, Any], lig_key_is_str: bool,
                           proteins: Dict[Any, Any], prot_key_is_str: bool,
                           split: str) -> Tuple[Optional[str], List[Tuple[Any, Any]]]:
    """ FILE , "  ligands/proteins  " .  (None, [])."""
    cand_paths = _list_fold_candidates(data_dir)
    if not cand_paths:
        return None, []

    lig_keys_order = _ordered_keys(ligands, lig_key_is_str)
    prot_keys_order = _ordered_keys(proteins, prot_key_is_str)

    best_path = None
    best_pairs: List[Tuple[Any, Any]] = []
    best_score = -1

    for fp in cand_paths:
        pairs = _read_pairs_from_file(fp, split=split)
        if not pairs:
            continue
        score, _ = _score_fold_file(pairs, lig_keys_order, prot_keys_order, lig_key_is_str, prot_key_is_str)
        if score > best_score:
            best_score = score
            best_path = fp
            best_pairs = pairs

    if best_path is None or best_score <= 0:
        return None, []
    print(f"[PAFDTA]  FILE:{best_path} |  ={best_score} /  pairs={len(best_pairs)}")
    return best_path, best_pairs

# ----------------- Fallback:  Y  Generate  -----------------
def _pairs_from_Y(Y: Any,
                  lig_keys_order: List[Any], prot_keys_order: List[Any],
                  split: str, fold_id: int = 0, k_folds: int = 5, seed: int = 42) -> List[Tuple[int, int]]:
    """
      folds  ,  Y  / Generate  K  :
      -   Y  (list/ndarray/tensor),  (i,j)
      -   NaN,
    """
    #   numpy
    if isinstance(Y, torch.Tensor):
        Y_np = Y.detach().cpu().numpy()
    elif isinstance(Y, np.ndarray):
        Y_np = Y
    elif isinstance(Y, (list, tuple)):
        Y_np = np.array(Y, dtype=float)
    elif isinstance(Y, dict):
        #  , , Fallback
        return []

    n_lig, n_prot = Y_np.shape[0], Y_np.shape[1]
    #  : ,
    n_lig = min(n_lig, len(lig_keys_order))
    n_prot = min(n_prot, len(prot_keys_order))

    #  :  NaN
    if np.isnan(Y_np).any():
        valid = np.argwhere(~np.isnan(Y_np[:n_lig, :n_prot]))
    else:
        #
        ii, jj = np.meshgrid(np.arange(n_lig), np.arange(n_prot), indexing="ij")
        valid = np.stack([ii.reshape(-1), jj.reshape(-1)], axis=1)

    if valid.shape[0] == 0:
        return []

    #   fold_id
    rng = np.random.default_rng(seed)
    idx = rng.permutation(valid.shape[0])
    fold_sizes = [(valid.shape[0] + i) // k_folds for i in range(k_folds)]
    starts = [sum(fold_sizes[:i]) for i in range(k_folds)]
    ends = [sum(fold_sizes[:i+1]) for i in range(k_folds)]
    fid = max(0, min(fold_id, k_folds - 1))
    test_idx = idx[starts[fid]:ends[fid]]
    train_idx = np.concatenate([idx[:starts[fid]], idx[ends[fid]:]], axis=0)

    chosen = train_idx if split == "train" else test_idx
    pairs = [(int(valid[i, 0]), int(valid[i, 1])) for i in chosen.tolist()]
    print(f"[PAFDTA][Fallback]   Y  Generate {split}  :{len(pairs)}(fold_id={fid})")
    return pairs

# -----------------   proteins / ligands / Y -----------------
def _load_proteins_ligands(data_dir: str, dataset: str) -> Tuple[Dict[Any, str], Dict[Any, Any]]:
    proteins_fp = _pick_first_exist([os.path.join(data_dir, "proteins.txt")])
    if not proteins_fp:
        raise FileNotFoundError(f"[PAFDTA] Not found proteins.txt(directory:{data_dir})")

    ligands_fp = _auto_find_ligands_file(data_dir, dataset)
    if not ligands_fp:
        raise FileNotFoundError(f"[PAFDTA] Not found FILE(directory:{data_dir}).")

    proteins_raw = load_json(proteins_fp)
    ligands_raw  = load_json(ligands_fp)
    return proteins_raw, ligands_raw

def _load_Y(data_dir: str, dataset: str, ligands_raw: Optional[Dict[Any, Any]] = None, proteins_raw: Optional[Dict[Any, Any]] = None):
    """
      Y / Y.pkl / Y_<dataset>(.pkl);
      dataset==kiba, Load kiba_binding_affinity(_v2).txt  :
      -   NaN  / (  NaN)
      -   ligands/proteins  ( )
    """
    # 1)
    y_fp = _pick_first_exist([
        os.path.join(data_dir, "Y"),
        os.path.join(data_dir, "Y.pkl"),
        os.path.join(data_dir, f"Y_{dataset}"),
        os.path.join(data_dir, f"Y_{dataset}.pkl"),
        os.path.join(data_dir, f"{dataset}_Y.pkl"),  #
    ])
    if y_fp:
        return load_pickle(y_fp)

    # 2) KIBA  Fallback
    if dataset.lower() == "kiba":
        txt_fp = _pick_first_exist([
            os.path.join(data_dir, "kiba_binding_affinity_v2.txt"),
            os.path.join(data_dir, "kiba_binding_affinity.txt"),
        ])
        if not txt_fp:
            raise FileNotFoundError(
                f"[PAFDTA] Not found Y  , Not found KIBA  (kiba_binding_affinity_v2.txt)directory:{data_dir}"
            )
        mat = pd.read_csv(
            txt_fp, sep="\\t", header=None, dtype=float,
            na_values=["nan", "NaN", "NA"]
        ).values

        #   NaN  / (v2  )
        col_mask = ~np.all(np.isnan(mat), axis=0)
        row_mask = ~np.all(np.isnan(mat), axis=1)
        mat = mat[row_mask][:, col_mask]

        #   ligands/proteins  ( = , =protein)
        n_lig = n_pro = None
        try:
            if ligands_raw is not None:
                n_lig = len(ligands_raw)
            if proteins_raw is not None:
                n_pro = len(proteins_raw)
        except Exception:
            pass
        if n_lig and n_pro:
            if (mat.shape[0], mat.shape[1]) == (n_lig, n_pro):
                pass
            elif (mat.shape[0], mat.shape[1]) == (n_pro, n_lig):
                mat = mat.T
            else:
                #   (n_lig, n_pro),
                if abs(mat.shape[0]-n_lig) + abs(mat.shape[1]-n_pro) > abs(mat.shape[0]-n_pro) + abs(mat.shape[1]-n_lig):
                    mat = mat.T

        return mat.astype(np.float32)

    # 3)  ,
    raise FileNotFoundError(
        f"[PAFDTA] Not found Y  , :Y / Y.pkl / Y_{dataset} / Y_{dataset}.pkl(directory:{data_dir})"
    )

# ----------------- dataset  -----------------
class PAFDTADataset(Dataset):

    def __init__(self, dataset: str, data_root: str, feature_dir: str,
                 split: str = "train", fold_id: int = 0,
                 target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        data_root/
          davis/   kiba/
            proteins.txt
            <auto-found-ligands>         #  FILE
            Y / Y.pkl / Y_<dataset>(.pkl)   (kiba) kiba_binding_affinity_v2.txt
            folds/*   .txt/.json   #  ,  Y   K
        feature_dir/
            MD5(seq).pt
        """
        assert dataset in ["davis", "kiba"]
        self.dataset = dataset
        self.data_root = data_root
        self.data_dir = _resolve_data_dir(data_root, dataset)
        self.feature_dir = feature_dir
        self.split = split
        self.fold_id = fold_id
        self.target_transform = target_transform  # ⬅

        # ----  FILE ----
        proteins_raw, ligands_raw = _load_proteins_ligands(self.data_dir, dataset)
        #  ,  _load_Y
        proteins_norm, prot_key_is_str = normalize_key_dict(proteins_raw)
        ligands_norm, lig_key_is_str   = normalize_key_dict(ligands_raw)

        #   dict   smiles
        for k, v in list(ligands_norm.items()):
            if not isinstance(v, str) and isinstance(v, dict):
                for fld in ("smiles", "SMILES", "can", "canonical_smiles"):
                    if fld in v and isinstance(v[fld], str):
                        ligands_norm[k] = v[fld]
                        break

        # Load Y(  kiba  )
        self.Y = _load_Y(self.data_dir, dataset, ligands_norm, proteins_norm)

        #
        self.proteins, self.prot_key_is_str = proteins_norm, prot_key_is_str
        self.ligands,  self.lig_key_is_str  = ligands_norm,  lig_key_is_str

        #  (  Y  )
        self.lig_keys_order = _ordered_keys(self.ligands, self.lig_key_is_str)
        self.prot_keys_order = _ordered_keys(self.proteins, self.prot_key_is_str)
        self.lig_key2idx = {k: i for i, k in enumerate(self.lig_keys_order)}
        self.prot_key2idx = {k: i for i, k in enumerate(self.prot_keys_order)}
        best_path = None
        best_pairs = _pairs_from_folds_setting1(self.Y, self.data_dir, self.split, self.fold_id)

        if not best_pairs:
            best_path, best_pairs = _select_best_fold_file(
                self.data_dir,
                self.ligands, self.lig_key_is_str,
                self.proteins, self.prot_key_is_str,
                split=self.split
            )

        if not best_pairs:
            best_pairs = _pairs_from_Y(
                self.Y, self.lig_keys_order, self.prot_keys_order,
                split=self.split, fold_id=self.fold_id, k_folds=5, seed=42
            )
            if not best_pairs:
                raise RuntimeError(
                    "[PAFDTA]  FILE ,  Y  . ."
                )

        # ----   pairs   ----
        # ---- heterogeneous graph PathScores( ) ----
        self.hg_X = None
        self.hg_dim = 0
        hg_dir = os.path.join(self.data_dir, "kg")
        hg_npz = os.path.join(hg_dir, f"pathscores_fold{self.fold_id}.npz")
        if os.path.isfile(hg_npz):
            try:
                import numpy as _np
                _npz = _np.load(hg_npz, allow_pickle=True)
                X = _npz["X"]  # [nD, nT, 6]
                self.hg_X = X.astype("float32")
                self.hg_dim = self.hg_X.shape[-1]
                print(f"[PAFDTA] Loaded heterogeneous graph path scores: {hg_npz}, dim={self.hg_dim}")
            except Exception as e:
                print(f"[PAFDTA] Warning: failed to load heterogeneous graph path scores: {e}")
        else:
            print(f"[PAFDTA] heterogeneous graph path scores not found (optional): {hg_npz}")

        samples: List[Tuple[Any, Any, int, int]] = []
        bad_examples = []
        for p in best_pairs:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            lig_raw, prot_raw = p[0], p[1]
            lig_raw = maybe_to_int(lig_raw)
            prot_raw = maybe_to_int(prot_raw)
            lig_key, lig_idx = _coerce_to_key_or_index(lig_raw, self.lig_key_is_str, self.lig_keys_order, self.lig_key2idx)
            prot_key, prot_idx = _coerce_to_key_or_index(prot_raw, self.prot_key_is_str, self.prot_keys_order, self.prot_key2idx)
            if lig_key is None or prot_key is None:
                if len(bad_examples) < 20:
                    bad_examples.append((lig_raw, prot_raw))
                continue
            samples.append((lig_key, prot_key, lig_idx, prot_idx))

        if not samples:
            msg = [
                "[PAFDTA]  .",
                f"- data_dir={self.data_dir}",
                f"-   pairs  ={len(best_pairs)}",
            ]
            if bad_examples:
                msg.append(f"-  ( 20 ):{bad_examples}")
            raise RuntimeError("\n".join(msg))

        print(f"[PAFDTA]  ={len(samples)}(split={self.split})")
        self.samples = samples

        # Y
        self._y_mode = "auto"
        if isinstance(self.Y, (list, tuple, np.ndarray, torch.Tensor)):
            self._y_mode = "matrix"
        elif isinstance(self.Y, dict):
            self._y_mode = "dict"

    def __len__(self):
        return len(self.samples)

    def _lookup_label(self, lig_key, prot_key, lig_idx, prot_idx) -> float:
        """  Y  : /  (idx, idx),  dict  ."""
        if self._y_mode == "matrix":
            if isinstance(self.Y, torch.Tensor):
                return float(self.Y[lig_idx, prot_idx].item())
            if isinstance(self.Y, np.ndarray):
                return float(self.Y[lig_idx, prot_idx])
            return float(self.Y[lig_idx][prot_idx])
        # dict
        y0 = self.Y.get(str(lig_key), self.Y.get(lig_key))
        if isinstance(y0, dict):
            #  , ,
            if str(prot_key) in y0:
                return float(y0[str(prot_key)])
            if prot_key in y0:
                return float(y0[prot_key])
            if isinstance(prot_idx, int) and prot_idx in y0:
                return float(y0[prot_idx])
            raise KeyError(f"Y  Not found : lig={lig_key} prot={prot_key}")
        if isinstance(y0, (list, tuple)):
            return float(y0[prot_idx])
        return float(self.Y[lig_key][prot_key])

    def __getitem__(self, idx):
        lig_key, prot_key, lig_idx, prot_idx = self.samples[idx]
        smiles = self.ligands[lig_key]
        seq    = self.proteins[prot_key]

        #
        seq_id = md5_hash(seq)
        feat_path = os.path.join(self.feature_dir, f"{seq_id}.pt")
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(
                f"[PAFDTA]  protein FILE:{feat_path}\n"
                f"  scripts/extract_esm_features.py Generate."
            )
        prot_feat = torch.load(feat_path, map_location="cpu")
        prot_feat = prot_feat.float()  #   float32,  Linear

        #
        y_val = float(self._lookup_label(lig_key, prot_key, lig_idx, prot_idx))
        y = torch.tensor(y_val, dtype=torch.float32)

        #  (  train_esm.py  )
        if self.target_transform is not None:
            y = self.target_transform(y)

        hg = None
        if self.hg_X is not None:
            hg = torch.from_numpy(self.hg_X[lig_idx, prot_idx, :])
        if hg is None:
            return smiles, prot_feat, y
        else:
            return smiles, prot_feat, y, hg



def pad_collate_fn(batch):
    """
     protein ; heterogeneous graph .
     :
      -  heterogeneous graph: (drug_list, feats[B,L,D], mask[B,L], y[B])
      -  heterogeneous graph: (drug_list, feats[B,L,D], mask[B,L], y[B], hg[B,K])
    """
    # unpack
    has_hg = (len(batch[0]) == 4)
    if has_hg:
        drugs, feats, ys, hgs = zip(*batch)
    else:
        drugs, feats, ys = zip(*batch)
        hgs = None

    lengths = [f.shape[0] for f in feats]
    max_len = max(lengths)
    feat_dim = feats[0].shape[1]
    dtype = feats[0].dtype

    padded = torch.zeros(len(feats), max_len, feat_dim, dtype=dtype)
    mask = torch.zeros(len(feats), max_len, dtype=torch.bool)
    for i, f in enumerate(feats):
        L = f.shape[0]
        padded[i, :L, :] = f
        mask[i, :L] = True

    #   ys   (label, extra)  ;  label
# ys   tensor / tuple / numpy / str
    ys = [_to_scalar_float(yy) for yy in ys]
    #  :  NaN  (  batch  )
    ys = [0.0 if (isinstance(v, float) and (v != v)) else v for v in ys]  # NaN!=NaN
    y = torch.tensor(ys, dtype=torch.float32)

    if has_hg:
        hg_mat = torch.stack(hgs, dim=0).float()
        return list(drugs), padded, mask, y, hg_mat
    else:
        return list(drugs), padded, mask, y

"""Dataset utilities for PAFDTA.

Provides a dataset loader (Davis/KIBA) with:
- SMILES strings for drugs
- Precomputed per-token protein features stored as .pt tensors (keyed by md5(sequence))
- Optional heterogeneous-graph path-score features."""

