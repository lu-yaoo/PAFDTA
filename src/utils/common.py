# FILE:src/utils/common.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def ensure_project_root_on_path():
    """ directory  sys.path  (  scripts  )"""
    here = Path(__file__).resolve()
    root = here.parents[2] if (len(here.parents) >= 3) else here.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def md5_hash(text: str) -> str:
    """Generate  MD5  , FILE """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str):
    """Ensuredirectory """
    os.makedirs(path, exist_ok=True)


def load_json(fp: str) -> Dict:
    """Load JSON FILE"""
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode_bytes(obj):
    """  bytes   str(utf-8  ,  latin1),  pickle."""
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("latin1", errors="ignore")
    if isinstance(obj, dict):
        return {_decode_bytes(k): _decode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_decode_bytes(x) for x in obj)
    return obj


def load_pickle(fp: str) -> Any:
    """
    Load pickle:
    1)  ;
    2)   UnicodeDecodeError(  Python2   pkl),  encoding='latin1'  ;
    3)   numpy.load(..., allow_pickle=True) Fallback;
    4)   bytes  / , .
    """
    #   1:
    try:
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        return _decode_bytes(obj)
    except UnicodeDecodeError:
        pass
    except Exception:
        pass

    #   2:latin1
    try:
        with open(fp, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        return _decode_bytes(obj)
    except Exception:
        pass

    #   3:numpy Fallback
    try:
        import numpy as np
        with open(fp, "rb") as f:
            obj = np.load(f, allow_pickle=True)
        return _decode_bytes(obj)
    except Exception as e:
        raise e


def normalize_key_dict(d: Dict) -> Tuple[Dict, bool]:
    """ ,  int  ;  str  .
     : ( ,  )
    """
    keys = list(d.keys())
    if all(isinstance(k, int) for k in keys):
        return d, False
    if all(isinstance(k, str) and k.isdigit() for k in keys):
        return {int(k): d[k] for k in keys}, False
    return d, True


def maybe_to_int(x):
    """  int, """
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return x


# ---------   SMILES   ----------
_SMILES_CHARS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#=+-[]()@\\/%.:*"))
_smiles_token_pattern = re.compile(r"^[A-Za-z0-9#=\+\-\[\]\(\)@\\/%.:\*]+$")

def is_probable_smiles(s: str) -> bool:
    """  SMILES(  ligands FILE)."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) < 5:
        return False
    if any(ch not in _SMILES_CHARS for ch in s):
        return False
    if not _smiles_token_pattern.match(s):
        return False
    #  protein  SMILES(protein  20  , )
    if s.isupper() and len(s) > 50:
        return False
    return True


def to_device(batch, device):
    """  batch   Tensor   device  """
    out = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            out.append(item.to(device, non_blocking=True))
        else:
            out.append(item)
    return tuple(out)
"""Common utilities used across the PAFDTA codebase."""

