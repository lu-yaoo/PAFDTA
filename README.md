# PAFDTA 

This repository provides a cleaned and unified implementation of **PAFDTA** for drug–target affinity prediction.

Key points:
- Single training entrypoint: `PAFDTA_train.py`
- datasets (Davis/KIBA) are supported
- Protein inputs are precomputed per-token features stored as `.pt` tensors
- Optional heterogeneous-graph path-score features are supported

## Environment

Recommended Python: 3.7

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Layout

Expected dataset layout:
```
data/
  davis/
    proteins.txt
    ligands_*.txt (or json dict)
    Y (or Y.pkl / other supported formats)
    folds/
      train_fold_setting1.txt
      test_fold_setting1.txt
    kg/ (optional)
      pathscores_fold0.npz
      pathscores_fold1.npz
      ...
    ...
```

## Protein Token Features
Extract protein_features.rar to the current directory.
Training uses precomputed per-token protein features stored as `.pt` files.
Each protein sequence is mapped to a file by:
- `md5(sequence)` -> `<md5>.pt`

Expected tensor shape: `[L, D]`.

Example:
```
protein_features/
  davis/
    <md5_1>.pt
    <md5_2>.pt
    ...
```

## Training

Example (Davis, fold 0):
```bash
python PAFDTA_train.py   --dataset davis   --data_root ./data   --feature_dir ./protein_features/davis   --config ./configs/PAFDTA_train.yaml   --fold 0   --save_path ./checkpoints/pafdta_davis_fold0.pth
```

## License

For academic submission/review use.
