# Parkinson’s Disease Volumetric Foundation Classifier — Reference Implementation

This repo is a Python implementation of the workflow described in the paper:
- MRI preprocessing (resample, normalize, crop/pad; optional N4 bias correction)
- Patch embedding + hierarchical encoder
- Discriminative feature refinement (channel gating + normalization)
- PD vs HC classification
- Within-dataset and cross-dataset evaluation
- Ablations (components + depth)
- Complexity and runtime benchmarking

> Skull stripping and MNI registration are tool-dependent. This code keeps the pipeline consistent and provides clean hooks, but assumes your volumes are already brain-extracted/aligned or that you have preprocessed externally.

## Install
```bash
pip install -r requirements.txt
```

## Data layout (raw)
```
data/
  PPMI/PD/*.nii.gz    PPMI/HC/*.nii.gz
  NEUROCON/PD/*.nii.gz  NEUROCON/HC/*.nii.gz
  TaoWu/PD/*.nii.gz     TaoWu/HC/*.nii.gz
```

## 1) Build manifests + splits
```bash
python scripts/make_manifests.py --data_root data --out_dir manifests
python scripts/make_splits.py --manifest_dir manifests --out_dir splits --seed 42
```

## 2) Offline preprocessing (updates split CSV paths to preprocessed files)
```bash
python scripts/preprocess_all.py --config configs/config.yaml
```

## 3) Train within-dataset
```bash
python scripts/train_within.py --config configs/config.yaml --cohort PPMI
```

## 4) Evaluate cross-dataset
```bash
python scripts/eval_cross.py --config configs/config.yaml --source PPMI --ckpt results/within_PPMI/ckpt_best.pt
```

## 5) Ablations
```bash
python scripts/ablate_components.py --config configs/config.yaml --cohort PPMI
python scripts/ablate_depth.py --config configs/config.yaml --cohort PPMI --depths 2 3 4 5
```

## 6) Runtime + complexity
```bash
python scripts/profile_model.py --config configs/config.yaml
python scripts/benchmark_runtime.py --config configs/config.yaml --cohort PPMI --ckpt results/within_PPMI/ckpt_best.pt
```

## 7) Synthetic smoke test (no MRI needed)
```bash
python scripts/smoke_test_synthetic.py --config configs/config.yaml
```
