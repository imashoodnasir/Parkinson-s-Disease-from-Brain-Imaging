import argparse
import torch
from src.utils.config import load_config
from src.models.model import ParkinsonFoundationNet

def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    m = cfg["model"]
    model = ParkinsonFoundationNet(
        in_ch=m["in_channels"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        depth_L=m["depth_L"],
        blocks_per_stage=m["blocks_per_stage"],
        encoder_dropout=m["encoder_dropout"],
        refinement_reduction=m["refinement"]["reduction_ratio"],
        refinement_zscore=m["refinement"]["zscore_on_refined"],
        clf_hidden=m["classifier"]["hidden_dim"],
        clf_dropout=m["classifier"]["dropout"],
    )
    print({"params_M": float(count_params_m(model))})

if __name__ == "__main__":
    main()
