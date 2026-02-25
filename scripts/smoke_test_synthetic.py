import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.model import ParkinsonFoundationNet
from src.utils.train import run_epoch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    B = 8
    x = torch.randn(B, 1, 128, 128, 128)
    y = torch.randint(0, 2, (B,), dtype=torch.float32)
    dl = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=True)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    metrics, _, _ = run_epoch(model, dl, opt, device, amp=False, grad_clip=1.0, train=True)
    print("OK:", metrics)

if __name__ == "__main__":
    main()
