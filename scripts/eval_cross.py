import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.dataset import NiftiCohortDataset, build_transforms
from src.models.model import ParkinsonFoundationNet
from src.utils.train import run_epoch, save_json

def build_model(cfg):
    m = cfg["model"]
    return ParkinsonFoundationNet(
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, choices=["PPMI","NEUROCON","TaoWu"])
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(cfg["data"]["split_dir"])
    tr = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    t_eval = build_transforms(cfg["augment"], train=False)
    dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    targets = [c for c in ["PPMI","NEUROCON","TaoWu"] if c != args.source]
    out = {}
    for tgt in targets:
        test_csv = split_dir/f"{tgt}_test.csv"
        ds = NiftiCohortDataset(str(test_csv), transforms=t_eval)
        dl = DataLoader(ds, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], pin_memory=True)
        metrics, _, _ = run_epoch(model, dl, dummy_opt, device, amp=False, grad_clip=0.0, train=False)
        out[f"{args.source}->{tgt}"] = metrics
        print(args.source, "->", tgt, metrics)

    out_dir = Path("results")/f"cross_from_{args.source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(out_dir/"metrics.json"), out)

if __name__ == "__main__":
    main()
