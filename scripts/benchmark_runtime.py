import argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.dataset import NiftiCohortDataset, build_transforms
from src.models.model import ParkinsonFoundationNet

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
    ap.add_argument("--cohort", required=True, choices=["PPMI","NEUROCON","TaoWu"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(cfg["data"]["split_dir"])
    test_csv = split_dir/f"{args.cohort}_test.csv"

    t_eval = build_transforms(cfg["augment"], train=False)
    ds = NiftiCohortDataset(str(test_csv), transforms=t_eval)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    x, _ = next(iter(dl))
    x = x.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(args.iters):
            t0 = time.time()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000.0)

    latency_ms = float(np.mean(times))
    throughput = float(1000.0 / latency_ms)
    mem_gb = float(torch.cuda.max_memory_allocated()/1e9) if device.type=="cuda" else 0.0
    print({"latency_ms": latency_ms, "throughput_vol_s": throughput, "gpu_mem_gb": mem_gb})

if __name__ == "__main__":
    main()
