import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import NiftiCohortDataset, build_transforms
from src.models.model import ParkinsonFoundationNet
from src.utils.train import run_epoch, CosineWithWarmup, save_json
from src.utils.plots import plot_roc, plot_confusion

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
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    split_dir = Path(cfg["data"]["split_dir"])
    train_csv = split_dir/f"{args.cohort}_train.csv"
    val_csv   = split_dir/f"{args.cohort}_val.csv"
    test_csv  = split_dir/f"{args.cohort}_test.csv"

    t_train = build_transforms(cfg["augment"], train=True)
    t_eval  = build_transforms(cfg["augment"], train=False)

    ds_train = NiftiCohortDataset(str(train_csv), transforms=t_train)
    ds_val   = NiftiCohortDataset(str(val_csv), transforms=t_eval)
    ds_test  = NiftiCohortDataset(str(test_csv), transforms=t_eval)

    tr = cfg["train"]
    dl_train = DataLoader(ds_train, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"], pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = CosineWithWarmup(opt, warmup_epochs=int(tr.get("warmup_epochs", 1)), max_epochs=int(tr["max_epochs"]))

    out_dir = Path("results")/f"within_{args.cohort}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir/"ckpt_best.pt"

    best_auc = -1.0
    patience = 0

    for epoch in range(1, int(tr["max_epochs"])+1):
        train_metrics, _, _ = run_epoch(model, dl_train, opt, device, amp=bool(tr["amp"]), grad_clip=float(tr["grad_clip"]), train=True)
        val_metrics, yv, pv = run_epoch(model, dl_val, opt, device, amp=bool(tr["amp"]), grad_clip=float(tr["grad_clip"]), train=False)
        sched.step()

        print({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
            patience = 0
        else:
            patience += 1

        if patience >= int(tr["early_stop_patience"]):
            print(f"Early stopping at epoch {epoch}. Best val AUC={best_auc:.4f}")
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, yt, pt = run_epoch(model, dl_test, opt, device, amp=False, grad_clip=0.0, train=False)
    save_json(str(out_dir/"metrics.json"), {"best_val_auc": best_auc, "test": test_metrics})

    plot_roc(yt, pt, str(out_dir/"roc.png"), title=f"{args.cohort} ROC")
    ypred = (pt >= 0.5).astype(int)
    plot_confusion(yt, ypred, str(out_dir/"confusion.png"), title=f"{args.cohort} Confusion Matrix")

    print(f"Saved: {out_dir}")

if __name__ == "__main__":
    main()
