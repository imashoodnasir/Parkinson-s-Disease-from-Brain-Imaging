import argparse, copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import NiftiCohortDataset, build_transforms
from src.models.model import ParkinsonFoundationNet
from src.utils.train import run_epoch, CosineWithWarmup, save_json

def build_model(cfg, disable_refine: bool):
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
    if disable_refine:
        model.refine = torch.nn.Identity()
    return model

def train_one(cfg, cohort, use_aug: bool, use_refine: bool):
    split_dir = Path(cfg["data"]["split_dir"])
    tr_csv = split_dir/f"{cohort}_train.csv"
    va_csv = split_dir/f"{cohort}_val.csv"
    te_csv = split_dir/f"{cohort}_test.csv"

    aug = copy.deepcopy(cfg["augment"])
    aug["enable"] = bool(use_aug)

    t_train = build_transforms(aug, train=True)
    t_eval  = build_transforms(aug, train=False)

    ds_tr = NiftiCohortDataset(str(tr_csv), transforms=t_train)
    ds_va = NiftiCohortDataset(str(va_csv), transforms=t_eval)
    ds_te = NiftiCohortDataset(str(te_csv), transforms=t_eval)

    tr = cfg["train"]
    dl_tr = DataLoader(ds_tr, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"], pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, disable_refine=not use_refine).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = CosineWithWarmup(opt, warmup_epochs=int(tr.get("warmup_epochs", 1)), max_epochs=int(tr["max_epochs"]))

    best_auc, best_state, patience = -1.0, None, 0
    for _ in range(1, int(tr["max_epochs"])+1):
        run_epoch(model, dl_tr, opt, device, amp=bool(tr["amp"]), grad_clip=float(tr["grad_clip"]), train=True)
        val_metrics, _, _ = run_epoch(model, dl_va, opt, device, amp=bool(tr["amp"]), grad_clip=float(tr["grad_clip"]), train=False)
        sched.step()
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(tr["early_stop_patience"]):
            break

    model.load_state_dict(best_state)
    te_metrics, _, _ = run_epoch(model, dl_te, opt, device, amp=False, grad_clip=0.0, train=False)
    return te_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cohort", required=True, choices=["PPMI","NEUROCON","TaoWu"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    variants = [
        ("Backbone_only", False, False),
        ("+Augmentation", True, False),
        ("+Refinement", False, True),
        ("Full", True, True),
    ]

    out = {}
    out_dir = Path("results")/f"ablation_components_{args.cohort}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, aug, ref in variants:
        m = train_one(cfg, args.cohort, aug, ref)
        out[name] = m
        print(name, m)

    save_json(str(out_dir/"metrics.json"), out)

if __name__ == "__main__":
    main()
