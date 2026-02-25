from typing import Dict, Any
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import compute_binary_metrics

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = max(0, warmup_epochs)
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            return [base_lr * epoch / self.warmup_epochs for base_lr in self.base_lrs]
        t = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        return [base_lr * 0.5 * (1.0 + np.cos(np.pi * t)) for base_lr in self.base_lrs]

def run_epoch(model, loader, optimizer, device, amp: bool, grad_clip: float, train: bool):
    model.train(train)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    all_y, all_p = [], []
    total_loss = 0.0

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)

        if train:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        prob = torch.sigmoid(logit).detach().cpu().numpy()
        all_p.append(prob)
        all_y.append(y.detach().cpu().numpy())
        total_loss += float(loss.item()) * x.size(0)
        pbar.set_postfix(loss=float(loss.item()))

    y_true = np.concatenate(all_y).astype(int)
    y_prob = np.concatenate(all_p)
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics, y_true, y_prob

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
