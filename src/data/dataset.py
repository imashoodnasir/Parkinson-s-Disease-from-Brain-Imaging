from typing import Dict, Any
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, RandAffined, RandGaussianNoised, EnsureChannelFirst, ToTensor

def build_transforms(augment_cfg: Dict[str, Any], train: bool):
    t = [EnsureChannelFirst(channel_dim="no_channel")]
    if train and augment_cfg.get("enable", True):
        rotate = float(augment_cfg.get("rotate_deg", 10)) * np.pi / 180.0
        translate = int(augment_cfg.get("translate_vox", 5))
        t.append(RandAffined(
            prob=0.9,
            rotate_range=(rotate, rotate, rotate),
            translate_range=(translate, translate, translate),
            padding_mode="border",
        ))
        noise_std = float(augment_cfg.get("noise_std", 0.01))
        t.append(RandGaussianNoised(prob=0.5, std=noise_std))
    t.append(ToTensor())
    return Compose(t)

class NiftiCohortDataset(Dataset):
    def __init__(self, csv_path: str, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vol = np.asarray(nib.load(row["path"]).get_fdata(), dtype=np.float32)
        y = float(row["label"])
        if self.transforms is not None:
            vol = self.transforms(vol)  # -> [1,D,H,W]
        else:
            vol = torch.from_numpy(vol[None, ...])
        return vol, torch.tensor(y, dtype=torch.float32)
