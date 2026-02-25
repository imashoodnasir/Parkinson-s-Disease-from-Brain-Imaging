import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.preprocess import preprocess_nifti

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(cfg["data"]["split_dir"])
    pre_root = Path(cfg["data"]["preprocessed_root"])
    pre_root.mkdir(parents=True, exist_ok=True)

    target_spacing = tuple(cfg["preprocess"]["target_spacing"])
    target_size = tuple(cfg["preprocess"]["target_size"])
    use_n4 = bool(cfg["preprocess"]["use_n4_bias_correction"])

    for split_csv in sorted(split_dir.glob("*_train.csv")):
        cohort = split_csv.name.replace("_train.csv", "")
        for split in ["train", "val", "test"]:
            sp = split_dir/f"{cohort}_{split}.csv"
            if not sp.exists():
                continue
            df = pd.read_csv(sp)
            out_rows = []
            pbar = tqdm(df.itertuples(index=False), total=len(df), desc=f"Preprocess {cohort}/{split}")
            for row in pbar:
                in_path = row.path
                rel = Path(in_path).name
                out_path = pre_root/cohort/row.label_name/rel
                preprocess_nifti(in_path, str(out_path), target_spacing, target_size, use_n4=use_n4)
                d = row._asdict()
                d["path"] = str(out_path)
                out_rows.append(d)
            pd.DataFrame(out_rows).to_csv(sp, index=False)

    print("Done. Updated split CSVs to use preprocessed paths.")

if __name__ == "__main__":
    main()
