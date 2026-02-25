import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for mpath in mdir.glob("*.csv"):
        df = pd.read_csv(mpath)
        df["label"] = (df["label_name"] == "PD").astype(int)
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["label"])
        val_rel = args.val_size / (1.0 - args.test_size)
        train_df, val_df = train_test_split(train_df, test_size=val_rel, random_state=args.seed, stratify=train_df["label"])

        cohort = mpath.stem
        train_df.to_csv(out/f"{cohort}_train.csv", index=False)
        val_df.to_csv(out/f"{cohort}_val.csv", index=False)
        test_df.to_csv(out/f"{cohort}_test.csv", index=False)
        print(f"{cohort}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

if __name__ == "__main__":
    main()
