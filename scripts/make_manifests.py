import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cohort_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        cohort = cohort_dir.name
        rows = []
        for cls in ["PD", "HC"]:
            cls_dir = cohort_dir/cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.rglob("*.nii*"):
                rows.append({"subject_id": f.stem, "path": str(f.resolve()), "label_name": cls})
        if rows:
            pd.DataFrame(rows).to_csv(out_dir/f"{cohort}.csv", index=False)
            print(f"Wrote {cohort}: {len(rows)} samples")

if __name__ == "__main__":
    main()
