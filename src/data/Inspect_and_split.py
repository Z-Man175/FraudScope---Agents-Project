#NEW: needed for the command-line options at the bottom of the file
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

LEAKY_COLS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

#NEW: columns a synthetic scenarios file must contain to be merged into PaySim
REQUIRED_PAYSIM_COLS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
}

#NEW: default location of the synthetic scenarios file (used by the --synthetic option)
DEFAULT_SYNTHETIC_PATH = REPO_ROOT / "data" / "raw" / "synthetic_scenarios.parquet"

#NEW: warn when a split has fewer fraud rows than this
MIN_FRAUD_PER_SPLIT = 100


#CHANGED: the type hint now accepts a Path as well as a str (the logic is unchanged)
def load_and_audit(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print("=== Dataset Shape & Columns ===")
    print(df.shape)
    print(list(df.columns))

    print("\n=== Missing Values ===")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\n=== Target & Type Distribution ===")
    print("Fraud rate:", df["isFraud"].mean())
    print(pd.crosstab(df["type"], df["isFraud"], margins=True))

    print("\n=== Account Activity Summary ===")
    orig_counts = df.groupby("nameOrig").size()
    dest_counts = df.groupby("nameDest").size()
    print(
        "Origin txns/account quantiles:",
        orig_counts.quantile([0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).to_dict(),
    )
    print(
        "Dest txns/account quantiles:",
        dest_counts.quantile([0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).to_dict(),
    )
    print("Step min/max:", df["step"].min(), df["step"].max())

    return df


#NEW: helper that builds stable IDs like PS_0000001 from a row position (vectorised, fast on 6M rows)
def _make_ids(prefix: str, n: int) -> pd.Series:
    return prefix + pd.Series(np.arange(n)).astype(str).str.zfill(7)


#CHANGED: out_dir also accepts a Path, and there is a new parameter extra_transactions_path
#(path of a synthetic scenarios file to merge before the split; None means no merge)
def clean_and_split(
    df: pd.DataFrame,
    out_dir: str | Path,
    train_max: int | None = None,
    val_max: int | None = None,
    extra_transactions_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    #NEW: every row gets a unique txn_id here, before the synthetic merge and before any sorting
    #PaySim rows get PS_<csv row position>, so IDs stay the same even if the synthetic file changes
    #A "source" column marks where each row came from
    df = df.assign(txn_id=_make_ids("PS_", len(df)).values, source="paysim")

    # Drop rows with missing target so supervised training doesn't hit NaN labels
    df = df.dropna(subset=["isFraud"]).copy()

    #NEW: optional synthetic scenario merge. It only happens when extra_transactions_path is given
    #(command-line flag --synthetic), so nobody gets synthetic rows by accident
    if extra_transactions_path is not None:
        synth_path = Path(extra_transactions_path)
        if not synth_path.exists():
            raise FileNotFoundError(f"Synthetic transactions file not found: {synth_path}")
        suffix = synth_path.suffix.lower()
        if suffix == ".parquet":
            extra_df = pd.read_parquet(synth_path)
        elif suffix == ".csv":
            extra_df = pd.read_csv(synth_path)
        else:
            raise ValueError(
                f"Unsupported synthetic transactions format '{suffix}'; expected .parquet or .csv"
            )

        missing_cols = REQUIRED_PAYSIM_COLS - set(extra_df.columns)
        if missing_cols:
            raise ValueError(
                f"Synthetic transactions file {synth_path} is missing required columns: "
                f"{sorted(missing_cols)}"
            )

        #NEW: synthetic rows get their own stable IDs (SYN_<file row position>),
        #assigned before dropna so they only change if the synthetic file itself changes
        extra_df = extra_df.assign(
            txn_id=_make_ids("SYN_", len(extra_df)).values, source="synthetic"
        )
        extra_df = extra_df.dropna(subset=["isFraud"]).copy()
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"Merged {len(extra_df)} synthetic transaction rows from {synth_path}")

    #CHANGED: reformatted, and reset_index(drop=True) now happens right after the sort
    #(before, it happened after dropping the leaky columns). The result is the same
    # Deterministic sort by step, preserving original row order as tie-breaker
    df = (
        df.assign(_orig_idx=np.arange(len(df)))
        .sort_values(["step", "_orig_idx"], kind="mergesort")
        .reset_index(drop=True)
    )

    #CHANGED: the helper column _orig_idx is now dropped here too, so it is no longer saved in the
    #parquet files (before, features.py had to remove it later)
    cols_to_drop = [c for c in LEAKY_COLS if c in df.columns] + ["_orig_idx"]
    cleaned = df.drop(columns=cols_to_drop)

    #NEW: safety check, txn_id values must be unique
    if cleaned["txn_id"].duplicated().any():
        raise ValueError("txn_id values are not unique")

    steps = sorted(cleaned["step"].unique())
    if len(steps) < 3:
        raise ValueError(f"Need at least 3 distinct steps to split, got {steps}")

    # Pick sensible boundaries if none were explicitly provided
    if train_max is None or val_max is None:
        if cleaned["step"].max() >= 700:
            # Default PaySim boundaries when full timeline is present
            train_max = 480 if train_max is None else train_max
            val_max = 600 if val_max is None else val_max
        else:
            # Fallback for smaller/subset step ranges (e.g. 1..8 -> train<=5, val==6, test>=7)
            idx_train = max(0, int(len(steps) * 0.6) - 1)
            idx_val = max(idx_train + 1, int(len(steps) * 0.8) - 1)
            idx_val = min(idx_val, len(steps) - 2)
            train_max = int(steps[idx_train]) if train_max is None else train_max
            val_max = int(steps[idx_val]) if val_max is None else val_max

    splits = {
        "train": cleaned[cleaned["step"] <= train_max].copy(),
        "val": cleaned[
            (cleaned["step"] > train_max) & (cleaned["step"] <= val_max)
        ].copy(),
        "test": cleaned[cleaned["step"] > val_max].copy(),
    }

    print(f"\nUsing boundaries: train <= {train_max}, val <= {val_max}, test > {val_max}")
    print("=== Split Fraud Summary ===")
    for name, part in splits.items():
        n_rows = len(part)
        n_fraud = int(part["isFraud"].sum())
        rate = (n_fraud / n_rows * 100) if n_rows else 0.0
        s_min = part["step"].min() if n_rows else np.nan
        s_max = part["step"].max() if n_rows else np.nan
        print(
            f"{name:5s}: rows={n_rows:9d} | fraud={n_fraud:5d} ({rate:.4f}%) | "
            f"steps={s_min}..{s_max}"
        )
        #NEW: warning when a period has too little fraud to evaluate on
        if n_fraud < MIN_FRAUD_PER_SPLIT:
            print(
                f"  WARNING: {name} has only {n_fraud} fraud rows "
                f"(< {MIN_FRAUD_PER_SPLIT}); consider moving the step boundaries."
            )
        part.to_parquet(out_path / f"{name}_raw_clean.parquet", index=False)

    return splits


#NEW: command-line options. Before, the paths and boundaries were hard-coded at the bottom
#Running with no options uses PaySim only and the default boundaries (train <= 480, val <= 600)
#Example with synthetic data:  python Inspect_and_split.py --synthetic
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit, clean and time-split PaySim.")
    parser.add_argument(
        "--csv", type=Path, default=REPO_ROOT / "data" / "raw" / "paysim dataset.csv"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "processed"
    )
    parser.add_argument("--train-max", type=int, default=None)
    parser.add_argument("--val-max", type=int, default=None)
    parser.add_argument(
        "--synthetic",
        nargs="?",
        const=DEFAULT_SYNTHETIC_PATH,
        default=None,
        type=Path,
        help="Merge synthetic scenarios before the split. Optional path; "
        "defaults to data/raw/synthetic_scenarios.parquet.",
    )
    return parser.parse_args()


#CHANGED: the old version hard-coded train_max=5, val_max=6 (a toy split that would make train
#just steps 1-5 on the full dataset). Now the values come from parse_args(), and by default the
#480/600 boundaries are used
if __name__ == "__main__":
    args = parse_args()
    raw_df = load_and_audit(args.csv)
    clean_and_split(
        raw_df,
        args.out_dir,
        train_max=args.train_max,
        val_max=args.val_max,
        extra_transactions_path=args.synthetic,
    )