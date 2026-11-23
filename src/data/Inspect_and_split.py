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


def load_and_audit(csv_path: str) -> pd.DataFrame:
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


def clean_and_split(
    df: pd.DataFrame,
    out_dir: str,
    train_max: int | None = None,
    val_max: int | None = None,
) -> dict[str, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Drop rows with missing target so supervised training doesn't hit NaN labels
    df = df.dropna(subset=["isFraud"]).copy()

    # Deterministic sort by step, preserving original row order as tie-breaker
    df = df.assign(_orig_idx=np.arange(len(df))).sort_values(
        ["step", "_orig_idx"], kind="mergesort"
    )

    cols_to_drop = [c for c in LEAKY_COLS if c in df.columns]
    cleaned = df.drop(columns=cols_to_drop).reset_index(drop=True)

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
        part.to_parquet(out_path / f"{name}_raw_clean.parquet", index=False)

    return splits


if __name__ == "__main__":
    csv_file = REPO_ROOT / "data" / "raw" / "paysim dataset.csv"
    raw_df = load_and_audit(csv_file)
    clean_and_split(raw_df, REPO_ROOT / "data" / "processed", train_max=5, val_max=6)