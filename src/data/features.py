from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def add_leak_free_features(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.drop(columns=["_orig_idx"], errors="ignore")
        .assign(_row_id=np.arange(len(df)))
        .sort_values(["step", "_row_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    # 1. Basic transaction & time features
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    out["hour_of_day"] = out["step"] % 24
    out["type_code"] = out["type"].astype("category").cat.codes

    # 2. Time since previous transaction for origin account (strictly past)
    prev_step = out.groupby("nameOrig", sort=False)["step"].shift(1)
    out["time_since_prev_orig"] = (out["step"] - prev_step).fillna(-1)
    out["is_first_orig_txn"] = prev_step.isna().astype(np.int8)

    # 3. Past amount statistics for origin account (strictly prior rows)
    s_amt = out.groupby("nameOrig", sort=False)["amount"].shift(1)
    prior_sum = s_amt.fillna(0.0).groupby(out["nameOrig"], sort=False).cumsum()
    prior_cnt = s_amt.notna().groupby(out["nameOrig"], sort=False).cumsum()

    out["orig_prev_amt_mean"] = np.where(prior_cnt > 0, prior_sum / prior_cnt, np.nan)
    out["amt_to_orig_prev_mean"] = np.where(
        out["orig_prev_amt_mean"] > 0,
        out["amount"] / out["orig_prev_amt_mean"],
        -1.0,
    )

    # 4. Prior pair count (origin -> destination before current row)
    out["orig_dest_prior_count"] = out.groupby(
        ["nameOrig", "nameDest"], sort=False
    ).cumcount()

    # 5. Prior destination incoming count (before current row)
    out["dest_prior_in_count"] = out.groupby("nameDest", sort=False).cumcount()

    # 6. Prior origin transaction count (before current row)
    out["orig_prior_txn_count"] = out.groupby("nameOrig", sort=False).cumcount()

    return out.drop(columns=["_row_id"])


def build_features(processed_dir: Path = REPO_ROOT / "data" / "processed") -> None:
    for split_name in ["train", "val", "test"]:
        in_file = processed_dir / f"{split_name}_raw_clean.parquet"
        out_file = processed_dir / f"{split_name}_features.parquet"
        split_df = pd.read_parquet(in_file)
        feat_df = add_leak_free_features(split_df)
        feat_df.to_parquet(out_file, index=False)
        non_null_prev = int(feat_df["orig_prev_amt_mean"].notna().sum())
        print(
            f"Saved {out_file} | shape={feat_df.shape} | "
            f"non_null orig_prev_amt_mean={non_null_prev}"
        )


if __name__ == "__main__":
    build_features()