from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

#NEW: fixed type -> code mapping. Before, codes came from astype("category"), so they depended on
#which types were present in the data. Unknown types map to -1
TYPE_MAP = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

#NEW: windows (in steps) for the rolling count features
ROLLING_WINDOWS = (1, 6, 24)


#NEW: for each row, count earlier rows with the same key whose step is within `window` steps
#(same step included). The current row is never counted, so this is leak-free
#Requires df to be sorted by step with a fresh 0..n-1 index (add_leak_free_features guarantees this)
#Vectorised so it stays fast on 6M rows; checked against a brute-force version
def prior_count_in_window(df: pd.DataFrame, key_col: str, window: int) -> np.ndarray:
    codes = pd.factorize(df[key_col])[0].astype(np.int64)
    step = df["step"].to_numpy(np.int64)
    big = 100_000
    if step.max() + window >= big:
        raise ValueError("step values too large for prior_count_in_window")
    #Order rows by (key, step, original position) so each key's rows are contiguous
    order = np.lexsort((np.arange(len(df)), step, codes))
    key_sorted = codes[order] * big + step[order]
    #First position (within the same key) whose step >= current step - window
    lo = np.searchsorted(key_sorted, key_sorted - window, side="left")
    out = np.empty(len(df), dtype=np.int64)
    out[order] = np.arange(len(df)) - lo
    return out


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
    #CHANGED: use the fixed TYPE_MAP instead of astype("category").cat.codes
    out["type_code"] = out["type"].map(TYPE_MAP).fillna(-1).astype(np.int8)

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

    #NEW: 7. Rolling counts over the last 1, 6 and 24 steps: how many earlier transactions the
    #origin account sent (orig_cnt_*) and the destination account received (dest_in_cnt_*).
    for w in ROLLING_WINDOWS:
        out[f"orig_cnt_{w}"] = prior_count_in_window(out, "nameOrig", w)
        out[f"dest_in_cnt_{w}"] = prior_count_in_window(out, "nameDest", w)

    return out.drop(columns=["_row_id"])


#CHANGED: the whole function. Before, each split was processed separately, so every account's
#history restarted at the split boundary. Now the three splits are joined in time order,
#features are computed ONCE, and the result is cut back into train/val/test
def build_features(processed_dir: Path = REPO_ROOT / "data" / "processed") -> None:
    #NEW: accept a str as well as a Path
    processed_dir = Path(processed_dir)
    split_names = ["train", "val", "test"]
    parts: list[pd.DataFrame] = []

    for split_name in split_names:
        in_file = processed_dir / f"{split_name}_raw_clean.parquet"
        split_df = pd.read_parquet(in_file)
        #NEW: remember which split each row belongs to, so it can be cut apart again below
        split_df["_split"] = split_name
        parts.append(split_df)

    #Concatenate chronologically across train -> val -> test so historical
    #origin/destination statistics do not reset at split boundaries (strictly past-only)
    combined_df = pd.concat(parts, ignore_index=True)
    combined_feat = add_leak_free_features(combined_df)

    for split_name in split_names:
        out_file = processed_dir / f"{split_name}_features.parquet"
        #NEW: select this split's rows from the combined table and drop the helper column
        feat_df = (
            combined_feat[combined_feat["_split"] == split_name]
            .drop(columns=["_split"])
            .reset_index(drop=True)
        )
        feat_df.to_parquet(out_file, index=False)
        non_null_prev = int(feat_df["orig_prev_amt_mean"].notna().sum())
        print(
            f"Saved {out_file} | shape={feat_df.shape} | "
            f"non_null orig_prev_amt_mean={non_null_prev}"
        )


if __name__ == "__main__":
    build_features()