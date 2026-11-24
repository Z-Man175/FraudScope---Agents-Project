#NEW: json is used to write the metadata file; joblib is used to save the trained models
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42

#CHANGED: added the six rolling-window count features created in features.py
FEATURE_COLS = [
    "amount",
    "log_amount",
    "hour_of_day",
    "type_code",
    "time_since_prev_orig",
    "is_first_orig_txn",
    "amt_to_orig_prev_mean",
    "orig_dest_prior_count",
    "dest_prior_in_count",
    "orig_prior_txn_count",
    "orig_cnt_1",
    "orig_cnt_6",
    "orig_cnt_24",
    "dest_in_cnt_1",
    "dest_in_cnt_6",
    "dest_in_cnt_24",
]

#NEW: K values reported for precision@K / recall@K
REPORT_KS = (100, 500, 1000, 5000)

#NEW: how many top-ranked transactions go into the candidate file
#PLACEHOLDER: set this from the LLM call budget and check recall@K on validation
CANDIDATE_K = 500


def load_split(processed_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_parquet(processed_dir / f"{split}_features.parquet")
    X = df[FEATURE_COLS].fillna(-1.0)
    y = df["isFraud"].astype(int)
    return df, X, y


def to_unit_rank(scores: np.ndarray) -> np.ndarray:
    if len(scores) <= 1:
        return np.zeros_like(scores, dtype=float)
    return (rankdata(scores, method="average") - 1.0) / (len(scores) - 1.0)


def evaluate_binary(y_true: pd.Series, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


#NEW: searches for the probability threshold that gives the best F1 on the validation set,
#because the fixed 0.5 threshold is arbitrary for a class-weighted model
def tune_threshold_f1(y_true: pd.Series, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    y_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    if len(probs_arr) == 0 or y_arr.sum() == 0:
        return 0.5, evaluate_binary(y_true, probs_arr, threshold=0.5)

    grid = np.linspace(0.01, 0.99, 99)
    quantiles = np.quantile(
        probs_arr,
        np.concatenate([np.linspace(0.05, 0.95, 19), np.linspace(0.96, 0.9999, 40)]),
    )
    candidates = np.unique(np.concatenate([grid, quantiles]))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    if len(candidates) == 0:
        candidates = np.array([0.5])

    best_threshold = 0.5
    best_f1 = -1.0
    for thr in candidates:
        preds = (probs_arr >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (y_arr == 1)))
        fp = int(np.sum((preds == 1) & (y_arr == 0)))
        fn = int(np.sum((preds == 0) & (y_arr == 1)))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)

    best_metrics = evaluate_binary(y_true, probs_arr, threshold=best_threshold)
    return best_threshold, best_metrics


#NEW: precision@K and recall@K for a score column. Precision@K = fraud share among the K
#highest-scored rows; recall@K = share of all fraud found in those K rows
def precision_recall_at_k(
    df: pd.DataFrame, score_col: str, ks: tuple[int, ...] = REPORT_KS
) -> dict[str, dict[str, float]]:
    ranked = df.sort_values(score_col, ascending=False, kind="mergesort")
    total_fraud = max(int(ranked["isFraud"].sum()), 1)
    result = {}
    for k in ks:
        top = ranked["isFraud"].head(k)
        result[str(k)] = {
            "precision": float(top.mean()) if len(top) else 0.0,
            "recall": float(top.sum() / total_fraud),
        }
    return result


def run_anomaly_agent(
    processed_dir: Path = REPO_ROOT / "data" / "processed",
    out_dir: Path = REPO_ROOT / "outputs" / "anomaly",
) -> None:
    #CHANGED: both folders are converted to Path first, so strings also work
    #(before, only out_dir.mkdir was called directly on the argument)
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, X_train, y_train = load_split(processed_dir, "train")
    df_val, X_val, y_val = load_split(processed_dir, "val")
    df_test, X_test, y_test = load_split(processed_dir, "test")

    # Class-weighted supervised model (Setup A baseline + supervised anomaly signal)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)

    gb_model = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED,
        max_iter=200,
        learning_rate=0.05,
    )
    gb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Unsupervised Isolation Forest detector
    iso_model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_model.fit(X_train)

    #NEW: save both trained models so scoring can be re-run later without retraining
    joblib.dump(gb_model, out_dir / "gb_model.joblib")
    joblib.dump(iso_model, out_dir / "iso_model.joblib")

    val_gb_prob = gb_model.predict_proba(X_val)[:, 1]
    val_iso_score = -iso_model.decision_function(X_val)

    #CHANGED: renamed from val_metrics to val_metrics_default, and a second, tuned-threshold
    #evaluation was added next to it (best_threshold is also used for setup_a_pred below)
    val_metrics_default = evaluate_binary(y_val, val_gb_prob, threshold=0.5)
    best_threshold, val_metrics_tuned = tune_threshold_f1(y_val, val_gb_prob)
    val_iso_prauc = float(average_precision_score(y_val, val_iso_score))

    print("=== Setup A Supervised Classifier (Val @ 0.5) ===")
    print(
        f"PR-AUC: {val_metrics_default['pr_auc']:.4f} | Prec: {val_metrics_default['precision']:.4f} | "
        f"Recall: {val_metrics_default['recall']:.4f} | F1: {val_metrics_default['f1']:.4f}"
    )
    #NEW: prints the metrics at the tuned threshold as well
    print(f"=== Setup A Supervised Classifier (Val @ tuned threshold={best_threshold:.4f}) ===")
    print(
        f"PR-AUC: {val_metrics_tuned['pr_auc']:.4f} | Prec: {val_metrics_tuned['precision']:.4f} | "
        f"Recall: {val_metrics_tuned['recall']:.4f} | F1: {val_metrics_tuned['f1']:.4f}"
    )
    print(f"Isolation Forest Val PR-AUC: {val_iso_prauc:.4f}")

    # Tune rank-average weight w on validation using PR-AUC
    val_gb_r = to_unit_rank(val_gb_prob)
    val_iso_r = to_unit_rank(val_iso_score)

    best_w = 1.0
    best_blend_prauc = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * val_gb_r + (1.0 - w) * val_iso_r
        score = float(average_precision_score(y_val, blend))
        if score > best_blend_prauc:
            best_blend_prauc = score
            best_w = float(w)

    print(f"Best Val Rank Blend w={best_w:.2f} | Val Blend PR-AUC: {best_blend_prauc:.4f}")

    #NEW: collects precision@K / recall@K for each split and score column
    at_k_by_split: dict[str, dict] = {}

    # Export scored tables for val and test
    for split_name, df_part, X_part in [("val", df_val, X_val), ("test", df_test, X_test)]:
        gb_prob = gb_model.predict_proba(X_part)[:, 1]
        iso_score = -iso_model.decision_function(X_part)
        combined_score = best_w * to_unit_rank(gb_prob) + (1.0 - best_w) * to_unit_rank(iso_score)

        #CHANGED: the output table gets three new columns: txn_id and type (so later stages can
        #identify the transaction) and setup_a_pred (the 0/1 Setup A prediction at the tuned threshold)
        out_df = pd.DataFrame({
            "txn_id": df_part["txn_id"].values,
            "step": df_part["step"].values,
            "type": df_part["type"].values,
            "nameOrig": df_part["nameOrig"].values,
            "nameDest": df_part["nameDest"].values,
            "amount": df_part["amount"].values,
            "isFraud": df_part["isFraud"].values,
            "setup_a_gb_prob": gb_prob,
            "setup_a_pred": (gb_prob >= best_threshold).astype(int),
            "iso_anomaly_score": iso_score,
            "combined_anomaly_score": combined_score,
        })
        out_path = out_dir / f"{split_name}_anomaly_scores.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"Saved {out_path} ({len(out_df)} rows)")

        #NEW: precision@K / recall@K for the three scores. K is chosen on validation only;
        #the test numbers are reported, never used for tuning
        at_k = {
            col: precision_recall_at_k(out_df, col)
            for col in ["setup_a_gb_prob", "iso_anomaly_score", "combined_anomaly_score"]
        }
        at_k_by_split[split_name] = at_k
        print(f"=== {split_name}: combined score, precision/recall @K ===")
        for k, vals in at_k["combined_anomaly_score"].items():
            print(f"  K={k:>5s} | precision={vals['precision']:.4f} | recall={vals['recall']:.4f}")

        #NEW: top-K candidate file (highest combined score), for the investigation stage
        #This simple version will later be replaced by the ranking step that also uses
        #the pattern agent's flags
        topk_path = out_dir / f"{split_name}_anomaly_topk.parquet"
        out_df.nlargest(CANDIDATE_K, "combined_anomaly_score").to_parquet(
            topk_path, index=False
        )
        print(f"Saved {topk_path} (top {min(CANDIDATE_K, len(out_df))} rows)")

    #NEW: metadata file with everything needed to reproduce and compare runs (seed, features,
    #blend weight, thresholds, validation metrics, precision/recall@K)
    metadata = {
        "random_seed": RANDOM_SEED,
        "feature_cols": list(FEATURE_COLS),
        "best_blend_weight_w": best_w,
        "setup_a_threshold": best_threshold,
        "candidate_k": CANDIDATE_K,
        "val_metrics": {
            "setup_a_pr_auc": val_metrics_tuned["pr_auc"],
            "setup_a_precision_tuned": val_metrics_tuned["precision"],
            "setup_a_recall_tuned": val_metrics_tuned["recall"],
            "setup_a_f1_tuned": val_metrics_tuned["f1"],
            "setup_a_f1_default_0_5": val_metrics_default["f1"],
            "iso_forest_pr_auc": val_iso_prauc,
            "rank_blend_pr_auc": best_blend_prauc,
        },
        "at_k": at_k_by_split,
    }
    metadata_path = out_dir / "anomaly_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    run_anomaly_agent()