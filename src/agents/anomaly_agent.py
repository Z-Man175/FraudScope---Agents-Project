from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42

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
]


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


def run_anomaly_agent(
    processed_dir: Path = REPO_ROOT / "data" / "processed",
    out_dir: Path = REPO_ROOT / "outputs" / "anomaly",
) -> None:
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

    val_gb_prob = gb_model.predict_proba(X_val)[:, 1]
    val_iso_score = -iso_model.decision_function(X_val)

    val_metrics = evaluate_binary(y_val, val_gb_prob, threshold=0.5)
    val_iso_prauc = float(average_precision_score(y_val, val_iso_score))

    print("=== Setup A Supervised Classifier (Val @ 0.5) ===")
    print(
        f"PR-AUC: {val_metrics['pr_auc']:.4f} | Prec: {val_metrics['precision']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}"
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

    # Export scored tables for val and test
    for split_name, df_part, X_part in [("val", df_val, X_val), ("test", df_test, X_test)]:
        gb_prob = gb_model.predict_proba(X_part)[:, 1]
        iso_score = -iso_model.decision_function(X_part)
        combined_score = best_w * to_unit_rank(gb_prob) + (1.0 - best_w) * to_unit_rank(iso_score)

        out_df = pd.DataFrame({
            "step": df_part["step"].values,
            "nameOrig": df_part["nameOrig"].values,
            "nameDest": df_part["nameDest"].values,
            "amount": df_part["amount"].values,
            "isFraud": df_part["isFraud"].values,
            "setup_a_gb_prob": gb_prob,
            "iso_anomaly_score": iso_score,
            "combined_anomaly_score": combined_score,
        })
        out_path = out_dir / f"{split_name}_anomaly_scores.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"Saved {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    run_anomaly_agent()