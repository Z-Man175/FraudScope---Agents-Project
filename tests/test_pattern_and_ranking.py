"""Unit tests for synthetic scenarios, Pattern Analysis Agent, and Top-K Ranking
Run from the repo root with:  python -m pytest tests -q
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.scenarios import (
    ALL_SCENARIO_COLS,
    CORE_PAYSIM_COLS,
    META_FILE_COLS,
    RANDOM_SEED,
    SCENARIO_META_COLS,
    TRAIN_MAX_STEP,
    VAL_MAX_STEP,
    add_synthetic_txn_ids,
    generate_scenarios,
    save_scenarios,
)
from src.agents.pattern_agent import (
    EVENT_COLS,
    PatternConfig,
    detect_patterns,
    run_pattern_agent,
)
from src.agents.ranking import (
    precision_at_k,
    rank_candidates,
    run_ranking,
    tune_ranking_params,
)
from src.data import Inspect_and_split  #Used to check that synthetic txn_ids agree


#Scenario table with SYN_ ids, labelled with the split each row falls in
def _scenarios_with_ids_and_split() -> pd.DataFrame:
    df = add_synthetic_txn_ids(generate_scenarios(seed=RANDOM_SEED))
    df["split"] = np.where(
        df["step"] <= TRAIN_MAX_STEP, "train", np.where(df["step"] <= VAL_MAX_STEP, "val", "test")
    )
    return df


#Pattern agent input = the core PaySim columns + txn_id (what the real pipeline has;
#the scenario labels must NOT be passed to the detectors)
def _detector_input(df: pd.DataFrame) -> pd.DataFrame:
    return df[["txn_id"] + CORE_PAYSIM_COLS]


#Look-alike families now include business_forwarding, and the metadata files must
#carry txn_id (META_FILE_COLS)
def test_generate_scenarios_schema_and_determinism(tmp_path: Path) -> None:
    df_a = generate_scenarios(seed=RANDOM_SEED)
    df_b = generate_scenarios(seed=RANDOM_SEED)

    pd.testing.assert_frame_equal(df_a, df_b)
    assert list(df_a.columns) == ALL_SCENARIO_COLS
    assert not df_a.empty

    #Verify all required fraud patterns are present with isFraud == 1
    fraud_df = df_a[df_a["isFraud"] == 1]
    assert set(fraud_df["pattern_name"].unique()) == {
        "burst_transfer",
        "repeated_recipient_structuring",
        "account_chain",
    }

    #Verify held-out candidate pattern exists and is marked appropriately
    held_out_df = df_a[df_a["held_out_candidate"]]
    assert not held_out_df.empty
    assert set(held_out_df["pattern_name"].unique()) == {"account_chain"}
    assert (held_out_df["isFraud"] == 1).all()

    #Verify legitimate look-alikes exist with isFraud == 0 and is_lookalike == True
    lookalike_df = df_a[df_a["is_lookalike"]]
    assert not lookalike_df.empty
    assert (lookalike_df["isFraud"] == 0).all()
    assert set(lookalike_df["pattern_name"].unique()) == {
        "landlord_recurring",
        "payroll_or_batch_legit",
        "business_forwarding",
    }

    #Verify save_scenarios writes clean PaySim parquet + metadata parquet + CSV
    raw_pq = tmp_path / "synthetic_scenarios.parquet"
    meta_pq = tmp_path / "synthetic_scenarios_meta.parquet"
    meta_csv = tmp_path / "synthetic_scenarios_meta.csv"

    save_scenarios(out_parquet=raw_pq, out_csv=meta_csv, out_meta_parquet=meta_pq, seed=RANDOM_SEED)

    raw_loaded = pd.read_parquet(raw_pq)
    meta_loaded = pd.read_parquet(meta_pq)
    csv_loaded = pd.read_csv(meta_csv)

    assert list(raw_loaded.columns) == CORE_PAYSIM_COLS
    assert len(raw_loaded) == len(df_a)
    assert list(meta_loaded.columns) == META_FILE_COLS
    assert len(csv_loaded) == len(df_a)


#Every family must exist in every split, except the held-out chain fraud which must be
#test-only. Without this the evaluation has nothing to measure in some splits
def test_scenarios_cover_every_split_and_hold_out_chains() -> None:
    df = _scenarios_with_ids_and_split()
    counts = df.groupby(["pattern_name", "split"]).size().unstack(fill_value=0)

    families = [
        "burst_transfer",
        "repeated_recipient_structuring",
        "landlord_recurring",
        "payroll_or_batch_legit",
        "business_forwarding",
    ]
    for family in families:
        for split in ("train", "val", "test"):
            assert counts.loc[family, split] > 0, f"{family} has no rows in {split}"

    chain = counts.loc["account_chain"]
    assert chain["train"] == 0 and chain["val"] == 0 and chain["test"] > 0

    #Every history tier must be present among the sender accounts (>= 10 / 3-9 / <= 2 txns)
    per_account = df[df["scenario_type"] == "background"].groupby("nameOrig").size()
    assert (per_account >= 10).any() and per_account.between(3, 9).any() and (per_account <= 2).any()


#Synthetic txn_ids in the metadata file must be exactly the ids Inspect_and_split.py gives
#the same rows after it merges the core parquet. If this fails, scenario labels no longer join
def test_synthetic_ids_match_split_script(tmp_path: Path) -> None:
    raw_pq = tmp_path / "synthetic_scenarios.parquet"
    meta_pq = tmp_path / "synthetic_scenarios_meta.parquet"
    save_scenarios(out_parquet=raw_pq, out_csv=None, out_meta_parquet=meta_pq, seed=RANDOM_SEED)

    #A tiny stand-in for PaySim (balance columns are dropped by the split script)
    paysim = pd.DataFrame(
        {
            "step": [1, 300, 700],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
            "amount": [10.0, 500.0, 70.0],
            "nameOrig": ["C1", "C2", "C3"],
            "oldbalanceOrg": [0.0] * 3,
            "newbalanceOrig": [0.0] * 3,
            "nameDest": ["M1", "C9", "C8"],
            "oldbalanceDest": [0.0] * 3,
            "newbalanceDest": [0.0] * 3,
            "isFraud": [0, 0, 1],
            "isFlaggedFraud": [0, 0, 0],
        }
    )
    splits = Inspect_and_split.clean_and_split(
        paysim, tmp_path / "processed", train_max=TRAIN_MAX_STEP, val_max=VAL_MAX_STEP,
        extra_transactions_path=raw_pq,
    )
    merged_out = pd.concat(splits.values(), ignore_index=True)
    syn_out = merged_out[merged_out["txn_id"].str.startswith("SYN_")]
    meta = pd.read_parquet(meta_pq)

    joined = syn_out.merge(meta, on="txn_id", suffixes=("_split", "_meta"))
    assert len(joined) == len(meta) == len(syn_out)
    for col in ("step", "nameOrig", "nameDest", "amount"):
        assert (joined[f"{col}_split"] == joined[f"{col}_meta"]).all()
    #PaySim ids keep their own prefix
    assert set(merged_out.loc[~merged_out["txn_id"].str.startswith("SYN_"), "txn_id"]) == {
        "PS_0000000", "PS_0000001", "PS_0000002"
    }


#txn_ids now use the real SYN_ convention (before: TXN_). Also checks that the event
#table carries the new event_id / window columns
def test_pattern_agent_detects_synthetic_patterns_deterministically() -> None:
    scenarios = generate_scenarios(seed=RANDOM_SEED)
    scenarios_with_id = add_synthetic_txn_ids(scenarios)

    events_1, flags_1 = detect_patterns(scenarios_with_id)
    events_2, flags_2 = detect_patterns(scenarios_with_id)

    pd.testing.assert_frame_equal(events_1, events_2)
    pd.testing.assert_frame_equal(flags_1, flags_2)

    assert list(events_1.columns) == EVENT_COLS
    assert not events_1.empty
    assert set(events_1["pattern_type"].unique()) == {
        "burst",
        "repeated_recipient",
        "account_chain",
    }
    assert events_1["pattern_score"].between(0.0, 1.0).all()
    assert events_1["reason"].astype(str).str.len().gt(0).all()
    #Event ids are unique and the window matches the involved steps
    assert events_1["event_id"].is_unique
    assert (events_1["window_start"] <= events_1["window_end"]).all()

    #Transaction-level summary matches input length and scores are bounded in [0, 1]
    assert len(flags_1) == len(scenarios_with_id)
    assert flags_1["pattern_score"].between(0.0, 1.0).all()
    assert flags_1["is_pattern_flagged"].sum() > 0

    #Every event id mentioned on a transaction exists in the event table
    known_ids = set(events_1["event_id"])
    mentioned = {e for cell in flags_1["pattern_event_ids"] for e in cell.split(",") if e}
    assert mentioned and mentioned <= known_ids

    #Also verify fallback deterministic txn_id synthesis when txn_id is omitted
    events_no_id, flags_no_id = detect_patterns(scenarios[CORE_PAYSIM_COLS])
    assert set(events_no_id["pattern_type"].unique()) == {
        "burst",
        "repeated_recipient",
        "account_chain",
    }
    assert flags_no_id["txn_id"].str.startswith("ROW_").all()

    #In the real pipeline a missing txn_id is an error, not something to invent
    with pytest.raises(ValueError, match="txn_id"):
        detect_patterns(scenarios[CORE_PAYSIM_COLS], require_txn_id=True)


#The detectors must catch every synthetic fraud row, and must also fire on the look-alikes
#(that is the point of look-alikes: the reviewer, not the pattern agent, separates them)
#A fixed cutoff of 30,000 is used for "small" so the result does not depend on the scenario median
def test_detectors_catch_all_synthetic_fraud_and_trip_on_lookalikes() -> None:
    df = _scenarios_with_ids_and_split()
    _, flags = detect_patterns(_detector_input(df), PatternConfig(small_amt_cutoff=30_000.0))
    labelled = df.merge(flags[["txn_id", "pattern_score", "pattern_types"]], on="txn_id")
    labelled["flagged"] = labelled["pattern_score"] > 0

    fraud = labelled[labelled["isFraud"] == 1]
    assert fraud["flagged"].all(), "some synthetic fraud rows were not flagged"

    #each fraud family is flagged by the matching detector
    expected = {
        "burst_transfer": "burst",
        "repeated_recipient_structuring": "repeated_recipient",
        "account_chain": "account_chain",
    }
    for pattern_name, detector in expected.items():
        types = fraud.loc[fraud["pattern_name"] == pattern_name, "pattern_types"]
        assert types.str.contains(detector).all(), pattern_name

    #every look-alike family trips at least one detector in train, val and test
    for family in ("landlord_recurring", "payroll_or_batch_legit", "business_forwarding"):
        part = labelled[labelled["pattern_name"] == family]
        for split in ("train", "val", "test"):
            assert part.loc[part["split"] == split, "flagged"].any(), f"{family} not flagged in {split}"

    #plain history rows of normal accounts are (almost) never flagged
    background = labelled[labelled["scenario_type"] == "background"]
    assert background["flagged"].mean() < 0.05


#The hand-made data now has a 5-txn burst, a 4-txn structuring run and a 3-hop chain,
#because the defaults moved from (4 / 3 / any 2-hop) to (5 / 4 / 3 hops). Also checks that the
#fixed train cutoff is written to the summary and that the summary reports fraud capture
def test_pattern_agent_targeted_detectors_and_cli_export(tmp_path: Path) -> None:
    #Minimal hand-crafted dataset with one burst, one structuring target, one 3-hop chain,
    #and two isolated normal transactions
    rows = [
        #Burst from O_BURST within 4 steps (5 txns)
        {"txn_id": "T01", "step": 10, "type": "TRANSFER", "amount": 50_000.0, "nameOrig": "O_BURST", "nameDest": "D1", "isFraud": 1},
        {"txn_id": "T02", "step": 11, "type": "TRANSFER", "amount": 48_000.0, "nameOrig": "O_BURST", "nameDest": "D2", "isFraud": 1},
        {"txn_id": "T03", "step": 12, "type": "CASH_OUT", "amount": 52_000.0, "nameOrig": "O_BURST", "nameDest": "D3", "isFraud": 1},
        {"txn_id": "T04", "step": 13, "type": "TRANSFER", "amount": 49_000.0, "nameOrig": "O_BURST", "nameDest": "D4", "isFraud": 1},
        {"txn_id": "T04b", "step": 14, "type": "TRANSFER", "amount": 51_000.0, "nameOrig": "O_BURST", "nameDest": "D5", "isFraud": 1},
        #Repeated small transfers to D_STRUCT within 9 steps (4 txns)
        {"txn_id": "T05", "step": 20, "type": "TRANSFER", "amount": 1_000.0, "nameOrig": "O_S1", "nameDest": "D_STRUCT", "isFraud": 1},
        {"txn_id": "T06", "step": 23, "type": "TRANSFER", "amount": 1_200.0, "nameOrig": "O_S2", "nameDest": "D_STRUCT", "isFraud": 1},
        {"txn_id": "T07", "step": 26, "type": "CASH_OUT", "amount": 1_100.0, "nameOrig": "O_S1", "nameDest": "D_STRUCT", "isFraud": 1},
        {"txn_id": "T07b", "step": 29, "type": "TRANSFER", "amount": 1_050.0, "nameOrig": "O_S2", "nameDest": "D_STRUCT", "isFraud": 1},
        #3-hop chain C_A -> C_B -> C_C -> C_D
        {"txn_id": "T08", "step": 40, "type": "TRANSFER", "amount": 80_000.0, "nameOrig": "C_A", "nameDest": "C_B", "isFraud": 1},
        {"txn_id": "T09", "step": 42, "type": "TRANSFER", "amount": 78_000.0, "nameOrig": "C_B", "nameDest": "C_C", "isFraud": 1},
        {"txn_id": "T10", "step": 44, "type": "CASH_OUT", "amount": 76_000.0, "nameOrig": "C_C", "nameDest": "C_D", "isFraud": 1},
        #Isolated normal transactions
        {"txn_id": "T11", "step": 100, "type": "PAYMENT", "amount": 250.0, "nameOrig": "O_NORM1", "nameDest": "M_1", "isFraud": 0},
        {"txn_id": "T12", "step": 150, "type": "PAYMENT", "amount": 300.0, "nameOrig": "O_NORM2", "nameDest": "M_2", "isFraud": 0},
        #A 2-hop pass-through is NOT a chain any more (needs >= 3 hops)
        {"txn_id": "T13", "step": 200, "type": "TRANSFER", "amount": 9_000.0, "nameOrig": "P_A", "nameDest": "P_B", "isFraud": 0},
        {"txn_id": "T14", "step": 201, "type": "TRANSFER", "amount": 8_900.0, "nameOrig": "P_B", "nameDest": "P_C", "isFraud": 0},
    ]
    df = pd.DataFrame(rows)
    events_df, flags_df = detect_patterns(df, config=PatternConfig())

    assert set(events_df["pattern_type"]) == {"burst", "repeated_recipient", "account_chain"}
    normal_flags = flags_df[flags_df["txn_id"].isin(["T11", "T12", "T13", "T14"])]
    assert (~normal_flags["is_pattern_flagged"]).all()
    assert (normal_flags["pattern_score"] == 0.0).all()

    #Exercise run_pattern_agent end-to-end on temp processed dir
    proc_dir = tmp_path / "processed"
    pat_out = tmp_path / "pattern_out"
    proc_dir.mkdir()
    df.to_parquet(proc_dir / "val_features.parquet", index=False)
    df.to_parquet(proc_dir / "test_features.parquet", index=False)
    #A train file, so the fixed "small amount" cutoff can be derived from it
    df.to_parquet(proc_dir / "train_raw_clean.parquet", index=False)

    run_pattern_agent(processed_dir=proc_dir, out_dir=pat_out, splits=("val", "test"))
    assert (pat_out / "val_pattern_flags.parquet").exists()
    assert (pat_out / "test_pattern_flags.parquet").exists()
    assert (pat_out / "val_pattern_events.parquet").exists()
    assert (pat_out / "pattern_summary.json").exists()

    #Summary records the train-derived cutoff and reports how many fraud rows were flagged
    summary = json.loads((pat_out / "pattern_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["small_amt_cutoff"] == pytest.approx(float(df["amount"].quantile(0.5)))
    val_summary = summary["splits"]["val"]
    assert val_summary["total_fraud"] == 12
    assert val_summary["flagged_fraud"] == 12
    assert val_summary["flag_recall"] == pytest.approx(1.0)
    #Event ids are prefixed with the split so they stay unique across files
    val_events = pd.read_parquet(pat_out / "val_pattern_events.parquet")
    assert val_events["event_id"].str.startswith("EV_VAL_").all()

    #A file without txn_id must be rejected by the real pipeline entry point
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    df.drop(columns=["txn_id"]).to_parquet(bad_dir / "val_raw_clean.parquet", index=False)
    with pytest.raises(ValueError, match="txn_id"):
        run_pattern_agent(processed_dir=bad_dir, out_dir=tmp_path / "bad_out", splits=("val",))


#Also checks the new candidate columns (source, reasons, event ids, flag count) and the
#new fixed-K default behaviour of tune_ranking_params
def test_ranking_merges_computes_precision_at_k_and_exports(tmp_path: Path) -> None:
    anomaly_df = pd.DataFrame(
        [
            {"txn_id": "TXN_0000000", "step": 501, "type": "TRANSFER", "nameOrig": "O1", "nameDest": "D1", "amount": 90_000.0, "isFraud": 1, "combined_anomaly_score": 0.95},
            {"txn_id": "TXN_0000001", "step": 502, "type": "TRANSFER", "nameOrig": "O2", "nameDest": "D2", "amount": 75_000.0, "isFraud": 1, "combined_anomaly_score": 0.70},
            {"txn_id": "TXN_0000002", "step": 503, "type": "PAYMENT", "nameOrig": "O3", "nameDest": "D3", "amount": 500.0, "isFraud": 0, "combined_anomaly_score": 0.80},
            {"txn_id": "TXN_0000003", "step": 504, "type": "CASH_OUT", "nameOrig": "O4", "nameDest": "D4", "amount": 60_000.0, "isFraud": 1, "combined_anomaly_score": 0.60},
            {"txn_id": "TXN_0000004", "step": 505, "type": "PAYMENT", "nameOrig": "O5", "nameDest": "D5", "amount": 120.0, "isFraud": 0, "combined_anomaly_score": 0.15},
            {"txn_id": "TXN_0000005", "step": 506, "type": "PAYMENT", "nameOrig": "O6", "nameDest": "D6", "amount": 200.0, "isFraud": 0, "combined_anomaly_score": 0.10},
        ]
    )
    pattern_df = pd.DataFrame(
        [
            {"txn_id": "TXN_0000000", "step": 501, "nameOrig": "O1", "nameDest": "D1", "amount": 90_000.0, "pattern_score": 0.85, "pattern_types": "burst", "pattern_flag_count": 1, "pattern_reasons": "r-burst", "pattern_event_ids": "EV_1"},
            {"txn_id": "TXN_0000001", "step": 502, "nameOrig": "O2", "nameDest": "D2", "amount": 75_000.0, "pattern_score": 0.90, "pattern_types": "account_chain", "pattern_flag_count": 1, "pattern_reasons": "r-chain", "pattern_event_ids": "EV_2"},
            {"txn_id": "TXN_0000002", "step": 503, "nameOrig": "O3", "nameDest": "D3", "amount": 500.0, "pattern_score": 0.0, "pattern_types": "", "pattern_flag_count": 0, "pattern_reasons": "", "pattern_event_ids": ""},
            {"txn_id": "TXN_0000003", "step": 504, "nameOrig": "O4", "nameDest": "D4", "amount": 60_000.0, "pattern_score": 0.80, "pattern_types": "repeated_recipient", "pattern_flag_count": 1, "pattern_reasons": "r-rep", "pattern_event_ids": "EV_3"},
            {"txn_id": "TXN_0000004", "step": 505, "nameOrig": "O5", "nameDest": "D5", "amount": 120.0, "pattern_score": 0.0, "pattern_types": "", "pattern_flag_count": 0, "pattern_reasons": "", "pattern_event_ids": ""},
            {"txn_id": "TXN_0000005", "step": 506, "nameOrig": "O6", "nameDest": "D6", "amount": 200.0, "pattern_score": 0.0, "pattern_types": "", "pattern_flag_count": 0, "pattern_reasons": "", "pattern_event_ids": ""},
        ]
    )

    top_3 = rank_candidates(anomaly_df, pattern_df, top_k=3, alpha=0.5)
    assert len(top_3) == 3
    assert list(top_3["rank"]) == [1, 2, 3]
    for col in [
        "txn_id",
        "step",
        "nameOrig",
        "nameDest",
        "amount",
        "isFraud",
        "combined_anomaly_score",
        "pattern_score",
        "pattern_types",
        "ranking_score",
        "rank",
        "pattern_flag_count",
        "pattern_reasons",
        "pattern_event_ids",
        "source",
    ]:
        assert col in top_3.columns

    #With alpha=0.5, the 3 fraud rows with high pattern scores outrank the benign high-anomaly row
    assert precision_at_k(top_3["isFraud"], k=3) == pytest.approx(1.0)
    assert top_3["ranking_score"].is_monotonic_decreasing

    #`source` semantics. The anomaly-only top 3 is TXN_0, _2, _1. The blended top 3 is
    #TXN_0, _1, _3. TXN_3 is pattern-flagged but outside the anomaly-only top 3 -> "pattern";
    #TXN_0 and _1 are flagged and inside it -> "both"
    sources = dict(zip(top_3["txn_id"], top_3["source"]))
    assert sources == {"TXN_0000000": "both", "TXN_0000001": "both", "TXN_0000003": "pattern"}
    assert top_3.set_index("txn_id").loc["TXN_0000003", "pattern_reasons"] == "r-rep"

    #Verify fallback merge without txn_id returns at most K rows
    top_2_fallback = rank_candidates(
        anomaly_df.drop(columns=["txn_id"]),
        pattern_df.drop(columns=["txn_id"]),
        top_k=2,
        alpha=0.5,
    )
    assert len(top_2_fallback) == 2
    assert "txn_id" in top_2_fallback.columns

    #With a fixed K only alpha is tuned and K is returned unchanged
    tuned = tune_ranking_params(anomaly_df, pattern_df, fixed_top_k=3)
    assert tuned["chosen_top_k"] == 3
    assert tuned["val_precision_at_k"] == pytest.approx(1.0)

    #Verify full run_ranking pipeline writes parquet + JSON summary
    anom_dir = tmp_path / "anomaly"
    pat_dir = tmp_path / "pattern"
    rank_dir = tmp_path / "ranking"
    anom_dir.mkdir()
    pat_dir.mkdir()

    anomaly_df.to_parquet(anom_dir / "val_anomaly_scores.parquet", index=False)
    pattern_df.to_parquet(pat_dir / "val_pattern_flags.parquet", index=False)
    anomaly_df.to_parquet(anom_dir / "test_anomaly_scores.parquet", index=False)
    pattern_df.to_parquet(pat_dir / "test_pattern_flags.parquet", index=False)

    summary = run_ranking(
        anomaly_dir=anom_dir,
        pattern_dir=pat_dir,
        out_dir=rank_dir,
        top_k=3,
        alpha=0.5,
    )

    val_out = pd.read_parquet(rank_dir / "val_top_k_candidates.parquet")
    test_out = pd.read_parquet(rank_dir / "test_top_k_candidates.parquet")
    summary_json = json.loads((rank_dir / "ranking_summary.json").read_text(encoding="utf-8"))

    assert len(val_out) <= 3
    assert len(test_out) <= 3
    assert summary["chosen_top_k"] == 3
    assert summary["chosen_alpha"] == pytest.approx(0.5)
    assert summary_json["splits"]["val"]["precision_at_k"] == pytest.approx(1.0)
    assert summary_json["splits"]["test"]["precision_at_k"] == pytest.approx(1.0)