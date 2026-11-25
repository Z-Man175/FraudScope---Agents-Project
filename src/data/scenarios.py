"""Deterministic synthetic fraud and look-alike scenario generator for FraudScope

WHY THIS FILE EXISTS
    In real PaySim almost every sender account appears exactly once, so "this account's own
    history" barely exists. The reviewer and the evidence tools need accounts
    that do have history, plus look-alike legitimate behaviour that resembles fraud. This file
    generates both, in the PaySim schema, so they can be merged into the real data

WHAT IT GENERATES (scenario_type column)
    fraud      (isFraud == 1)
        burst_transfer                  rapid run of transfers from one account
        repeated_recipient_structuring  many small transfers into one destination
        account_chain                   A -> B -> C -> D forwarding (held-out pattern)
    lookalike  (isFraud == 0, is_lookalike == True)   legit behaviour that trips the same detectors
        landlord_recurring              tenants pay one landlord on a regular cycle
        payroll_or_batch_legit          employer pays several staff in one batch, every cycle
        business_forwarding             same A -> B -> C -> D shape, but middle accounts are
                                        established and the chain repeats every cycle
    background (isFraud == 0)           ordinary accounts with 15-40 / 3-9 / 1-2 transactions
                                        (established / sparse / cold-start history tiers)
    history    (isFraud == 0)           earlier normal activity of an account that later turns
                                        fraudulent ("warm" fraud) or relays a legit chain

SPLIT PLACEMENT (must match the defaults in Inspect_and_split.py: train <= 480, val <= 600)
    Every fraud family and every look-alike family has rows in train, val AND test,
    EXCEPT account_chain fraud, which is test-only (it is the held-out pattern for the memory
    experiment, so it must not appear in train or val)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42

#Split boundaries, copied from the defaults in Inspect_and_split.py. They are only used to
#document and test where scenarios land; if the split script's defaults change, change these too
TRAIN_MAX_STEP = 480
VAL_MAX_STEP = 600
LAST_STEP = 740  #PaySim runs from step 1 to 743; stay a little inside so +offsets never overflow

#How many background accounts of each history tier to generate
N_ESTABLISHED_ACCOUNTS = 120  #15-40 transactions each
N_SPARSE_ACCOUNTS = 30        #3-9 transactions each
N_COLD_START_ACCOUNTS = 30    #1-2 transactions each

#Background transactions are placed on a grid 6 steps apart, so two transactions of the
#same account are always >= 6 steps apart. That makes it impossible for background accounts to
#satisfy the burst rule by accident (5 txns within 6 steps)
BACKGROUND_STEP_GRID_GAP = 6

#Columns that match PaySim; these are what Inspect_and_split.py requires for the merge
CORE_PAYSIM_COLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
]

SCENARIO_META_COLS = [
    "scenario_id",
    "scenario_type",
    "pattern_name",
    "is_lookalike",
    "held_out_candidate",
]

ALL_SCENARIO_COLS = CORE_PAYSIM_COLS + SCENARIO_META_COLS

#The metadata files also carry txn_id (see add_synthetic_txn_ids) so that scenario labels
#can be joined back onto the pipeline outputs. generate_scenarios() itself does not include it
META_FILE_COLS = ["txn_id"] + ALL_SCENARIO_COLS

DEFAULT_RAW_PARQUET = REPO_ROOT / "data" / "raw" / "synthetic_scenarios.parquet"
DEFAULT_META_PARQUET = REPO_ROOT / "data" / "processed" / "synthetic_scenarios_meta.parquet"
DEFAULT_META_CSV = REPO_ROOT / "data" / "processed" / "synthetic_scenarios_meta.csv"


def _make_row(
    *,
    step: int,
    txn_type: str,
    amount: float,
    name_orig: str,
    name_dest: str,
    is_fraud: int,
    scenario_id: str,
    scenario_type: str,
    pattern_name: str,
    is_lookalike: bool,
    held_out_candidate: bool,
) -> dict[str, object]:
    """Build one transaction row (PaySim columns + scenario metadata) with clean dtypes."""
    return {
        "step": int(step),
        "type": str(txn_type),
        "amount": round(float(amount), 2),
        "nameOrig": str(name_orig),
        "nameDest": str(name_dest),
        "isFraud": int(is_fraud),
        "scenario_id": str(scenario_id),
        "scenario_type": str(scenario_type),
        "pattern_name": str(pattern_name),
        "is_lookalike": bool(is_lookalike),
        "held_out_candidate": bool(held_out_candidate),
    }


#Shared helper that writes ordinary past activity for ONE account. Used for background
#accounts, for fraud accounts that were normal before the attack ("warm" fraud), and for the
#middle accounts of a legitimate forwarding chain
#  -amounts are lognormal around a per-account median (returned so attacks can scale from it)
#  -80% of transactions are TRANSFERs to 1-3 "usual" recipients, 20% are PAYMENTs to new merchants
#  -steps come from a grid 6 apart, so these rows can never form a burst
def _history_rows(
    rng: np.random.Generator,
    account: str,
    n_txns: int,
    last_step: int,
    scenario_id: str,
    scenario_type: str,
    pattern_name: str,
) -> tuple[list[dict[str, object]], float]:
    grid = np.arange(1, max(int(last_step) - BACKGROUND_STEP_GRID_GAP + 1, 2), BACKGROUND_STEP_GRID_GAP)
    n_txns = int(min(n_txns, len(grid)))
    steps = np.sort(rng.choice(grid, size=n_txns, replace=False))

    median_amt = float(np.exp(rng.normal(np.log(20_000.0), 0.7)))
    n_usual = int(rng.integers(1, 4))
    usual_recipients = [f"C_SYN_USUAL_{account}_{k}" for k in range(1, n_usual + 1)]

    rows: list[dict[str, object]] = []
    for j, step in enumerate(steps, start=1):
        amount = median_amt * float(np.exp(rng.normal(0.0, 0.35)))
        if rng.random() < 0.8:
            txn_type = "TRANSFER"
            dest = usual_recipients[int(rng.integers(0, n_usual))]
        else:
            txn_type = "PAYMENT"
            dest = f"M_SYN_{account}_{j:03d}"
        rows.append(
            _make_row(
                step=int(step),
                txn_type=txn_type,
                amount=amount,
                name_orig=account,
                name_dest=dest,
                is_fraud=0,
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                pattern_name=pattern_name,
                is_lookalike=False,
                held_out_candidate=False,
            )
        )
    return rows, median_amt


#Background accounts in all three history tiers (established / sparse / cold start), so the
#reviewer's tier logic has every tier to work with and the anomaly agent sees organic-looking rows
def _generate_background_accounts(rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tiers = [
        ("EST", N_ESTABLISHED_ACCOUNTS, 15, 41),
        ("SPARSE", N_SPARSE_ACCOUNTS, 3, 10),
        ("COLD", N_COLD_START_ACCOUNTS, 1, 3),
    ]
    for tier, n_accounts, lo, hi in tiers:
        for idx in range(1, n_accounts + 1):
            account = f"C_SYN_BG_{tier}_{idx:03d}"
            n_txns = int(rng.integers(lo, hi))
            hist, _ = _history_rows(
                rng,
                account=account,
                n_txns=n_txns,
                last_step=LAST_STEP,
                scenario_id=f"SYN_BG_{tier}_{idx:03d}",
                scenario_type="background",
                pattern_name="normal_activity",
            )
            rows.extend(hist)
    return rows


#Bursts now use a 5-7 transaction run (the pattern agent needs >= 5 within 6 steps)
#and every second burst account is "warm": it first gets 12-20 normal transactions, so it is an
#established account that suddenly misbehaves. Before, every fraud account had no history, which
#let "has history" separate fraud from look-alikes on its own
def _generate_burst_transfers(rng: np.random.Generator) -> list[dict[str, object]]:
    """Rapid bursts of outgoing transfers from a single origin within <= 3 steps."""
    rows: list[dict[str, object]] = []
    #Two bursts in train (<=480), two in val (481..600), two in test (>600)
    start_steps = [120, 340, 495, 560, 625, 690]
    for idx, base_step in enumerate(start_steps, start=1):
        scenario_id = f"SYN_BURST_{idx:03d}"
        orig = f"C_SYN_BURST_ORIG_{idx:03d}"

        #Warm accounts (idx 2, 4, 6) get normal history before the burst
        warm = idx % 2 == 0
        hist_median = None
        if warm:
            hist, hist_median = _history_rows(
                rng,
                account=orig,
                n_txns=int(rng.integers(12, 21)),
                last_step=base_step - 1,
                scenario_id=scenario_id,
                scenario_type="history",
                pattern_name="warm_history",
            )
            rows.extend(hist)

        n_txns = int(rng.integers(5, 8))  #5..7 txns (>= burst_min_txns=5)
        offsets = np.sort(rng.integers(0, 4, size=n_txns))  #span <= 3 (<= burst_window_steps=6)
        for j, offset in enumerate(offsets, start=1):
            txn_type = "TRANSFER" if j % 2 == 1 else "CASH_OUT"
            dest = f"C_SYN_BURST_DEST_{idx:03d}_{j:02d}"
            #Warm bursts are 1.5-4x the account's usual amount (a mild-to-clear spike
            #against its own baseline); cold bursts keep the original 25k-120k range
            if hist_median is not None:
                amount = hist_median * float(rng.uniform(1.5, 4.0))
            else:
                amount = float(rng.uniform(25_000.0, 120_000.0))
            rows.append(
                _make_row(
                    step=int(base_step + offset),
                    txn_type=txn_type,
                    amount=amount,
                    name_orig=orig,
                    name_dest=dest,
                    is_fraud=1,
                    scenario_id=scenario_id,
                    scenario_type="fraud",
                    pattern_name="burst_transfer",
                    is_lookalike=False,
                    held_out_candidate=False,
                )
            )
    return rows


#4-6 small transfers into one destination from two mule origins
def _generate_repeated_recipient_structuring(rng: np.random.Generator) -> list[dict[str, object]]:
    """Multiple smaller TRANSFER/CASH_OUT txns routed to the same destination within <= 15 steps."""
    rows: list[dict[str, object]] = []
    #Two in train, two in val, two in test
    start_steps = [160, 390, 510, 575, 640, 705]
    for idx, base_step in enumerate(start_steps, start=1):
        scenario_id = f"SYN_STRUCT_{idx:03d}"
        dest = f"C_SYN_STRUCT_DEST_{idx:03d}"
        n_txns = int(rng.integers(4, 7))  #4..6 txns (>= repeat_dest_min_txns=4)
        offsets = np.sort(rng.integers(0, 16, size=n_txns))  #span <= 15 (<= 24)
        for j, offset in enumerate(offsets, start=1):
            #Two mule origins alternate, each sending small structured amounts
            orig = f"C_SYN_STRUCT_ORIG_{idx:03d}_{(j % 2) + 1:02d}"
            txn_type = "TRANSFER" if j < n_txns else "CASH_OUT"
            #Intentionally small amounts so they fall well below the median amount
            amount = float(rng.uniform(1_800.0, 4_800.0))
            rows.append(
                _make_row(
                    step=int(base_step + offset),
                    txn_type=txn_type,
                    amount=amount,
                    name_orig=orig,
                    name_dest=dest,
                    is_fraud=1,
                    scenario_id=scenario_id,
                    scenario_type="fraud",
                    pattern_name="repeated_recipient_structuring",
                    is_lookalike=False,
                    held_out_candidate=False,
                )
            )
    return rows


#All six chains now start after step 600 (test only)
def _generate_account_chains(rng: np.random.Generator) -> list[dict[str, object]]:
    """Sequential fund hops A -> B -> C -> D within short step gaps (held-out, test period only)."""
    rows: list[dict[str, object]] = []
    start_steps = [610, 630, 650, 670, 690, 710]
    for idx, base_step in enumerate(start_steps, start=1):
        scenario_id = f"SYN_CHAIN_{idx:03d}"
        accounts = [f"C_SYN_CHAIN_{idx:03d}_{node}" for node in ("A", "B", "C", "D")]
        step_cursor = int(base_step)
        amt_cursor = float(rng.uniform(55_000.0, 160_000.0))
        for hop in range(3):  #3 hops: A->B, B->C, C->D
            if hop > 0:
                step_cursor += int(rng.integers(1, 4))  #1..3 steps gap (<= chain_max_step_gap)
                amt_cursor *= float(rng.uniform(0.95, 0.995))  #each hop forwards 95-99.5%
            txn_type = "TRANSFER" if hop < 2 else "CASH_OUT"
            rows.append(
                _make_row(
                    step=step_cursor,
                    txn_type=txn_type,
                    amount=amt_cursor,
                    name_orig=accounts[hop],
                    name_dest=accounts[hop + 1],
                    is_fraud=1,
                    scenario_id=scenario_id,
                    scenario_type="fraud",
                    pattern_name="account_chain",
                    is_lookalike=False,
                    held_out_candidate=True,
                )
            )
    return rows


#Rent is now sent as TRANSFER, not PAYMENT. The detectors only look at
#TRANSFER/CASH_OUT, so as PAYMENT this look-alike never triggered anything and tested nothing
#The cycle start steps moved so that 5 cycles land in train (3), val (1) and test (1); before,
#none reached test. Five cycles also mean the last two cycles have >= 3 prior cycles of history
def _generate_landlord_recurring(rng: np.random.Generator) -> list[dict[str, object]]:
    """Tenants pay one landlord the same rent every 168 steps (7 days, the longest cycle that
    fits several times into PaySim's 30 days)."""
    rows: list[dict[str, object]] = []
    interval_steps = 168
    for idx in range(1, 4):
        scenario_id = f"SYN_LANDLORD_{idx:03d}"
        landlord = f"C_SYN_LANDLORD_{idx:03d}"
        base_step = 40 + idx * 6  #cycles at base, +168, +336 (train), +504 (val), +672 (test)
        n_tenants = 4
        tenant_rents = [round(float(rng.uniform(14_000.0, 26_000.0)), 2) for _ in range(n_tenants)]
        for cycle in range(5):
            cycle_step = base_step + cycle * interval_steps
            if cycle_step > LAST_STEP:
                break
            for t_idx, rent_amt in enumerate(tenant_rents, start=1):
                tenant = f"C_SYN_TENANT_{idx:03d}_{t_idx:02d}"
                rows.append(
                    _make_row(
                        step=cycle_step + (t_idx % 2),
                        txn_type="TRANSFER",
                        amount=rent_amt,
                        name_orig=tenant,
                        name_dest=landlord,
                        is_fraud=0,
                        scenario_id=scenario_id,
                        scenario_type="lookalike",
                        pattern_name="landlord_recurring",
                        is_lookalike=True,
                        held_out_candidate=False,
                    )
                )
    return rows


#Cycle start steps moved so 5 cycles land in train (3), val (1) and test (1)
#Before, the cycles were at 106+168k, which put 3 in train, none in val and only 1 in test
def _generate_payroll_batches(rng: np.random.Generator) -> list[dict[str, object]]:
    """An established employer pays 5 staff in one batch every 168 steps."""
    rows: list[dict[str, object]] = []
    interval_steps = 168
    for idx in range(1, 4):
        scenario_id = f"SYN_PAYROLL_{idx:03d}"
        employer = f"C_SYN_PAYROLL_ORIG_{idx:03d}"
        base_step = 44 + idx * 6  #cycles at base, +168, +336 (train), +504 (val), +672 (test)
        n_employees = 5
        salaries = [round(float(rng.uniform(18_000.0, 35_000.0)), 2) for _ in range(n_employees)]
        for cycle in range(5):
            payday_step = base_step + cycle * interval_steps
            if payday_step > LAST_STEP:
                break
            for emp_idx, salary in enumerate(salaries, start=1):
                employee = f"C_SYN_EMP_{idx:03d}_{emp_idx:02d}"
                rows.append(
                    _make_row(
                        #5 payments land on 2 consecutive steps -> looks like a burst
                        step=payday_step + (emp_idx // 3),
                        txn_type="TRANSFER",
                        amount=salary,
                        name_orig=employer,
                        name_dest=employee,
                        is_fraud=0,
                        scenario_id=scenario_id,
                        scenario_type="lookalike",
                        pattern_name="payroll_or_batch_legit",
                        is_lookalike=True,
                        held_out_candidate=False,
                    )
                )
    return rows


#The look-alike for account_chain fraud. Same A -> B -> C -> D shape and the same forwarding
#ratio, but (a) the middle accounts B and C are established (10-15 normal transactions before the
#first cycle) and (b) the chain repeats every 168 steps, so earlier cycles are history.
#Flagged as a chain by the pattern agent, but legitimate: the reviewer must use history to tell
def _generate_business_forwarding(rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    interval_steps = 168
    for idx in range(1, 4):
        scenario_id = f"SYN_FORWARD_{idx:03d}"
        accounts = [f"C_SYN_FWD_{idx:03d}_{node}" for node in ("A", "B", "C", "D")]
        base_step = 150 + idx * 8  #cycles at base, +168 (train), +336 (train), +504 (val), +672 (test)

        #Make the two relay accounts established before the first cycle
        for relay in accounts[1:3]:
            hist, _ = _history_rows(
                rng,
                account=relay,
                n_txns=int(rng.integers(10, 16)),
                last_step=base_step - 1,
                scenario_id=scenario_id,
                scenario_type="history",
                pattern_name="relay_history",
            )
            rows.extend(hist)

        for cycle in range(5):
            step_cursor = base_step + cycle * interval_steps
            if step_cursor + 6 > LAST_STEP:
                break
            amt_cursor = float(rng.uniform(55_000.0, 160_000.0))
            for hop in range(3):
                if hop > 0:
                    step_cursor += int(rng.integers(1, 4))
                    amt_cursor *= float(rng.uniform(0.95, 0.995))
                rows.append(
                    _make_row(
                        step=step_cursor,
                        txn_type="TRANSFER" if hop < 2 else "CASH_OUT",
                        amount=amt_cursor,
                        name_orig=accounts[hop],
                        name_dest=accounts[hop + 1],
                        is_fraud=0,
                        scenario_id=scenario_id,
                        scenario_type="lookalike",
                        pattern_name="business_forwarding",
                        is_lookalike=True,
                        held_out_candidate=False,
                    )
                )
    return rows


#Also generates the new background accounts and business-forwarding look-alikes
def generate_scenarios(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate deterministic synthetic fraud and look-alike scenarios.

    Returns a DataFrame containing core PaySim columns plus scenario metadata columns,
    sorted deterministically by step and scenario identifiers. Same seed -> identical output.
    """
    rng = np.random.default_rng(seed)

    #NOTE: the order of these calls matters. All generators share one random stream, so
    #reordering them (or adding a call in the middle) changes every number generated after it
    rows: list[dict[str, object]] = []
    rows.extend(_generate_burst_transfers(rng))
    rows.extend(_generate_repeated_recipient_structuring(rng))
    rows.extend(_generate_account_chains(rng))
    rows.extend(_generate_landlord_recurring(rng))
    rows.extend(_generate_payroll_batches(rng))
    rows.extend(_generate_business_forwarding(rng))
    rows.extend(_generate_background_accounts(rng))

    df = pd.DataFrame(rows, columns=ALL_SCENARIO_COLS)
    df = (
        df.sort_values(
            ["step", "scenario_id", "nameOrig", "nameDest", "amount"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    df["step"] = df["step"].astype(np.int64)
    df["amount"] = df["amount"].astype(np.float64)
    df["isFraud"] = df["isFraud"].astype(np.int64)
    df["is_lookalike"] = df["is_lookalike"].astype(bool)
    df["held_out_candidate"] = df["held_out_candidate"].astype(bool)
    return df


#Gives each synthetic row the SAME txn_id that Inspect_and_split.py will give it when it
#merges the core parquet: "SYN_" + the row position in that file, zero-padded to 7 digits
#This is what lets scenario labels (scenario_id, pattern_name, ...) be joined onto the pipeline
#outputs. The rule lives in two places, so a test (test_synthetic_ids_match_split_script)
#checks they agree. The row order of the core parquet must therefore never be changed
def add_synthetic_txn_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "txn_id", [f"SYN_{i:07d}" for i in range(len(out))])
    return out


#The metadata parquet/CSV now include txn_id (see add_synthetic_txn_ids)
def save_scenarios(
    out_parquet: Path = DEFAULT_RAW_PARQUET,
    out_csv: Path | None = DEFAULT_META_CSV,
    out_meta_parquet: Path = DEFAULT_META_PARQUET,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate and persist synthetic scenarios.

    - `out_parquet`: stores core PaySim columns (`step`, `type`, `amount`, `nameOrig`,
      `nameDest`, `isFraud`) so Ziad's pre-split hook in `Inspect_and_split.py` merges
      cleanly without adding sparse metadata columns to raw PaySim rows.
    - `out_meta_parquet` and optional `out_csv`: store the full tagged scenario table
      including `txn_id`, `scenario_id`, `scenario_type`, `pattern_name`, `is_lookalike`, and
      `held_out_candidate`. The labels in here are for evaluation and tests only. Agents must
      never read them (they would reveal which rows are fraud or look-alikes).
    """
    df = generate_scenarios(seed=seed)
    meta_df = add_synthetic_txn_ids(df)

    out_parquet_path = Path(out_parquet)
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df[CORE_PAYSIM_COLS].to_parquet(out_parquet_path, index=False)

    out_meta_path = Path(out_meta_parquet)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(out_meta_path, index=False)

    if out_csv is not None:
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(out_csv_path, index=False)

    n_fraud = int((df["isFraud"] == 1).sum())
    n_lookalike = int(df["is_lookalike"].sum())
    n_held_out = int(df["held_out_candidate"].sum())
    print(
        f"Saved {len(df)} synthetic scenario rows (fraud={n_fraud}, lookalike={n_lookalike}, "
        f"held_out_candidate={n_held_out})"
    )
    print(f"  Core PaySim parquet : {out_parquet_path}")
    print(f"  Metadata parquet    : {out_meta_path}")
    if out_csv is not None:
        print(f"  Metadata CSV        : {Path(out_csv)}")

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic fraud & look-alike PaySim scenarios."
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=DEFAULT_RAW_PARQUET,
        help="Path for PaySim-schema parquet merged before splitting.",
    )
    parser.add_argument(
        "--out-meta-parquet",
        type=Path,
        default=DEFAULT_META_PARQUET,
        help="Path for full scenario metadata parquet.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_META_CSV,
        help="Path for full scenario metadata CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_scenarios(
        out_parquet=args.out_parquet,
        out_csv=args.out_csv,
        out_meta_parquet=args.out_meta_parquet,
        seed=args.seed,
    )