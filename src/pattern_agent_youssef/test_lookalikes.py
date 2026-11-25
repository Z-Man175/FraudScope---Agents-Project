import pandas as pd
from pattern_agent.scenarios import build_dataset, export, SCENARIO_PATTERN
from pattern_agent.detectors import detect_bursts, detect_repeated_recipients, detect_account_chains

df = build_dataset()
kinds = df["kind"].value_counts()
print("rows:", len(df), "| normal:", kinds["normal"], "| fraud:", kinds["fraud"], "| look-alike:", kinds["lookalike"])

flags = detect_bursts(df) + detect_repeated_recipients(df) + detect_account_chains(df)

cases = df[df["kind"] != "normal"].groupby("scenario_id").agg(
    scenario=("scenario", "first"), kind=("kind", "first"), pattern=("pattern", "first"),
    txs=("tx_id", lambda s: set(s)))

def n_flags(row):
    if row["pattern"] == "single_transaction":
        return 0
    return sum(1 for f in flags if f.pattern == row["pattern"] and row["txs"] & set(f.tx_ids))

cases["flags"] = cases.apply(n_flags, axis=1)
cases["flagged"] = cases["flags"] > 0
report = cases.groupby(["kind", "scenario"]).agg(cases=("flagged", "size"),
                                                  flagged=("flagged", "sum"),
                                                  flags=("flags", "sum"))
print("\nDETECTOR FLAGS PER SCENARIO (look-alikes SHOULD be flagged too):")
print(report)

export(df)
print("\nwrote data/transactions.csv and data/scenario_labels.csv")
print("held-out rows (account_chain family):", int(df["held_out"].sum()),
      "| in train/val:", int(df[df["split"] != "test"]["held_out"].sum()))