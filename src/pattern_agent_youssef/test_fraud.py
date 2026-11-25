import pandas as pd
from pattern_agent.schema import prepare
from pattern_agent.scenarios import make_background, inject_fraud, split_of
from pattern_agent.detectors import detect_bursts, detect_repeated_recipients, detect_account_chains

df = prepare(inject_fraud(make_background()))
fraud = df[df["is_fraud"] == 1]
print("rows:", len(df), "| fraud rows:", len(fraud), f"({len(fraud) / len(df):.1%})")

flags = detect_bursts(df) + detect_repeated_recipients(df) + detect_account_chains(df)

# which flag pattern we EXPECT to catch each fraud scenario (takeover: none, by design)
EXPECTED = {"mule_burst": "burst", "small_repeated": "repeated_recipient",
            "account_chain": "account_chain", "account_takeover": None}

cases = fraud.groupby("scenario_id").agg(scenario=("scenario", "first"), start=("step", "min"),
                                         txs=("tx_id", lambda s: set(s)))
cases["split"] = cases["start"].map(split_of)

def caught(row):
    want = EXPECTED[row["scenario"]]
    return want is not None and any(f.pattern == want and row["txs"] & set(f.tx_ids) for f in flags)

cases["caught"] = cases.apply(caught, axis=1)
report = cases.groupby(["scenario", "split"]).agg(cases=("caught", "size"), caught=("caught", "sum"))
print("\nFRAUD CASES CAUGHT BY THE EXPECTED DETECTOR:")
print(report)

fraud_ids = set(fraud["tx_id"])
false_alarms = [f for f in flags if not (set(f.tx_ids) & fraud_ids)]
print("\nflags that touch no fraud (false alarms):", len(false_alarms), "of", len(flags))