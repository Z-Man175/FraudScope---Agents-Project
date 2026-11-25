from pattern_agent.schema import prepare
from pattern_agent.scenarios import make_background, split_of
from pattern_agent.detectors import detect_bursts, detect_repeated_recipients, detect_account_chains

df = prepare(make_background())

print("rows:", len(df), "| accounts:", df["nameOrig"].nunique())
print("\nrows per split:")
print(df["step"].map(split_of).value_counts().reindex(["train", "val", "test"]))
print("\ntype share:")
print(df["type"].value_counts(normalize=True).round(2))
print("\namount quantiles:")
print(df["amount"].quantile([0.05, 0.5, 0.95, 0.99]).round(0))
print("\ntxns per account (min / median / max):",
      df.groupby("nameOrig").size().agg(["min", "median", "max"]).tolist())

print("\nBASELINE FLAGS on ordinary data (these are false alarms):")
print("  bursts:", len(detect_bursts(df)))
print("  repeated recipients:", len(detect_repeated_recipients(df)))
print("  chains:", len(detect_account_chains(df)))