import pandas as pd
from pattern_agent.scenarios import build_dataset, TRAIN_END, VAL_END
from pattern_agent.ranking import run_detectors, placeholder_anomaly, rank_candidates, cases_reached

df = build_dataset()
flags = run_detectors(df)
anomaly = placeholder_anomaly(df)

# tune on VALIDATION only
val = df[(df["step"] > TRAIN_END) & (df["step"] <= VAL_END)]
val_flags = [f for f in flags if TRAIN_END < f.window_start <= VAL_END]
val_anom = anomaly.loc[val["tx_id"]]
print("validation:", len(val), "rows |", len(val_flags), "flags |",
      val[val["is_fraud"] == 1]["scenario_id"].nunique(), "fraud cases")

rows = []
for k_a, k_p in [(0, 0), (30, 0), (0, 10), (0, 30), (0, 60), (30, 10), (30, 30), (30, 60), (60, 60)]:
    cands = rank_candidates(val_flags, val_anom, k_a, k_p)
    reached, total = cases_reached(cands, val)
    rows.append(dict(K_a=k_a, K_p=k_p, candidates=len(cands),
                     recall=f"{len(reached)}/{len(total)}", recall_pct=round(len(reached) / len(total), 2)))
print("\nRECALL OF FRAUD CASES ON VALIDATION:")
print(pd.DataFrame(rows).to_string(index=False))

# which patterns does each side find?  (K_a=30, K_p=30)
cands = rank_candidates(val_flags, val_anom, 30, 30)
reached, total = cases_reached(cands, val)
fraud_cases = val[val["is_fraud"] == 1].groupby("scenario_id")["scenario"].first()
print("\nPER PATTERN at K_a=30, K_p=30:")
for scen, n in fraud_cases.value_counts().items():
    got = sum(1 for sid, s in fraud_cases.items() if s == scen and sid in reached)
    print(f"  {scen:18s} {got}/{n}")
print("\nsources in that list:", pd.Series([c["source"] for c in cands]).value_counts().to_dict())