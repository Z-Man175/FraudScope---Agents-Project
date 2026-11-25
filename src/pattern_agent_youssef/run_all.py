import os
import json
from pattern_agent.scenarios import build_dataset, export, VAL_END
from pattern_agent.ranking import run_detectors, placeholder_anomaly, rank_candidates, cases_reached
from pattern_agent.schema import save_flags

os.makedirs("outputs", exist_ok=True)

# 1. dataset + labels  -> data/transactions.csv (Ziad), data/scenario_labels.csv (Jana, Freedy)
df = build_dataset()
export(df)

# 2. pattern flags     -> outputs/flags.jsonl (Rina)
flags = run_detectors(df)
save_flags(flags, "outputs/flags.jsonl")

# 3. ranked candidates on the TEST period (placeholder anomaly scores until Ziad's arrive)
anomaly = placeholder_anomaly(df)
test = df[df["step"] > VAL_END]
test_flags = [f for f in flags if f.window_start > VAL_END]
cands = rank_candidates(test_flags, anomaly.loc[test["tx_id"]], k_a=30, k_p=30)
with open("outputs/candidates_test.json", "w") as f:
    json.dump(cands, f, indent=2)

reached, total = cases_reached(cands, test)
print("rows:", len(df), "| flags:", len(flags), "| test candidates:", len(cands))
print(f"test fraud cases reached: {len(reached)}/{len(total)}")
print("wrote data/ and outputs/")